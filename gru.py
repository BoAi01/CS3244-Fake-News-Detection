import pandas as pd
import numpy as np

import gensim

import nltk
from nltk.corpus import stopwords

from numpy.random import seed

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GRU, Bidirectional


# Training hyperparameters
USING_EXISTING_MODEL = False
SEED = 0

DIMENSION_COUNT = 128
EPOCHS = 2
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
DROPOUT = 0.2
RECURRENT_DROPOUT = 0.2
SEQUENCE_MAX_LENGTH = 64
BATCH_SIZE = 64

# Input-output file names
# INPUT_FILE_PREFIX = "data/second_dataset/News_dataset_FakeNewsNet/fnn_"
INPUT_FILE_PREFIX = "data/first_dataset/News_dataset/"
OUTPUT_MODEL_NAME = "saved_models/first_dataset/" \
                    "gru_orig_{}dim_{}eps_{}train_{}val_{}test_{}drop_{}recdrop_{}maxlen_{}batch".format(
                            str(DIMENSION_COUNT), str(EPOCHS), str(int((1 - TEST_SPLIT - VALIDATION_SPLIT) * 10)),
                            str(int(VALIDATION_SPLIT * 10)), str(int(TEST_SPLIT * 10)), str(int(DROPOUT * 100)),
                            str(int(RECURRENT_DROPOUT * 100)), str(SEQUENCE_MAX_LENGTH), str(BATCH_SIZE))
VERBOSE = False


seed(SEED)
random.set_seed(SEED)

# Parse data into dataframes

# df_fake_dev = pd.read_csv(INPUT_FILE_PREFIX + "dev_false.csv")
# df_true_dev = pd.read_csv(INPUT_FILE_PREFIX + "dev_true.csv")
# df_fake_test = pd.read_csv(INPUT_FILE_PREFIX + "test_false.csv")
# df_true_test = pd.read_csv(INPUT_FILE_PREFIX + "test_true.csv")
# df_fake_train = pd.read_csv(INPUT_FILE_PREFIX + "train_false.csv")
# df_true_train = pd.read_csv(INPUT_FILE_PREFIX + "train_true.csv")

# df_true = pd.concat([df_true_train, df_true_dev, df_true_test]).reset_index(drop=True)
# df_fake = pd.concat([df_fake_train, df_fake_dev, df_fake_test]).reset_index(drop=True)
df_true = pd.read_csv(INPUT_FILE_PREFIX + "True.csv")
df_fake = pd.read_csv(INPUT_FILE_PREFIX + "Fake.csv")
df_true['isfake'] = 0
df_fake['isfake'] = 1
# df_true = pd.concat([df_true_train, df_true_dev]).reset_index(drop=True)
# df_fake = pd.concat([df_fake_train, df_fake_dev]).reset_index(drop=True)
# df_true['isfake'] = 1
# df_fake['isfake'] = 0
# df_true_test = pd.concat([df_true_test]).reset_index(drop=True)
# df_fake_test = pd.concat([df_fake_test]).reset_index(drop=True)
# df_true_test['isfake'] = 1
# df_fake_test['isfake'] = 0


df = pd.concat([df_true, df_fake]).reset_index(drop=True)
df.drop(columns=['date'], inplace=True)
df['title'] = df['title'].fillna('').astype(str)
df['text'] = df['text'].fillna('').astype(str)
df['text'] = df['text'].replace('(Reuters)', '')
df['original'] = df['title'] + ' ' + df['text']
# df['original'] = df['fullText_based_content']

# df_test = pd.concat([df_true_test, df_fake_test]).reset_index(drop=True)
# df_test.drop(columns=['date'], inplace=True)
# df_test['original'] = df_test['fullText_based_content']


# Preprocess data tokens

# nltk.download('punkt')
# nltk.download("stopwords")
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'to', 'as'])


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_words and len(token) > 3:
            result.append(token)
    return result


df['clean'] = df['original'].apply(preprocess)
# df_test['clean'] = df_test['original'].apply(preprocess)

list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)

total_words = len(list(set(list_of_words)))

df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))
# df_test['clean_joined'] = df_test['clean'].apply(lambda x: " ".join(x))

maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if maxlen < len(tokens):
        maxlen = len(tokens)

if VERBOSE:
    print("The maximum number of words in any document is =", maxlen)


# Tokenize data

x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size=TEST_SPLIT)
# x_train, y_train = df.clean_joined, df.isfake
# x_test, y_test = df_test.clean_joined, df_test.isfake

tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

if VERBOSE:
    print("The encoding for document\n", df.clean_joined[0], "\n is : ", train_sequences[0])

padded_train = pad_sequences(train_sequences, maxlen=SEQUENCE_MAX_LENGTH, padding='post', truncating='post')
padded_test = pad_sequences(test_sequences, maxlen=SEQUENCE_MAX_LENGTH, truncating='post')

if VERBOSE:
    for i, doc in enumerate(padded_train[:2]):
        print("The padded encoding for document", i + 1, " is : ", doc)


# Train model or load existing model

if USING_EXISTING_MODEL:
    model = load_model(OUTPUT_MODEL_NAME)
else:
    model = Sequential()

    # embedding layer
    model.add(Embedding(total_words, output_dim=DIMENSION_COUNT))

    # Bi-Directional RNN and LSTM
    model.add(Bidirectional(GRU(DIMENSION_COUNT, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)))

    # Dense layers
    model.add(Dense(DIMENSION_COUNT, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    y_train = np.asarray(y_train)

    model.fit(padded_train, y_train, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, epochs=EPOCHS)
    model.save(OUTPUT_MODEL_NAME)


# Evaluate model

print("Evaluating model...")
pred = model.predict(padded_test)
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

accuracy = accuracy_score(list(y_test), prediction)
print("Model Accuracy : ", accuracy)

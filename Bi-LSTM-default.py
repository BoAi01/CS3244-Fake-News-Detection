import nltk
nltk.download('punkt')
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re 
import seaborn as sns
import gensim
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model 
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df_train = pd.read_csv("fnn_train.csv")
df_test = pd.read_csv("fnn_test.csv")
df_val = pd.read_csv("fnn_dev.csv")

ratio = 1.0 #0.5/0.1
num_epochs = 2 #4/20

df_train_fake = df_train[df_train["label_fnn"] == "fake"]
df_train_true = df_train[df_train["label_fnn"] == "real"]
df_test_fake = df_test[df_test["label_fnn"] == "fake"]
df_test_true = df_test[df_test["label_fnn"] == "real"]
df_val_fake = df_val[df_val["label_fnn"] == "fake"]
df_val_true = df_val[df_val["label_fnn"] == "real"]
df_train_true['isfake'] = 1
df_test_true['isfake'] = 1
df_val_true['isfake'] = 1
df_train_fake['isfake'] = 0
df_test_fake['isfake'] = 0
df_val_fake['isfake'] = 0
df_train = pd.concat([df_train_true, df_train_fake]).reset_index(drop = True)
df_test = pd.concat([df_test_true, df_test_fake]).reset_index(drop = True)
df_val = pd.concat([df_val_true, df_val_fake]).reset_index(drop = True)

df_train.drop(columns = ['id', 'date'], inplace = True)
df_test.drop(columns = ['id', 'date'], inplace = True)
df_val.drop(columns = ['id', 'date'], inplace = True)
df_train.drop(columns = ['sources', 'label_fnn', 'paragraph_based_content'], inplace = True)
df_test.drop(columns = ['sources', 'label_fnn', 'paragraph_based_content'], inplace = True)
df_val.drop(columns = ['sources', 'label_fnn', 'paragraph_based_content'], inplace = True)
df_train['original'] = df_train['statement'] + ' ' + df_train['speaker'] + ' ' + df_train['fullText_based_content']
df_test['original'] = df_test['statement'] + ' ' + df_test['speaker'] + ' ' + df_test['fullText_based_content']
df_val['original'] = df_val['statement'] + ' ' + df_val['speaker'] + ' ' + df_val['fullText_based_content']

nltk.download("stopwords")
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'to', 'as'])

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_words and len(token) > 3:
            result.append(token)
            
    return result

df_train['clean'] = df_train['original'].apply(preprocess)
df_test['clean'] = df_test['original'].apply(preprocess)
df_val['clean'] = df_val['original'].apply(preprocess)

list_of_words = []
for i in df_train.clean:
    for j in i:
        list_of_words.append(j)
for i in df_test.clean:
    for j in i:
        list_of_words.append(j)
for i in df_val.clean:
    for j in i:
        list_of_words.append(j)

total_words = len(list(set(list_of_words)))

df_train['clean_joined'] = df_train['clean'].apply(lambda x: " ".join(x))
df_test['clean_joined'] = df_test['clean'].apply(lambda x: " ".join(x))
df_val['clean_joined'] = df_val['clean'].apply(lambda x: " ".join(x))

maxlen = -1
for doc in df_train.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
for doc in df_test.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
for doc in df_val.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
print("The maximum number of words in any document is =", maxlen)

x_train = df_train.clean_joined
y_train = df_train.isfake
x_test = df_test.clean_joined
y_test = df_test.isfake
x_val = df_val.clean_joined
y_val = df_val.isfake

tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)
val_sequences = tokenizer.texts_to_sequences(x_val)

padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_val = pad_sequences(val_sequences,maxlen = 40, padding = 'post', truncating = 'post')

# Sequential Model
model = Sequential()

# embeddidng layer
model.add(Embedding(total_words, 128))

# Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(128)))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1,activation= 'sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(padded_train, y_train, batch_size = 128, validation_split = 0.1, epochs = num_epochs)

pred = model.predict(padded_test)
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)
cm = confusion_matrix(list(y_test), prediction)
sns.heatmap(cm, annot = True)
category = { 0: 'Fake News', 1 : "Real News"}


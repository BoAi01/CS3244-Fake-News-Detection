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

df_train = pd.read_csv("/content/drive/My Drive/CS3244/fnn_train.csv")
df_test = pd.read_csv("/content/drive/My Drive/CS3244/fnn_test.csv")
df_val = pd.read_csv("/content/drive/My Drive/CS3244/fnn_dev.csv")

ratio = 1.0 #0.5/0.1
num_epochs = 2 #4/20

df = pd.concat([df_train, df_val, df_test], ignore_index=True, sort=False)  #merge
df = df.sample(frac=1).reset_index(drop=True)   # shuffle
df = df.sample(frac=1).reset_index(drop=True)   # shuffle
df = df[:int(df.shape[0]*ratio)]

df_fake = df[df["label_fnn"] == "fake"]
df_true = df[df["label_fnn"] == "real"]
df_true['isfake'] = 1
df_fake['isfake'] = 0

df = pd.concat([df_true, df_fake]).reset_index(drop = True)

df.drop(columns = ['id', 'date'], inplace = True)
df.drop(columns = ['sources', 'label_fnn', 'paragraph_based_content'], inplace = True)
df['original'] = df['statement'] + ' ' + df['speaker'] + ' ' + df['fullText_based_content']

nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'to', 'as'])

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_words and len(token) > 3:
            result.append(token)
            
    return result

df['clean'] = df['original'].apply(preprocess)

list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)
total_words = len(list(set(list_of_words)))

df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)

x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 40, padding = 'post', truncating = 'post')

# Sequential Model
model = Sequential()

# embeddidng layer
#model.add(Embedding(total_words, output_dim = 128))
model.add(Embedding(total_words, 128))

# Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(128)))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1,activation= 'sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(padded_train, y_train, batch_size = 128,validation_split = 0.1, epochs = num_epochs)

pred = model.predict(padded_test)
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)
cm = confusion_matrix(list(y_test), prediction)
sns.heatmap(cm, annot = True)
category = { 0: 'Fake News', 1 : "Real News"}
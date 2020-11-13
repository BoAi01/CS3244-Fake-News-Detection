## CS3244-fake-news-detection

This repository contains the source code used in implementing the 3 ML models and 3 DL models used in the project, in addition to 2 other ML models that were ultimately not discussed.

## Models included:
### Traditional Machine Learning models
1. Logistic Regression
2. k-Nearest-Neighbours
3. Support Vector Machine

### Deep Learning models
1. Bi-Directional Long Short-Term Memory
2. Gated Recurrent Units
3. Bi-Directional Encoder Representations from Transformers

### Other Machine Learning models
1. Naive Bayes
2. Random Forest

### NOTE & Preparation
    1. document ID for fake news starts from: 21417
    2. install nltk:
       1. pip install nltk
       2. launch interactive python interpreter (usually: python3)
          >>> import nltk
          >>> nltk.download()
       3. The download page should pop-up, and go to "Models" tab, and click "punkt"
    3. You should have the following file structures: "News_dataset/Fake.csv", "News_dataset/True.csv", "input/"
    4. Produce the documents first using get_content.py => input/x: corresponding lines in csv; true and then fake
    5. Specify your setups at the end, under "# TODO"


### DS
    1. xxx_dict.txt: <words(ngram), log-IDF, word count, start position, length>
       1. a particular ngram's start position at xxx_postings.txt and the length of its posting
    2. xxx_postings.txt: a list of {docId_1: (normalized) tf_1, docId_2: tf_2...} if the term appears in he docId


### RECOURSES
    The functions are adapted from
    1. CS3245 Information Retrieval  [for tf-idf]                  @Li Wei, Zhang Xiaoyu
    2. public github repo            [for GloVe]                   @https://github.com/stanfordnlp/GloVe

    Stop words: https://www.ranks.nl/stopwords

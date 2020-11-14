## CS3244-fake-news-detection

This repository contains the source code for a student project, fake news detection, in module CS3244 Machine Learning, at the School of Computing (SoC), National University of Singapore (NUS). The project aims to find efficient and accurate ways of classifying fake news based only on body text of new articles. Multiple traditional ML and DL models are benchmarked on two datasets, [ISOT](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php), and [FakeNewsNet](https://ieee-dataport.org/open-access/fnid-fake-news-inference-dataset#files). We achieved decent accuracies on both datasets (>95% and >70% test acc respectively), and further investigation is conducted to analyse some key observations. See [paper](https://drive.google.com/file/d/1KhcO22HTHitYyAhgO58Z-VYYUfzr-JLC/view?usp=sharing) for details. 

Apart from 3 ML and 3 DL models reported in the papper, 2 other ML models are also included. 

## Models included:
### Traditional Machine Learning models
1. Logistic Regression
2. k-Nearest-Neighbours (kNN)
3. Support Vector Machine (SVM)

### Deep Learning models
1. Bi-Directional Long Short-Term Memory (Bi-LSTM)
2. Gated Recurrent Units (GRU)
3. Bi-Directional Encoder Representations from Transformers (BERT)

### Other Machine Learning models
1. Naive Bayes
2. Random Forest

### Table of Results (Test Acc)
| Dataset | Logistic Regression | kNN |  SVM  |  Bi-LSTM  | GRU | BERT |
| ------- | ------------------- | --- | ----- | --------- | --- | ---- |
| ISOT | 98.64 | 95.61 | 99.37 | 98.81 | 97.23 | **99.96** |
| FakeNewsNet | **71.48** | 63.31 | 69.76 | 61.82 | 62.94 | 71.21 |


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

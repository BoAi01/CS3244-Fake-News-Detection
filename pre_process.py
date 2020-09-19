import re
import os
import nltk
import sys
import getopt
import math
import json

from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import *
from nltk import bigrams
from nltk import trigrams


class Processor:
    """
    Encapsulates the methods to build dictionary file and posting file.
    """

    def __init__(self, in_dir, dictionary, posting, n):
        """
        Args:
            in_dir:     [str] work path for process
            dictionary: [str] dictionary filename
            posting:    [str] posting filename
        """
        self.work_dir = in_dir                  # the path to dataset
        self.dict_filename = dictionary         # the dictionary file to write
        self.posting_filename = posting         # the posting file to write
        self.ngram_level = n                    # the specification of x-gram

        self.corpoa = []                        # the corpoa of all document ids
        self.keywords = {}                      # the dictionary of terms and corresponding postings
        self.tfs = {}                           # the normalized tf score
        self.idfs = {}                          # the idf scores
        
        self.stop_words = set()
        self.stemmer = PorterStemmer()          # the stem function

    def process_document(self, filename):
        """
        Processes words in the document for the dictionary and postings.
        This methods tokenizes and normalizes words in the document,
        and adds docId and the normalized term weights into corresponding postings. 
        Args:
            filename:   [str] the filename of Reuters data set (equivalent as docId)
        """
        # tokenizing and normalizing words
        f = open(self.work_dir+'/'+filename, 'r').read().lower()
        f = remove_character(f)
        sentence_set = [sent for sent in sent_tokenize(f)]
        words = []
        for sent in sentence_set:
            words += [self.stemmer.stem(word.lower())
                         for word in word_tokenize(sent)]
        words = remove_punctuation(words)

        # remove stop-words   
        filtered_words = []
        for w in words:
            if w not in self.stop_words:
                filtered_words.append(w)
        # print(len(words))
        # print(len(filtered_words))
        
        # get ngrams
        if (self.ngram_level == 2):
            filtered_words = list(bigrams(filtered_words))
        elif (self.ngram_level == 3):
            filtered_words = list(trigrams(filtered_words))
        
        # recording processed terms and their weights
        for word in filtered_words:
            self.keywords.setdefault(word, [])
            self.keywords[word].append(int(filename))

        self.tfs[filename] = Counter(filtered_words)                                        # raw tf
        doc_length = math.sqrt(
            sum([(1+math.log(self.tfs[filename][w], 10))**2 for w in self.tfs[filename]]))
        for word in self.tfs[filename]:
            self.tfs[filename][word] = (
                1+math.log(self.tfs[filename][word], 10))/(doc_length)                      # normalized tf_weight

    def process_postings(self):
        """
        Processes the posting lists into desired format and calculated idfs.
        Specifically, the format of postings is the mapping as <docId, tf_weight>.
        """
        for w in self.keywords:
            self.keywords[w] = list(set(self.keywords[w]))
            self.idfs[w] = math.log(len(self.corpoa)/len(self.keywords[w]), 10)
            self.keywords[w] = dict(zip([docId for docId in self.keywords[w]], 
                                        [self.tfs[str(docId)][w] for docId in self.keywords[w]]))
            self.keywords[w] = dict(sorted(self.keywords[w].items(), key=lambda k: k[1], reverse=True))

    def write_to_disk(self):
        """
        Outputs the indexed dictionary file and postings file.
        """
        d = open(self.dict_filename, 'w')
        p = open(self.posting_filename, 'w')
        self.corpoa = sorted(self.corpoa)
        p_prev = p.tell()
        for w in self.keywords:
            p_prev = p.tell()
            d.write(str(w)+' '+str(self.idfs[w])+' ' +
                    str(len(self.keywords[w]))+' '+str(p_prev))
            p.write(json.dumps(self.keywords[w])+'\n')
            d.write(f' {p.tell()-p_prev}\n')
        d.close()
        p.close()

    def load_stop_words(self):
        with open("stop_words.txt","r") as sp:
            self.stop_words= set(sp.readline().split())


    def build_dic_and_posting(self):
        """
        Calls relevant methods to complete the indexing.
        """
        self.load_stop_words()
        for root, dirs, files in os.walk(self.work_dir):
            self.len_document = len(files)
            for i, file in enumerate(files):
                if '.' in file:
                    continue
                self.process_document(file)
                self.corpoa.append(int(file))
        self.process_postings()
        self.write_to_disk()


def remove_punctuation(words):
    """
    Removes the punctuation characters, such as ','.
    Args:
        words: [list] a list of words to be processed
    Return:
        words: [list] words without any punctuations
    """
    Pattern = re.compile(u'[a-z|0-9]+')
    words=[i for i in words if Pattern.search(i)]
    return words

def remove_character(data):
    """
    Removes the trivial characters, including "/", "\s", and "\n".
    Args:
        data: [str] a raw string to be processed
    Return:
        data: [str] a string without trivial characters
    """
    data = re.sub('\n|\s+|/', ' ', data)
    return data

def build_index(in_dir, out_dict, out_postings, N):
    print('indexing...')

    processor = Processor(in_dir, out_dict, out_postings, N)
    processor.build_dic_and_posting()

# TODO
N = 1
input_directory = "input/"
output_file_dictionary = str(N) + "_dict"
output_file_postings = str(N) + "_postings"

build_index(input_directory, output_file_dictionary, output_file_postings, N)
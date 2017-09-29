#!/usr/bin/env python

"""
Description: Text handling utilities.

Author: Dylan Elliott
Date: 09/07/2017

"""
import numpy as np
import gensim
import logging
import time

NULL = '__'

# takes a list of strings, splits them by white space, lowercases all words,
# filters out stop words, optionally performs stemming and returns tokenized
# texts
def filterTok(documents, stoplist, stem=False):
    # filter text using the stoplist
    texts = [[word for word in document.lower().split() if word not in stoplist]
            for document in documents]

    # remove words that appear once
    """
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
            for text in texts]
    """

    return texts

# needs tokenized texts returned by filterDocs function
def buildDict(texts):
    # build dictionary
    dictionary = gensim.corpora.Dictionary(texts)

    return dictionary

# needs tokenized texts returned by filterDocs() function
def docs2Bow(texts):
    # build dictionary
    dictionary = gensim.corpora.Dictionary(texts)

    # initialize bag of words array
    n = len(texts)
    d = len(dictionary)
    X = np.zeros((n, d))

    corpus = [dictionary.doc2bow(text) for text in texts]

    for i, vec in enumerate(corpus):
        for idx, val in vec:
            X[i, idx] = val

    # log normalize the vectors
    for i in range(n):
        x = X[None, i, :]
        x = np.log(1 + x)
        x = x/np.max(x)
        X[i, :] = x

    return X

# needs tokenized texts returned by filterDocs() funstion
def docs2Tfidf(texts):
    # build dictionary
    dictionary = gensim.corpora.Dictionary(texts)

    # initialize bag of words array
    n = len(texts)
    d = len(dictionary)
    X = np.zeros((n, d))

    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = gensim.models.TfidfModel(corpus)

    for i, vec in enumerate(corpus):
        for idx, val in tfidf[vec]:
            X[i, idx] = val

    return X

def docs2DictIndex(texts):
    # insert a null character so that it takes the zeroith spot of the vocab
    texts.insert(0, [NULL])

    # build dictionary
    dictionary = gensim.corpora.Dictionary(texts)

    texts.remove([NULL])

    reverse_dictionary = dict()

    for key, value in dictionary.iteritems():
        reverse_dictionary[value] = key

    # initialize bag of words array
    n = len(texts)
    d = max([len(text) for text in texts])

    X = np.zeros((n, d))

    for i, text in enumerate(texts):
        for j, word in enumerate(text):
            try:
                X[i, j] = reverse_dictionary[word]
            except:
                X[i, j] = reverse_dictionary[NULL]

    return X

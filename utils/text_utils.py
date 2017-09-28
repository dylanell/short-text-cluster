#!/usr/bin/env python

"""
Description: Text handling utilities.

Author: Dylan Elliott
Date: 09/07/2017

"""
import numpy as np
import gensim

def buildDict(documents, stoplist, stem=False):
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

    # build dictionary
    dictionary = gensim.corpora.Dictionary(texts)

    return texts, dictionary

# log normalized bag-of-words
def docs2Bow(texts, dictionary):
    # initialize bag of words array
    n = len(texts)
    d = len(dictionary)
    X = np.zeros((n, d))

    corpus = [dictionary.doc2bow(text) for text in texts]

    for i in range(n):
        for idx, val in corpus[i]:
            X[i, idx] = val

    # log normalize the vectors
    for i in range(n):
        x = X[None, i, :]
        x = np.log(1 + x)
        x = x/np.max(x)
        X[i, :] = x
    
    return X

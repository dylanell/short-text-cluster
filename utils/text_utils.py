#!/usr/bin/env python

"""
Description: Text handling utilities.

Author: Dylan Elliott
Date: 09/07/2017

"""
import numpy as np
import gensim

def buildDict(documents, stoplist, stem=False):
    print(stoplist)

    texts = [[word for word in document.lower().split() if word not in stoplist]
            for document in documents]

    print(texts)

    # remove words that appear once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
            for text in texts]

    print texts

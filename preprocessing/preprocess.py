#!/usr/bin/env python

"""
Description:

Author: Dylan Elliott

Date: 09/28/2017

"""

import numpy as np

from utils import text_utils as tu

qtype_label_d = {'ABBR': 0, 'ENTY': 1, 'DESC': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5}
QTYPE_N_TR = 5452
QTYPE_N_TE = 500

# convert the qtype dataset to log normalized bag of words vectors
def qtype2Bow(train_fp, test_fp, sw_fp):
    documents = []
    train_Y = np.zeros((QTYPE_N_TR, 1))
    test_Y = np.zeros((QTYPE_N_TE, 1))

    # extract samples and labels from training data
    for i, line in enumerate(train_fp):
        # separate the label and sample from the line
        sample = ' '.join(line.split()[1:])
        label = qtype_label_d[line.split(':')[0]]

        # append sample to the sentences corpus
        documents.append(sample)

        # add label to train labels array
        train_Y[i, 0] = label

    # extract samples and labels from testing data
    for i, line in enumerate(test_fp):
        # separate the label and sample from the line
        sample = ' '.join(line.split()[1:])
        label = qtype_label_d[line.split(':')[0]]

        # append sample to the sentences corpus
        documents.append(sample)

        # add label to train labels array
        test_Y[i, 0] = label

    # get stopwords
    stopwords = []
    for i, line in enumerate(sw_fp):
        stopwords.append(line.split()[0])

    # build the dictionary
    texts, dictionary = tu.buildDict(documents, stopwords, stem=True)

    train_X = tu.docs2Bow(texts[:QTYPE_N_TR], dictionary)
    test_X = tu.docs2Bow(texts[-QTYPE_N_TE:], dictionary)

    return train_X, train_Y, test_X, test_Y

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

def qtypeSplit(train_fp, test_fp):
    train_X = []
    test_X = []
    train_Y = np.zeros((QTYPE_N_TR, 1))
    test_Y = np.zeros((QTYPE_N_TE, 1))

    # extract samples and labels from training data
    for i, line in enumerate(train_fp):
        # separate the label and sample from the line
        sample = ' '.join(line.split()[1:])
        label = qtype_label_d[line.split(':')[0]]

        # append sample to the sentences corpus
        train_X.append(sample)

        # add label to train labels array
        train_Y[i, 0] = label

    # extract samples and labels from testing data
    for i, line in enumerate(test_fp):
        # separate the label and sample from the line
        sample = ' '.join(line.split()[1:])
        label = qtype_label_d[line.split(':')[0]]

        # append sample to the sentences corpus
        test_X.append(sample)

        # add label to train labels array
        test_Y[i, 0] = label

    return train_X, train_Y, test_X, test_Y


# convert the qtype dataset to log normalized bag of words vectors
def qtype2Bow(train_fp, test_fp, sw_fp):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = qtypeSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterDocs(documents, stoplist, stem=False)

    # build the dictionary
    dictionary = tu.buildDict(texts)

    train_X = tu.docs2Bow(texts[:QTYPE_N_TR], dictionary)
    test_X = tu.docs2Bow(texts[-QTYPE_N_TE:], dictionary)

    return train_X, train_Y, test_X, test_Y

def qtype2WordVec(train_fp, test_fp, sw_fp):
    print('word 2 vec')

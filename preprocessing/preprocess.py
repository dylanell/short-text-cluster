#!/usr/bin/env python

"""
Description:

Author: Dylan Elliott

Date: 09/28/2017

"""

import numpy as np
import time
import pickle

from utils import text_utils as tu
from utils import data_utils as du

# q-type dataset params
qtype_label_d = {'ABBR': 0, 'ENTY': 1, 'DESC': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5}
QTYPE_N_TR = 5452
QTYPE_N_TE = 500

# stackoverflow dataset params
STK_N = 20000
STK_N_TR = 16000
STK_N_TE = 4000

# ag-news dataset params
AG_N = 127600
AG_N_TR = 120000
AG_N_TE = 7600

# q-type dataset has samples and labels in one file, so split them into
# separate arrays (already divided by training and testing sets)
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

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)

    return train_X, train_Y, test_X, test_Y

# convert the q-type dataset to list of lists
def qtype2Texts(train_fp, test_fp, sw_fp, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = qtypeSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    X = tu.filterTok(documents, stoplist, stem=stem)

    train_X = X[:QTYPE_N_TR]
    test_X = X[-QTYPE_N_TE:]

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


# convert the qtype dataset to log normalized bag of words vectors
def qtype2Bow(train_fp, test_fp, sw_fp, prune_dict=5000, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = qtypeSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    dictionary = tu.buildDict(texts, prune_at=prune_dict)

    X = tu.docs2Bow(texts, dictionary)

    del texts

    train_X = X[:QTYPE_N_TR]
    test_X = X[-QTYPE_N_TE:]

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y

# convert the qtype dataset to tf-idf vectors
def qtype2Tfidf(train_fp, test_fp, sw_fp, prune_dict=5000, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = qtypeSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    dictionary = tu.buildDict(texts, prune_at=prune_dict)

    X = tu.docs2Tfidf(texts, dictionary)

    del texts

    train_X = X[:QTYPE_N_TR]
    test_X = X[-QTYPE_N_TE:]

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


def qtype2DictIndex(train_fp, test_fp, sw_fp, prune_dict=5000, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = qtypeSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    dictionary = tu.buildDict(texts, prune_at=prune_dict)

    X, null_idx = tu.docs2DictIndex(texts, dictionary)

    del texts

    train_X = X[:QTYPE_N_TR].astype(np.float32)
    test_X = X[-QTYPE_N_TE:].astype(np.float32)

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y, dictionary, null_idx


def qtype2Embed(train_fp, test_fp, sw_fp, embed_dir, vtype='average', stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = qtypeSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    if (vtype=='average'):
        X = tu.docs2AvgEmbed(texts, embed_dir)
    elif (vtype=='representative'):
        X = tu.docs2RepEmbed(texts, embed_dir)
    elif (vtype=='attention'):
        X = tu.docs2WeightEmbed(texts, embed_dir)
    elif (vtype=='gensim'):
        X = tu.docs2Vector(texts)

    del texts

    train_X = X[:QTYPE_N_TR].astype(np.float32)
    test_X = X[-QTYPE_N_TE:].astype(np.float32)

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


# q-type dataset has samples and labels in one file, so split them into
# separate arrays (already divided by training and testing sets)
def stkSplit(sample_fp, label_fp):
    X = []
    Y = np.zeros((STK_N, 1), dtype=np.int32)

    # extract samples and labels from training data
    for i, line in enumerate(sample_fp):
        # separate the label and sample from the line
        sample = ' '.join(line.split())

        # append sample to the sentences corpus
        X.append(sample)

    # extract samples and labels from training data
    for i, line in enumerate(label_fp):
        # separate the label and sample from the line
        label = int(' '.join(line.split()))

        # append label to the label corpus (make 0 indexed)
        Y[i, :] = label - 1

    # randomly shuffle the dataset with a constant seed so it is identical
    # every time
    np.random.seed(635)
    r = np.random.permutation(np.arange(STK_N))

    shuff_X = []
    for i in r:
        shuff_X.append(X[i])

    Y = Y[r]
    X = shuff_X

    # split the dataset into the training and testing sets
    train_X = X[:STK_N_TR]
    test_X = X[-STK_N_TE:]
    train_Y = Y[:STK_N_TR]
    test_Y = Y[-STK_N_TE:]

    # set fp's back to beginning of file
    sample_fp.seek(0)
    label_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


def stk2Texts(sample_fp, label_fp, sw_fp, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = stkSplit(sample_fp, label_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    X = tu.filterTok(documents, stoplist, stem=stem)

    train_X = X[:STK_N_TR]
    test_X = X[-STK_N_TE:]

    # set fp's back to beginning of file
    sample_fp.seek(0)
    label_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y

# convert the stackoverflow dataset to log-normalized bag of words vectors
def stk2Bow(sample_fp, label_fp, sw_fp, prune_dict=5000, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = stkSplit(sample_fp, label_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    dictionary = tu.buildDict(texts, prune_at=prune_dict)

    X = tu.docs2Bow(texts, dictionary)

    del texts

    train_X = X[:STK_N_TR]
    test_X = X[-STK_N_TE:]

    # set fp's back to beginning of file
    sample_fp.seek(0)
    label_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


# convert the stackoverflow dataset to tf-idf vectors
def stk2Tfidf(sample_fp, label_fp, sw_fp, prune_dict=5000, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = stkSplit(sample_fp, label_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    dictionary = tu.buildDict(texts, prune_at=prune_dict)

    X = tu.docs2Tfidf(texts, dictionary)

    del texts

    train_X = X[:STK_N_TR]
    test_X = X[-STK_N_TE:]

    # set fp's back to beginning of file
    sample_fp.seek(0)
    label_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


def stk2DictIndex(sample_fp, label_fp, sw_fp, prune_dict=5000, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = stkSplit(sample_fp, label_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    dictionary = tu.buildDict(texts, prune_at=prune_dict)

    X, null_idx = tu.docs2DictIndex(texts, dictionary)

    del texts

    train_X = X[:STK_N_TR].astype(np.float32)
    test_X = X[-STK_N_TE:].astype(np.float32)

    # set fp's back to beginning of file
    sample_fp.seek(0)
    label_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y, dictionary, null_idx


def stk2Embed(sample_fp, label_fp, sw_fp, embed_dir, vtype='average', stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = stkSplit(sample_fp, label_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    if (vtype=='average'):
        X = tu.docs2AvgEmbed(texts, embed_dir)
    elif (vtype=='representative'):
        X = tu.docs2RepEmbed(texts, embed_dir)
    elif (vtype=='attention'):
        X = tu.docs2WeightEmbed(texts, embed_dir)
    elif (vtype=='gensim'):
        X = tu.docs2Vector(texts)

    del texts

    train_X = X[:STK_N_TR].astype(np.float32)
    test_X = X[-STK_N_TE:].astype(np.float32)

    # set fp's back to beginning of file
    sample_fp.seek(0)
    label_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


# ag-news dataset has samples and labels in one file, so split them into
# separate arrays (already divided by training and testing sets)
def agnewsSplit(train_fp, test_fp):
    train_X = []
    test_X = []
    train_Y = np.zeros((AG_N_TR, 1))
    test_Y = np.zeros((AG_N_TE, 1))

    # extract samples and labels from training data
    for i, line in enumerate(train_fp):
        # separate the label and sample from the line
        line_split = line.split(',')

        label = int(line_split[0].split('"')[1])
        sample = line_split[1].strip('"')

        # append sample to the sentences corpus
        train_X.append(sample)

        # add label to train labels array (make llabel 0 indexed)
        train_Y[i, 0] = label - 1


    # extract samples and labels from testing data
    for i, line in enumerate(test_fp):
        # separate the label and sample from the line
        line_split = line.split('"')
        line_split.remove('')
        line_split.remove(',')

        label = int(line_split[0])
        sample = line_split[1]

        # append sample to the sentences corpus
        test_X.append(sample)

        # add label to train labels array
        test_Y[i, 0] = label - 1

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


# convert the ag-news dataset to list of lists
def agnews2Texts(train_fp, test_fp, sw_fp, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = agnewsSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    X = tu.filterTok(documents, stoplist, stem=stem)

    train_X = X[:AG_N_TR]
    test_X = X[-AG_N_TE:]

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y

# convert the ag-news dataset to log normalized bag of words vectors
def agnews2Bow(train_fp, test_fp, sw_fp, prune_dict=5000, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = agnewsSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    dictionary = tu.buildDict(texts, prune_at=prune_dict)

    X = tu.docs2Bow(texts, dictionary)

    del texts

    train_X = X[:AG_N_TR]
    test_X = X[-AG_N_TE:]

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


# convert the ag-news dataset to tf-idf vectors
def agnews2Tfidf(train_fp, test_fp, sw_fp, prune_dict=5000, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = agnewsSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    dictionary = tu.buildDict(texts, prune_at=prune_dict)

    X = tu.docs2Tfidf(texts, dictionary)

    del texts

    train_X = X[:AG_N_TR]
    test_X = X[-AG_N_TE:]

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


def agnews2DictIndex(train_fp, test_fp, sw_fp, prune_dict=5000, stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = agnewsSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    dictionary = tu.buildDict(texts, prune_at=prune_dict)

    X, null_idx = tu.docs2DictIndex(texts, dictionary)

    del texts

    train_X = X[:AG_N_TR].astype(np.float32)
    test_X = X[-AG_N_TE:].astype(np.float32)

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y, dictionary, null_idx


# uses google word vectors to create sentence embeddings
def agnews2Embed(train_fp, test_fp, sw_fp, embed_dir, vtype='average', stem=False):
    # split the data into samples and labels
    train_D, train_Y, test_D, test_Y = agnewsSplit(train_fp, test_fp)

    documents = train_D + test_D

    # get stopwords
    stoplist = []
    for i, line in enumerate(sw_fp):
        stoplist.append(line.split()[0])

    texts = tu.filterTok(documents, stoplist, stem=stem)

    del documents

    if (vtype=='average'):
        X = tu.docs2AvgEmbed(texts, embed_dir)
    elif (vtype=='representative'):
        X = tu.docs2RepEmbed(texts, embed_dir)
    elif (vtype=='attention'):
        X = tu.docs2WeightEmbed(texts, embed_dir)
    elif (vtype=='gensim'):
        X = tu.docs2Vector(texts)

    del texts

    train_X = X[:AG_N_TR].astype(np.float32)
    test_X = X[-AG_N_TE:].astype(np.float32)

    # set fp's back to beginning of file
    train_fp.seek(0)
    test_fp.seek(0)
    sw_fp.seek(0)

    return train_X, train_Y, test_X, test_Y


def qtype2LPP(train_X_fn, train_Y_fn, train_T_fn, test_X_fn,
              test_Y_fn, test_T_fn, embed_dir=None, k=10, t=1e0, l=2,
              binary=False, batch_size=1000, metric='l2'):

    train_X = np.loadtxt(train_X_fn, delimiter=' ')
    train_Y = np.loadtxt(train_Y_fn, delimiter=' ').astype(np.int32)

    with open(train_T_fn, 'r') as fp:
        train_T = pickle.load(fp)

    test_X = np.loadtxt(test_X_fn, delimiter=' ')
    test_Y = np.loadtxt(test_Y_fn, delimiter=' ').astype(np.int32)

    with open(test_T_fn, 'r') as fp:
        test_T = pickle.load(fp)

    X = np.concatenate([train_X, test_X], axis=0)
    texts = train_T + test_T

    if (metric=='wmd'):
        X_lpp = tu.docs2LPP(X, texts, embed_dir=embed_dir, k=k, t=t, l=l,
                            binary=binary, batch_size=batch_size)
    else:
        X_lpp = du.embedLPP(X, k=k, t=t, l=l, metric=metric,
                            binary=binary, batch_size=batch_size)

    train_X = X_lpp[:QTYPE_N_TR].astype(np.float32)
    test_X = X_lpp[-QTYPE_N_TE:].astype(np.float32)

    return train_X, train_Y, test_X, test_Y


def stk2LPP(train_X_fn, train_Y_fn, train_T_fn, test_X_fn,
              test_Y_fn, test_T_fn, embed_dir=None, k=10, t=1e0, l=2,
              binary=False, batch_size=1000, metric='l2'):

    train_X = np.loadtxt(train_X_fn, delimiter=' ')
    train_Y = np.loadtxt(train_Y_fn, delimiter=' ').astype(np.int32)

    with open(train_T_fn, 'r') as fp:
        train_T = pickle.load(fp)

    test_X = np.loadtxt(test_X_fn, delimiter=' ')
    test_Y = np.loadtxt(test_Y_fn, delimiter=' ').astype(np.int32)

    with open(test_T_fn, 'r') as fp:
        test_T = pickle.load(fp)

    X = np.concatenate([train_X, test_X], axis=0)
    texts = train_T + test_T

    if (metric=='wmd'):
        X_lpp = tu.docs2LPP(X, texts, embed_dir=embed_dir, k=k, t=t, l=l,
                            binary=binary, batch_size=batch_size)
    else:
        X_lpp = du.embedLPP(X, k=k, t=t, l=l, metric=metric,
                            binary=binary, batch_size=batch_size)

    train_X = X_lpp[:STK_N_TR].astype(np.float32)
    test_X = X_lpp[-STK_N_TE:].astype(np.float32)

    return train_X, train_Y, test_X, test_Y


def agnews2LPP(train_X_fn, train_Y_fn, train_T_fn, test_X_fn,
              test_Y_fn, test_T_fn, embed_dir=None, k=10, t=1e0, l=2,
              binary=False, batch_size=1000, metric='l2'):

    train_X = np.loadtxt(train_X_fn, delimiter=' ')
    train_Y = np.loadtxt(train_Y_fn, delimiter=' ').astype(np.int32)

    with open(train_T_fn, 'r') as fp:
        train_T = pickle.load(fp)

    test_X = np.loadtxt(test_X_fn, delimiter=' ')
    test_Y = np.loadtxt(test_Y_fn, delimiter=' ').astype(np.int32)

    with open(test_T_fn, 'r') as fp:
        test_T = pickle.load(fp)

    X = np.concatenate([train_X, test_X], axis=0)
    texts = train_T + test_T

    if (metric=='wmd'):
        X_lpp = tu.docs2LPP(X, texts, embed_dir=embed_dir, k=k, t=t, l=l,
                            binary=binary, batch_size=batch_size)
    else:
        X_lpp = du.embedLPP(X, k=k, t=t, l=l, metric=metric,
                            binary=binary, batch_size=batch_size)

    train_X = X_lpp[:AG_N_TR].astype(np.float32)
    test_X = X_lpp[-AG_N_TE:].astype(np.float32)

    return train_X, train_Y, test_X, test_Y

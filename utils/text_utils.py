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
def buildDict(texts, prune_at=5000):
    # TODO: keep most frequent words
    for text in texts:
        text.append(NULL)

    # insert a null character so that it takes the zeroith spot of the vocab
    texts.insert(0, [NULL])

    # build dictionary
    dictionary = gensim.corpora.Dictionary(texts)

    dictionary.filter_extremes(no_below=0, no_above=1.0,
                               keep_n=prune_at, keep_tokens=NULL)

    texts.remove([NULL])

    return dictionary

# needs tokenized texts returned by filterDocs() function
# and prebuilt dictionary
def docs2Bow(texts, dictionary):
    # initialize bag of words array
    n = len(texts)
    d = len(dictionary)
    X = np.zeros((n, d))

    corpus = [dictionary.doc2bow(text, allow_update=False) for text in texts]

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
# and prebuilt dictionary
def docs2Tfidf(texts, dictionary):
    # initialize bag of words array
    n = len(texts)
    d = len(dictionary)
    X = np.zeros((n, d))

    corpus = [dictionary.doc2bow(text, allow_update=False) for text in texts]

    tfidf = gensim.models.TfidfModel(corpus)

    for i, vec in enumerate(corpus):
        for idx, val in tfidf[vec]:
            X[i, idx] = val

    return X

def docs2DictIndex(texts, dictionary):
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
        # fill remaining gaps with NULL character
        X[i, j:] = reverse_dictionary[NULL]

    return X

# input: list of tokenized texts as lists
def docs2AvgEmbed(texts, embed_dir):
    print('average word embedding')
    n = len(texts)
    d = 300
    X = np.zeros((n, d))

    model = gensim.models.KeyedVectors.load_word2vec_format(embed_dir,
                                                            binary=True)

    for i, text in enumerate(texts):
        s = len(text)
        W = np.zeros((s, d))
        # get the sentence embedding matrix W
        for j, word in enumerate(text):
            try:
                W[j, :] = model.wv[word]
            except:
                W[j, :] = np.random.normal(0, 1, (1, d))

        # calculate the average embedding vector
        X[i, :] = np.mean(W, axis=0)

    del model

    return X

# input: list of tokenized texts as lists
def docs2RepEmbed(texts, embed_dir):
    print('representative embedding')
    n = len(texts)
    d = 300
    X = np.zeros((n, d))

    model = gensim.models.KeyedVectors.load_word2vec_format(embed_dir,
                                                            binary=True)

    for i, text in enumerate(texts):
        s = len(text)
        W = np.zeros((s, d))
        # get the sentence embedding matrix W
        for j, word in enumerate(text):
            try:
                W[j, :] = model.wv[word]
            except:
                W[j, :] = np.random.normal(0, 1, (1, d))

        from sklearn import metrics
        pw = metrics.pairwise.pairwise_distances(W)

        rep = np.argmin(np.sum(pw, axis=0))

        # calculate the representative for the embedding
        X[i, :] = W[rep, :]

    del model

    return X

# input: list of tokenized texts as lists
def docs2WeightEmbed(texts, embed_dir, temp=5e-3):
    def softmax(x, t):
        e_x = np.exp(x/t)
        return e_x / e_x.sum()

    print('weighted word embedding')
    n = len(texts)
    d = 300
    X = np.zeros((n, d))

    model = gensim.models.KeyedVectors.load_word2vec_format(embed_dir,
                                                            binary=True)

    for i, text in enumerate(texts):
        s = len(text)
        W = np.zeros((s, d))
        # get the sentence embedding matrix W
        for j, word in enumerate(text):
            try:
                W[j, :] = model.wv[word]
            except:
                W[j, :] = np.random.normal(0, 1, (1, d))

        from sklearn import metrics
        pw = metrics.pairwise.pairwise_distances(W)

        vec = 1/(1 + np.sum(pw, axis=0))

        att = softmax(vec, temp).reshape((vec.shape[0], 1))

        # calculate the attention for the embedding
        X[i, :] = np.matmul(att.T, W)

    del model

    return X

# convert documents (T) to locality-preserving-projections using
# word-movers-distance as a distance metric and a pretrained sentence to
# vector dataset (X) as the vector constraints
def docs2LPP(X, T, embed_dir, k=10, t=1e2, l=2, binary=False):
    from scipy import linalg

    # get number of samples 'm'
    m = X.shape[0]
    n = X.shape[1]

    Y = np.zeros((m, l))
    N = np.zeros((m, m))

    # center X
    X = X - np.mean(X, axis=0)

    # open the google word2vec model
    model = gensim.models.KeyedVectors.load_word2vec_format(embed_dir,
                                                            binary=True)

    PW = np.zeros((m, m))
    # construct pairwise distance matrix from word movers distance
    for i in range(m):
        for j in range(m):
            PW[i, j] = model.wmdistance(T[i], T[j])

    del model

    # find  k nearest for each row
    for i in range(m):
        order = np.argsort(PW[i, :])

        k_nearest = order[1:k+1]

        # get neighbors
        N[i, k_nearest] = 1

    # N = 1 if i is j's neighbor or j is i's neighbor
    N = 1 * ((N.T + N) > 0)

    W = np.exp(-1 * (PW**2) / t) * N

    # create a diagonal matrix from the weights
    D = np.diag(np.sum(W, axis=0))

    # create a laplacian matrix
    L = D - W

    # solve the generalized eigenvalue equation X^TLXa = lam * X^TDXa
    # where N = X^TDX and M = X^TLX and E = N^-1M so we have
    # Ea = lam*a ----> get eigenvalues of E
    N = np.matmul(X.T, np.matmul(D, X))

    M = np.matmul(X.T, np.matmul(L, X))

    # generalized eigenval equation now: Ma = lam * Na

    lam, v = linalg.eigh(M, b=N)

    order = np.argsort(lam)

    bot_l = order[:l]

    evecs = v[:, bot_l]

    # project X to new space in R^l
    Y = np.matmul(X, evecs)

    if(binary):
        # get the median of each projected sample
        median = np.median(Y, axis=1).reshape((m, 1))

        # if the element of a sample is greater than its median, it becomes 1,
        # otherwise it is 0
        Y = 1*((Y - median) >= 0)

    return Y

#!/usr/bin/env python

"""
Description: Text handling utilities.

Author: Dylan Elliott
Date: 09/07/2017

"""
import numpy as np
import gensim
from collections import namedtuple
import logging
import sys
import copy

import data_utils as du

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
    #for i in range(n):
    #    x = X[None, i, :]
    #    x = np.log(1 + x)
    #    x = x/np.max(x)
    #    X[i, :] = x

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

    null_idx = reverse_dictionary[NULL]

    for i, text in enumerate(texts):
        for j, word in enumerate(text):
            try:
                X[i, j] = reverse_dictionary[word]
            except:
                X[i, j] = null_idx
        # fill remaining gaps with NULL character
        X[i, j:] = null_idx

    return X, null_idx

# input: list of tokenized texts as lists
def docs2AvgEmbed(texts, embed_dir):
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
def docs2WeightEmbed(texts, embed_dir=None, temp=1e-2):
    def softmax(x, t):
        e_x = np.exp(x/t)
        return e_x / e_x.sum()

    n = len(texts)
    d = 300
    X = np.zeros((n, d))

    if (embed_dir):
        print('\t>> using google vecs')
        model = gensim.models.KeyedVectors.load_word2vec_format(embed_dir,
                                                                binary=True)
    else:
        print('\t>> using custom vecs')
        model = gensim.models.Word2Vec(texts, size=300, min_count=0)
        model.train(texts, total_examples=model.corpus_count, epochs=20)

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

        # normalize pw if its not 0
        if (np.linalg.norm(pw)):
            pw = pw/np.linalg.norm(pw)

        vec = 1/(1 + np.sum(pw, axis=0))

        att = softmax(vec, temp).reshape((vec.shape[0], 1))

        # calculate the attention for the embedding
        X[i, :] = np.matmul(att.T, W)

    del model

    return X

# input: list of tokenized texts as lists
def docs2Vector(texts):
    from gensim.models.doc2vec import LabeledSentence

    n = len(texts)
    d = 300
    X = np.zeros((n, d))

    # we need to create a corpus of tagged documents for the doc2vec model
    docs = []
    for i, text in enumerate(texts):
        tag = [str(i)]
        docs.append(LabeledSentence(text, tag))

    model = gensim.models.Doc2Vec(docs, size=300, min_count=0)
    model.train(docs, total_examples=model.corpus_count, epochs=20)

    # infer all doc vectors and save to array
    for i, text in enumerate(texts):
        # get the doc2vec embedding fro this sentence
        X[i, :] = model.infer_vector(text)

    del model

    return X

# convert documents (T) to locality-preserving-projections using
# word-movers-distance as a distance metric and a pretrained sentence to
# vector dataset (X) as the vector constraints
def docs2LPP(X, texts, embed_dir=None, k=10,
             t=1e0, l=2, binary=False, batch_size=1000):
    from scipy import linalg

    if (embed_dir):
        print('\t>> using google vecs')
        model = gensim.models.KeyedVectors.load_word2vec_format(embed_dir,
                                                                binary=True)
    else:
        print('\t>> using custom vecs')
        model = gensim.models.Word2Vec(texts, size=300, min_count=0)
        model.train(texts, total_examples=model.corpus_count, epochs=20)

    # center X
    X = X - np.mean(X, axis=0)

    # project X to a lower dimesnional space using PCA
    sigma = np.matmul(X.T, X)           # calculate covariance of dataset
    lam, v = linalg.eigh(sigma)         # get the eigenvals/vectors
    order = np.argsort(lam)[::-1]       # order vals highest to lowest
    lam = lam[order]
    v = v[:, order]
    tot_var = np.sum(lam)               # get the total variance of the data
    cum_var = np.cumsum(lam)            # get the cumulative variance
    capture = cum_var/tot_var           # determine dim to capture 98% of var
    dim = np.sum(1 * (capture <= 0.98))

    evecs = v[:, :dim]

    W_pca = copy.deepcopy(evecs)

    # project X with W_pca to get LPP's
    X_pca = np.matmul(X, W_pca)

    batch_X, _, _ = du.getBatch(X_pca, X_pca, batch_size)

    # get number of samples 'm'
    m = batch_X.shape[0]
    n = batch_X.shape[1]

    Y = np.zeros((m, l))
    Ne = np.zeros((m, m))

    PW = np.zeros((m, m))
    # construct pairwise distance matrix from word movers distance
    for i in range(m):
        for j in range(m):
            PW[i, j] = model.wmdistance(texts[i], texts[j])

        # report conversion progress
        progress = format(float(i)/float(m)*100.0, '.2f')
        sys.stdout.write('\r\t>> LPP progress: ' + str(progress) + '%')
        sys.stdout.flush()

    sys.stdout.write('\r\t>> LPP progress: 100%\n\n')
    sys.stdout.flush()
    del model

    # normalize pw if its not 0
    if (np.linalg.norm(PW)):
        PW = PW/np.linalg.norm(PW)

    # find  k nearest for each row
    for i in range(m):
        order = np.argsort(PW[i, :])

        k_nearest = order[1:k+1]

        # get neighbors
        Ne[i, k_nearest] = 1

    # Ne = 1 if i is j's neighbor or j is i's neighbor
    Ne = 1 * ((Ne.T + Ne) > 0)

    W = np.exp(-1 * PW / t) * Ne

    # create a diagonal matrix from the weights
    D = np.diag(np.sum(W, axis=0))

    # create a laplacian matrix
    L = D - W

    # solve the generalized eigenvalue equation X^TLXa = lam * X^TDXa
    # where N = X^TDX and M = X^TLX and E = N^-1M so we have
    # Ea = lam*a ----> get eigenvalues of E
    N = np.matmul(batch_X.T, np.matmul(D, batch_X))

    M = np.matmul(batch_X.T, np.matmul(L, batch_X))

    # generalized eigenval equation now: Ma = lam * Na
    lam, v = linalg.eigh(M, b=N)

    order = np.argsort(lam)

    bot_l = order[:l]

    evecs = v[:, bot_l]

    W_lpp = copy.deepcopy(evecs)

    # project X to new space in R^l
    Y = np.matmul(X, np.matmul(W_pca, W_lpp))

    if(binary):
        # get the median of each projected sample
        median = np.median(Y, axis=1).reshape((Y.shape[0], 1))

        # if the element of a sample is greater than its median, it becomes 1,
        # otherwise it is 0
        Y = 1*((Y - median) >= 0)

    return Y

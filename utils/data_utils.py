#!/usr/bin/env python

"""
Description: Data handling/processing utilities.

Author: Dylan Elliott
Date: 09/07/2017

"""
import numpy as np
import copy

from scipy import linalg

def embedTSNE(X, l=2, binary=False):
    from sklearn.manifold import TSNE
    Y = TSNE(n_components=2).fit_transform(X)

    if(binary):
        # get the median of each projected sample
        median = np.median(Y, axis=1).reshape((m, 1))

        # if the element of a sample is greater than its median, it becomes 1,
        # otherwise it is 0
        Y = 1*((Y - median) >= 0)

    return Y

def embedPCA(X, l=2, binary=False):
    from sklearn.decomposition import PCA
    Y = PCA(n_components=2).fit_transform(X)

    if(binary):
        # get the median of each projected sample
        median = np.median(Y, axis=1).reshape((m, 1))

        # if the element of a sample is greater than its median, it becomes 1,
        # otherwise it is 0
        Y = 1*((Y - median) >= 0)

    return Y

# create projections of samples in X using locality-pereserving-constraints
# (LPP) in R^l
# src: https://papers.nips.cc/paper/2359-locality-preserving-projections.pdf
# TODO: can be very expensive to run, add workaround for singular N
# (psuedo-inverse?)
def embedLPP(X, k=5, t=1e1, l=2, metric='l2', binary=False):
    # get number of samples 'm'
    m = X.shape[0]
    n = np.prod(X.shape[1:])

    # make X 2D by flattening all axis other than axis 0
    X = X.reshape((-1, n))

    # center X
    X = X - np.mean(X, axis=0)

    Y = np.zeros((m, l))
    N = np.zeros((m, m))

    from sklearn import metrics
    try:
        PW = metrics.pairwise.pairwise_distances(X, metric=metric)
    except:
        print('unknown distance metric: %s, using default' % metric)
        PW = metrics.pairwise.pairwise_distances(X, metric='l2')

    # find  k nearest for each row
    for i in range(m):
        order = np.argsort(PW[i, :])
        if (metric=='cosine'):
            k_nearest = order[::-1][1:k+1]
        else:
            k_nearest = order[1:k+1]

            # get neighbors
            N[i, k_nearest] = 1

    # N = 1 if i is j's neighbor or j is i's neighbor
    N = 1 * ((N.T + N) > 0)

    if (metric != 'cosine'):
        W = np.exp(-1 * PW / t) * N
    else:
        W = (PW + 1)/2

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

def getBatch(X, Y, batch_size):
    batchX = 0 * copy.deepcopy(X[:batch_size])
    batchY = 0 * copy.deepcopy(Y[:batch_size])

    nx = batchX.shape[0]

    selected = []
    indices = []
    for i in range(batch_size):
        r = np.random.randint(0, nx)
        while(r in selected):
             r = np.random.randint(0, nx)
        batchX[i] = X[r]
        batchY[i] = Y[r]
        indices.append(r)

    return batchX, batchY, indices

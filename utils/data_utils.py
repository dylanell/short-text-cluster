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
def embedLPP(X, k=5, t=1e0, l=2, metric='l2', binary=False, batch_size=1000):
    # get number of samples 'm'
    m = X.shape[0]
    n = np.prod(X.shape[1:])

    # make X 2D by flattening all axis other than axis 0
    X = X.reshape((-1, n))

    # center X
    X = X - np.mean(X, axis=0)

    # project X to a lower dimensional space using PCA
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

    batch_X, _, _ = getBatch(X_pca, X_pca, batch_size)

    # get number of samples 'm'
    m = batch_X.shape[0]
    n = batch_X.shape[1]

    Y = np.zeros((m, l))
    Ne = np.zeros((m, m))

    from sklearn import metrics
    try:
        PW = metrics.pairwise.pairwise_distances(batch_X, metric=metric)
    except:
        print('unknown distance metric: %s, using default' % metric)
        PW = metrics.pairwise.pairwise_distances(batch_X, metric='l2')

    # normalize pw if its not 0
    if (np.linalg.norm(PW)):
        PW = PW/np.linalg.norm(PW)

    # find  k nearest for each row
    for i in range(m):
        order = np.argsort(PW[i, :])
        if (metric=='cosine'):
            k_nearest = order[::-1][1:k+1]
        else:
            k_nearest = order[1:k+1]

            # get neighbors
            Ne[i, k_nearest] = 1

    # N = 1 if i is j's neighbor or j is i's neighbor
    Ne = 1 * ((Ne.T + Ne) > 0)

    if (metric != 'cosine'):
        W = np.exp(-1 * PW / t) * Ne
    else:
        W = (PW + 1)/2

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

def getBatch(X, Y, batch_size):
    if(batch_size > X.shape[0]):
        print('batch_size larger than dataset; setting to dataset size')
        batch_size = X.shape[0]
        return X, Y, range(X.shape[0])

    batchX = 0 * copy.deepcopy(X[:batch_size])
    batchY = 0 * copy.deepcopy(Y[:batch_size])

    nx = X.shape[0]
    indices = []
    for i in range(batch_size):
        r = np.random.randint(0, nx)
        while(r in indices):
             r = np.random.randint(0, nx)
        indices.append(r)
        batchX[i] = X[r]
        batchY[i] = Y[r]

    return batchX, batchY, indices


# converts a vector of numeric class labels to one hot encoding
def label2OneHot(labels):
    # store number of labels
    n = labels.size

    # list all possible class values
    vals = np.unique(labels)

    # fill one hot array
    oneHot  = (vals == labels)*1.0

    return oneHot

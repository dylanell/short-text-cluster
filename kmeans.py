#!/usr/bin/env python

import sys

import numpy as np
from sklearn import cluster
from sklearn import metrics

if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 3):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./kmeans.py <samples_fp> <labels_fp> <K>')
        sys.exit()

    # get the training data
    X = np.loadtxt(sys.argv[1])
    Y = np.loadtxt(sys.argv[2])
    K = int(sys.argv[3])

    n, d = X.shape

    # must be 1D for scikit learn metrics
    Y = Y.reshape((n, ))

    # create a kmeans model
    model = cluster.KMeans(n_clusters=K, init='k-means++',
                           max_iter=100, n_init=10)

    # convert sparse matrix X to dense array if using hierarchical clustering
    Y_pred = model.fit_predict(X)

    # get the ARI score of the clustering
    score = metrics.adjusted_rand_score(Y, Y_pred)

    print score

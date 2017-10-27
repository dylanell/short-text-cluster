#!/usr/bin/env python

import sys

import numpy as np
import scipy
from sklearn import cluster
from sklearn import metrics

from utils import cluster_utils as cu

if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=1000, suppress=True)

    # retrieve command line args
    if (len(sys.argv) < 4):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./cluster.py <algorithm> <samples_fp> <labels_fp> <K>')
        sys.exit()

    # get args
    algo = sys.argv[1]
    X = np.loadtxt(sys.argv[2])
    T = np.loadtxt(sys.argv[3]).astype(np.int32)
    K = int(sys.argv[4])

    n, d = X.shape

    # must be 1D for scikit learn metrics
    T = T.reshape((n, ))

    if (algo == 'dbscan'):
        print('dbscan')
        exit()
        # create a dbscan model
        #model = cluster.DBSCAN(n_clusters=K, init='k-means++',
        #                       max_iter=300, n_init=100)
    elif (algo == 'kmeans'):
        # create a kmeans model
        model = cluster.KMeans(n_clusters=K, init='k-means++',
                               max_iter=300, n_init=100)
    else:
        print('[INFO] unknown clustering algorithm %s; using kmeans' % algo)
        # create a kmeans model
        model = cluster.KMeans(n_clusters=K, init='k-means++',
                               max_iter=300, n_init=100)

    # convert sparse matrix X to dense array if using hierarchical clustering
    L = model.fit_predict(X)

    print 'Cluster Labels:', L[0:10]
    print 'Ground Truth Labels:', T[0:10]

    # map truth labels to labels in our clustering
    mapped_T, G = cu.mapping(T, L, X)

    print 'Ground Truth Mapped:', mapped_T[0:10]
    print 'Mapping: ', G

    C = cu.buildContingency(mapped_T, L)

    print C

    F1_score = cu.computeF1(C)

    # get the NMI score of the clustering
    score = metrics.normalized_mutual_info_score(T, L)

    print 'NMI:', score
    print 'F1:', F1_score

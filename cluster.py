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
    if (len(sys.argv) < 5):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./cluster.py <samples_fp> <labels_fp> <K> <out_dir>')
        sys.exit()

    # get args
    X = np.loadtxt(sys.argv[1])
    T = np.loadtxt(sys.argv[2]).astype(np.int32)
    K = int(sys.argv[3])

    n, d = X.shape

    # must be 1D for scikit learn metrics
    T = T.reshape((n, ))

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

    f_score = cu.FMeasure(C)

    # get the NMI score of the clustering
    ami_score = metrics.adjusted_mutual_info_score(T, L)

    print 'AMI:', ami_score
    print 'F:', f_score

    if ((not np.isnan(ami_score)) and (not np.isnan(f_score))):
        out_dir = sys.argv[4]

        ami = np.loadtxt(out_dir + 'ami_log.txt')
        ami = np.append(ami, ami_score)
        np.savetxt(out_dir + 'ami_log.txt', ami)

        f = np.loadtxt(out_dir + 'f_log.txt')
        f = np.append(f, f_score)
        np.savetxt(out_dir + 'f_log.txt', f)
    else:
        print('NAN found, not a good clustering')

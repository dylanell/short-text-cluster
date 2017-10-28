#!/usr/bin/env python

"""
Description: Clustering and evaluation utilities.

Author: Dylan Elliott
Date: 09/07/2017

"""
import numpy as np
import copy
import numpy as np
import scipy
from sklearn import cluster
from sklearn import metrics

import matplotlib.pyplot as plt

# src: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
def arrayRowIntersection(a,b):
   tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
   return a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]

# uses scikit learn to solve the minimum assignment problem to construct
# an optimal mapping from true labels T to current labels L
# "for each truth label, what is the equivalent label in our cluster labels?"
# this is what mapping G(.) does. cluster_label = G(truth_label)
def mapping(T, L, X, one_hot=False):
    classes = np.unique(T)
    K = classes.shape[0]

    C = np.zeros((K, K))

    if (one_hot):
        # convert L from one hot to numerical labels
        L = np.argmax(L, axis=1)

    n = T.shape[0]

    for i in range(K):
        for j in range(K):
            Xi = X[np.where(T == i)]
            Xj = X[np.where(L == j)]

            Xi_and_Xj = arrayRowIntersection(Xi, Xj)

            len_Xi = Xi.shape[0]
            len_Xj = Xj.shape[0]
            len_Xi_and_Xj = Xi_and_Xj.shape[0]

            dis = len_Xi + len_Xj - 2*(len_Xi_and_Xj)

            C[i, j] = dis

    # solve the linear assignment problem on C with scipy
    G = scipy.optimize.linear_sum_assignment(C)

    # map each truth label in T to the correspondng cluster label in mapped_T
    mapped_T = np.zeros_like(T)

    for i in range(n):
        mapped_T[i] = G[-1][int(T[i])]

    return mapped_T, G[-1]

# builds a contingency table from the truth labels. IMPORTANT: truth labels
# must be mapped to the corresponding predicted labels by solving the
# minimum sum assignment problem using the mapping() function above
def buildContingency(mapped_T, L):
    K = len(np.unique(L))
    C = np.zeros((K, K))

    n = L.shape[0]

    for i in range(n):
        r = L[i]
        c = mapped_T[i]

        C[r, c] += 1

    return C

# computes f1 score of a clustering assignment. IMPORTANT: C must be the
# contingency table with rows as predicted cluster labels and columns as
# the mapped truth labels using the minimum assignment problem, hence C will
# already be organized such that the correct assignments are matched by
# row = col pairs
def FMeasure(C):
    P = np.diag(C)/np.sum(C, axis=1)
    R = np.diag(C)/np.sum(C, axis=0)

    F = (2 * P * R)/(P + R)

    F1_score = np.sum(F)/len(F)

    return F1_score

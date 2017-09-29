#!/usr/bin/env python

"""
Description: Numerical data handling utilities.

Author: Dylan Elliott
Date: 09/07/2017

"""
import numpy as np
import copy

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

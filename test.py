#!/usr/bin/env python

import numpy as np
from utils import data_utils as du
from utils import text_utils as tu
import matplotlib.pyplot as plt
import gensim
import pickle

if __name__ == '__main__':
    embed_dir = '/home/dylan/rpi/thesis/GoogleNews-vectors-negative300.bin'


    X = np.loadtxt('datasets/q-type/train_lpp.dat', delimiter=' ')
    L = np.loadtxt('datasets/q-type/train_label.dat',
                    delimiter=' ').astype(np.int32)

    with open('datasets/q-type/train_texts.dat', 'r') as fp:
        T = pickle.load(fp)

    """

    X = np.loadtxt('datasets/stk-ovflw/train_sentvec.dat', delimiter=' ')
    L = np.loadtxt('datasets/stk-ovflw/train_label.dat',
                    delimiter=' ').astype(np.int32)

    with open('datasets/stk-ovflw/train_texts.dat', 'r') as fp:
        T = pickle.load(fp)


    X = np.loadtxt('datasets/ag-news/train_sentvec.dat', delimiter=' ')
    L = np.loadtxt('datasets/ag-news/train_label.dat',
                    delimiter=' ').astype(np.int32)

    with open('datasets/ag-news/train_texts.dat', 'r') as fp:
        T = pickle.load(fp)
    """

    num_samples = X.shape[0]

    Y = du.embedPCA(X, l=2, binary=False)

    #Y = du.embedTSNE(X, l=2, binary=False)

    #Y = du.embedLPP(X, k=10, t=1e0, l=2, metric='l2',
    #                binary=False, batch_size=2000)

    #Y = tu.docs2LPP(X, T, embed_dir, k=10, t=1e0, l=2,
    #                binary=False, batch_size=1000)

    colors = ['r', 'g', 'b', 'y', 'k', 'c']

    plt.figure(0)
    for i in range(num_samples):
        try:
            plt.scatter(Y[i, 0], Y[i, 1], color=colors[L[i]], s=3)
        except:
            plt.scatter(Y[i, 0], Y[i, 1], s=3)
    plt.show()

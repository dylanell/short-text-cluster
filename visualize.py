#!/usr/bin/env python

import numpy as np
from utils import data_utils as du
from utils import text_utils as tu
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 6):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./visualize.py <embedding> <vectors> <labels> ' \
              '<title> <plt_file>')
        sys.exit()

    embedding = sys.argv[1]
    vector_fn = sys.argv[2]
    label_fn = sys.argv[3]
    plt_title = sys.argv[4]
    plt_fn = sys.argv[5]

    X = np.loadtxt(vector_fn, delimiter=' ')
    L = np.loadtxt(label_fn, delimiter=' ').astype(np.int32)

    n, d = X.shape

    # number of classes
    K = len(np.unique(L))

    print('Input Shape: (%d, %d)' % (n, d))

    if (embedding == 'tsne'):
        print('Embedding Model: T-SNE')
        Y = du.embedTSNE(X, l=2, binary=False)
    elif (embedding == 'lpp'):
        print('Embedding Model: LPP')
        Y = du.embedLPP(X, k=15, t=2e0, l=2, metric='l2', binary=False)
    else:
        print('Embedding Model: PCA')
        Y = du.embedPCA(X, l=2, binary=False)

    n, e = Y.shape

    print('Embedded Shape: (%d, %d)' % (n, e))

    plt.scatter(Y[:, 0], Y[:, 1], c=L, s=10)
    plt.title(plt_title)

    plt.savefig(plt_fn)


    #Y = du.embedLPP(X, k=15, t=2e0, l=50, metric='l2', binary=True)

    #Y = du.embedPCA(Y, l=2, binary=False)

    #Y = tu.docs2LPP(X, T, embed_dir, k=10, t=1e0, l=2,
    #                binary=False, batch_size=1000)

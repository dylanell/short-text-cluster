#!/usr/bin/env python

"""
Description:

Author: Dylan Elliott

Date: 09/28/2017

"""

import sys
import numpy as np

from preprocessing.preprocess import qtype2Bow, qtype2DictIndex

if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 4):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./process_qtype.py <train_src> <test_src> ' \
              '<out_dir>')
        sys.exit()

    # open the datafiles
    try:
        train_fp = open(sys.argv[1], 'r')
    except Exception as e:
        raise e
    try:
        test_fp = open(sys.argv[2], 'r')
    except Exception as e:
        raise e

    # open the stop words file
    try:
        sw_fp = open('preprocessing/stopwords.txt', 'r')
    except Exception as e:
        raise e

    # get the out directory
    out_dir = sys.argv[3]

    print('converting to bag-of-words')
    # convert question type data to bag of words vectors
    train_X, train_Y, test_X, test_Y = qtype2Bow(train_fp, test_fp, sw_fp)

    # save processed data to the out directory
    np.savetxt(out_dir + 'train_lnbow.dat', train_X)
    np.savetxt(out_dir + 'test_lnbow.dat', test_X)

    del train_X
    del test_X


    # convert question type data to trained word2Vec embeddings
    print('converting to dictionary indices')
    train_X, _, test_X, _ = qtype2DictIndex(train_fp, test_fp, sw_fp)

    # save processed data to the out directory
    np.savetxt(out_dir + 'train_indices.dat', train_X)
    np.savetxt(out_dir + 'test_indices.dat', test_X)

    # save label files
    np.savetxt(out_dir + 'train_label.dat', train_Y)
    np.savetxt(out_dir + 'test_label.dat', test_Y)

    # close original data files
    train_fp.close()
    test_fp.close()

    # close the stopwords file
    sw_fp.close()

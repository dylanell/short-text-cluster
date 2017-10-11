#!/usr/bin/env python

"""
Description:

Author: Dylan Elliott

Date: 09/28/2017

"""

import sys
import numpy as np

from preprocessing.preprocess import qtype2Bow, qtype2DictIndex, qtype2Tfidf
from preprocessing.preprocess import qtype2Embed

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

    print('converting to log-normalized bag-of-words')
    # convert question type data to bag of words vectors
    train_X, train_Y, test_X, test_Y = qtype2Bow(train_fp, test_fp,
                                                 sw_fp, prune_dict=2000)

    # save processed data to the out directory
    np.savetxt(out_dir + 'train_lnbow.dat', train_X)
    np.savetxt(out_dir + 'test_lnbow.dat', test_X)

    del train_X
    del test_X


    print('converting to tf-idf')
    # convert question type data to bag of words vectors
    train_X, train_Y, test_X, test_Y = qtype2Tfidf(train_fp, test_fp,
                                                   sw_fp, prune_dict=2000)

    # save processed data to the out directory
    np.savetxt(out_dir + 'train_tfidf.dat', train_X)
    np.savetxt(out_dir + 'test_tfidf.dat', test_X)

    del train_X
    del test_X

    # convert question type data to indexes from a dictionary ---
    # used for joint training of the word embeddings
    print('converting to dictionary indices')
    train_X, train_Y, test_X, test_Y = qtype2DictIndex(train_fp, test_fp,
                                            sw_fp, prune_dict=2000)

    # save processed data to the out directory
    np.savetxt(out_dir + 'train_indices.dat', train_X)
    np.savetxt(out_dir + 'test_indices.dat', test_X)

    print('converting to sentence vector')
    embed_dir = '/home/dylan/rpi/thesis/GoogleNews-vectors-negative300.bin'
    train_X, train_Y, test_X, test_Y = qtype2Embed(train_fp, test_fp,
                                                   sw_fp, embed_dir,
                                                   vtype='attention')

    # save processed data to the out directory
    np.savetxt(out_dir + 'train_sentvec.dat', train_X)
    np.savetxt(out_dir + 'test_sentvec.dat', test_X)

    # save label files
    np.savetxt(out_dir + 'train_label.dat', train_Y)
    np.savetxt(out_dir + 'test_label.dat', test_Y)

    # close original data files
    train_fp.close()
    test_fp.close()

    # close the stopwords file
    sw_fp.close()

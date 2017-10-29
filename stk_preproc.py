#!/usr/bin/env python

"""
Description:

Author: Dylan Elliott

Date: 09/28/2017

"""

import sys
import numpy as np
import pickle

from preprocessing.preprocess import stk2Texts
from preprocessing.preprocess import stk2Bow
from preprocessing.preprocess import stk2Tfidf
from preprocessing.preprocess import stk2DictIndex
from preprocessing.preprocess import stk2Embed
from preprocessing.preprocess import stk2LPP

from utils import data_utils as du

if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 4):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./process_qtype.py <sample_src> <label_src> ' \
              '<out_dir>')
        sys.exit()

    # open the datafiles
    try:
        sample_fp = open(sys.argv[1], 'r')
    except Exception as e:
        raise e
    try:
        label_fp = open(sys.argv[2], 'r')
    except Exception as e:
        raise e

    # open the stop words file
    try:
        sw_fp = open('preprocessing/stopwords.txt', 'r')
    except Exception as e:
        raise e

    # get the out directory
    out_dir = sys.argv[3]

    # get the vocabulary (dictionary) size
    vocab_limit = 200000
    vector_limit = 20000
    stem = False

    print('[INFO] encoding filtered texts')
    # convert stackoverflow data to bag of words vectors
    train_X, train_Y, test_X, test_Y = stk2Texts(sample_fp, label_fp,
                                                 sw_fp, stem=stem)

    # save processed data to the out directory
    try:
        with open(out_dir + 'train_texts.dat', 'w') as fp:
            pickle.dump(train_X, fp)
        with open(out_dir + 'test_texts.dat', 'w') as fp:
            pickle.dump(test_X, fp)
    except Exception as e:
        raise e

    del train_X
    del test_X

    print('[INFO] encoding bag-of-words')
    # convert stackoverflow data to bag of words vectors
    train_X, train_Y, test_X, test_Y = stk2Bow(sample_fp, label_fp, sw_fp,
                                            prune_dict=vector_limit, stem=stem)

    # save processed data to the out directory
    np.savetxt(out_dir + 'train_bow.dat', train_X)
    np.savetxt(out_dir + 'test_bow.dat', test_X)

    del train_X
    del test_X

    print('[INFO] encoding tf-idf')
    # convert stackoverflow data to bag of words vectors
    train_X, train_Y, test_X, test_Y = stk2Tfidf(sample_fp, label_fp, sw_fp,
                                            prune_dict=vector_limit, stem=stem)

    # save processed data to the out directory
    np.savetxt(out_dir + 'train_tfidf.dat', train_X)
    np.savetxt(out_dir + 'test_tfidf.dat', test_X)

    del train_X
    del test_X

    # convert stackoverflow data to indexes from a dictionary
    # used for joint training of the word embeddings
    print('[INFO] encoding dictionary indices')
    train_X, train_Y, test_X, \
    test_Y, dictionary, null_idx = stk2DictIndex(sample_fp, label_fp, sw_fp,
                                            prune_dict=vocab_limit, stem=stem)


    dictionary.save(out_dir + 'vocabulary.dat')

    # save processed data to the out directory
    np.savetxt(out_dir + 'train_indices.dat', train_X)
    np.savetxt(out_dir + 'test_indices.dat', test_X)

    # save label files
    np.savetxt(out_dir + 'train_label.dat', train_Y)
    np.savetxt(out_dir + 'test_label.dat', test_Y)

    # close original data files
    sample_fp.close()
    label_fp.close()

    # close the stopwords file
    sw_fp.close()

    X = np.concatenate([train_X, test_X], axis=0)
    avg_len = round(np.mean(np.sum(1 * (X != null_idx), axis=1)))

    # print output statistics
    K = len(np.unique(train_Y))
    N_train = train_X.shape[0]
    N_test = test_X.shape[0]
    N = N_train + N_test
    max_len = train_X.shape[1]
    V = len(dictionary)
    print('\n[INFO] Dataset: StackOverflow')
    print('[INFO] Num. Classes: %d' % K)
    print('[INFO] Num. Samples: %d' % N)
    print('[INFO] Num. Train: %d' % N_train)
    print('[INFO] Num. Test: %d' % N_test)
    print('[INFO] Max Sent. Length: %d' % max_len)
    print('[INFO] Avg. Sent. Length: %d' % avg_len)
    print('[INFO] Vocab Size: %d' % V)

    # sent vec encoding
    #print('converting to sentence vector')
    #train_X, train_Y, test_X, test_Y = stk2Embed(sample_fp, label_fp,
    #                                               sw_fp, embed_dir=None,
    #                                               vtype='attention')

    # save processed data to the out directory
    #np.savetxt(out_dir + 'train_sentvec.dat', train_X)
    #np.savetxt(out_dir + 'test_sentvec.dat', test_X)

    # LPP encoding
    # save training sample hashes using LPP's
    #print('converting to binary hashes')
    #train_X = np.loadtxt(out_dir + 'train_bow.dat', delimiter=' ')
    #train_H = du.embedLPP(train_X, k=15, t=2e0, l=70, metric='l2', binary=True)
    #np.savetxt(out_dir + 'train_hash.dat', train_H)

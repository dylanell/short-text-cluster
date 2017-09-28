#!/usr/bin/env python

"""
Description:

Author: Dylan Elliott

Date: 09/28/2017

"""

import sys
import numpy as np

from preprocessing.preprocess import qtype2Bow

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

    # convert question type data to bag of words vectors
    train_X, train_Y, text_X, text_Y = qtype2Bow(train_fp, test_fp, sw_fp)

    # save processed data to the out directory

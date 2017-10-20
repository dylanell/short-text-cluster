#!/usr/bin/env python

"""

Description:

Author: Dylan Elliott
Date: 08/03/2017

"""

import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import time
import gensim

from utils.data_utils import getBatch
from utils.data_utils import label2OneHot

from neural.models import TextCNN

NULL = '__'

if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 3):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./vanilla_cnn.py <data_dir> <out_dir>')
        sys.exit()

    # get the data
    data_dir = sys.argv[1]
    train_X = np.loadtxt(data_dir + 'train_indices.dat')
    train_Y = np.loadtxt(data_dir + 'train_label.dat')
    test_X = np.loadtxt(data_dir + 'test_indices.dat')
    test_Y = np.loadtxt(data_dir + 'test_label.dat')

    vocab = gensim.corpora.Dictionary.load(data_dir + 'vocabulary.dat')
    vocab_len = len(vocab)

    train_OH = label2OneHot(np.reshape(train_Y, (train_Y.shape[0], 1)))
    test_OH = label2OneHot(np.reshape(test_Y, (test_Y.shape[0], 1)))

    # find the index of the NULL word '__'
    rev_vocab = dict()
    for key, value in vocab.iteritems():
        rev_vocab[value] = key
    NULL_IDX = rev_vocab[NULL]
    del vocab
    del rev_vocab

    n = len(train_X)
    d = 64

    K = len(np.unique(train_Y))

    # for collecting error values
    E = []

    """ hyper parameters """
    eta = 1e-3
    num_maps = 100
    flat_dim = num_maps * 3
    latent_dim = 50

    """ runtime parameters """
    num_iter = 1000
    plot_per = 100
    batch_size = 8
    plot = 1

    """ model parameters """
    emb_dims = [vocab_len, d]
    filt_dims = [[3, num_maps], [4, num_maps], [5, num_maps]]
    fc_dims = [latent_dim, K]

    """ tensorflow ops """
    # dropout probability
    keep_prob = tf.placeholder(tf.float32)

    # shape [batch_size, max_seq_len]
    inputs = tf.placeholder(tf.int32, [None, None])

    # shape [batch_size, max_seq_len]
    targets = tf.placeholder(tf.float32, [None, K])

    model = TextCNN(emb_dims, filt_dims, fc_dims, inputs,
                targets, NULL_IDX, kp=keep_prob, eta=eta)

    init = tf.global_variables_initializer()

    """ Tensorflow Session """
    with tf.Session() as sess:
        sess.run(init)

        inst_E = []

        # train the network
        for i in range(num_iter):
            # grab a randomly chosen batch from the training data
            batch_X, batch_Y, indices = getBatch(train_X, train_OH, batch_size)
            train_feed = {inputs: batch_X, targets: batch_Y, keep_prob: 0.5}

            #probe = sess.run(model.probe, train_feed)
            #print probe
            #exit()

            # run a step of the autoencoder trainer
            sess.run(model.optimize, train_feed)

            # get loss of this sample
            e = sess.run(model.loss, train_feed)
            inst_E.append(e)

            # if we have hit a plotting period and we are in plotting
            # mode, calculate the current batch loss and append to
            # a growing list of loss values
            if (plot and (i % plot_per == 0) and i != 0):
                # report the average value of the loss function over the set
                # of iterations isince the last plotting period
                e = np.mean(inst_E)
                E.append(e)
                inst_E = []

            # report training progress
            progress = int(float(i)/float(num_iter)*100.0)
            sys.stdout.write('\rTraining: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rTraining: 100%\n\n')
        sys.stdout.flush()

        test_Y_hat = np.zeros_like(test_Y)
        num_test = test_X.shape[0]
        for i in range(num_test):
            # grab a randomly chosen batch from the training data
            batch_X = test_X[None, i, :]
            test_feed = {inputs: batch_X, keep_prob: 1.0}

            # run a step of the autoencoder trainer
            test_Y_hat[i] = sess.run(model.predict, test_feed)

        # get accuracy
        if (0 in np.unique(train_Y)):
            correct = np.sum(1 * (test_Y_hat == test_Y))
        else:
            correct = np.sum(1 * (test_Y_hat == (test_Y-1)))
        accuracy = float(correct)/float(num_test)*100.0
        print accuracy

        O = np.zeros((n, latent_dim))
        for i in range(n):
            test_feed = {inputs: train_X[None, i, :], keep_prob: 1.0}

            out = sess.run(model.encode, test_feed)

            O[i, :] = out

        np.savetxt('train_latent.dat', O)

        sess.close()

    # save loss values to a csv file
    l_fp = open('loss.csv', 'w')
    for e in E:
        l_fp.write(str(e) + '\n')
    l_fp.close()

    y = np.loadtxt('loss.csv', delimiter=',')

    if plot:
        #plt.plot(y)
        plt.plot(range(len(E)), E)
        plt.show()

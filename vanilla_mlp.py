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
import time

from neural.models import MultiLayerPerceptron
from neural.layers import KCompetitiveLayer
from utils.data_utils import getBatch
from utils.data_utils import label2OneHot

if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 7):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./vanilla_ae.py <train_x_fn> <train_y_fn> \
                <test_x_fn> <test_y_fn> <num_hidden> <out_dir>')
        sys.exit()

    # get the data
    train_X = np.loadtxt(sys.argv[1])
    train_Y = np.loadtxt(sys.argv[2])
    test_X = np.loadtxt(sys.argv[3])
    test_Y = np.loadtxt(sys.argv[4])

    # center data
    train_X = train_X - np.mean(train_X, axis=0)
    test_X = test_X - np.mean(test_X, axis=0)

    train_OH = label2OneHot(np.reshape(train_Y, (train_Y.shape[0], 1)))
    test_OH = label2OneHot(np.reshape(test_Y, (test_Y.shape[0], 1)))

    n, d = train_X.shape

    K = len(np.unique(train_Y))

    # for collecting error values
    E = []

    """ hyper parameters """
    eta = 1e-3
    latent_dim = int(sys.argv[5])

    """ runtime parameters """
    num_iter = 10000
    plot_per = 10
    batch_size = 128
    plot = 1
    save_states = 1

    """ tensor flow ops """
    # graph input
    inputs = tf.placeholder(tf.float32, [None, d])
    targets = tf.placeholder(tf.float32, [None, K])

    model = MultiLayerPerceptron([d, latent_dim, K],
                                   inputs,
                                   scope='mlp',
                                   targets=targets,
                                   out_type='logits')

    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(model.loss)

    correct = tf.cast(tf.equal(tf.argmax(model.predict, 1), tf.argmax(targets, 1)), tf.float32)

    accuracy = tf.reduce_mean(correct)

    init = tf.global_variables_initializer()

    """ Tensorflow Session """
    with tf.Session() as sess:
        sess.run(init)

        # train the network
        for i in range(num_iter):
            # grab a randomly chosen batch from the training data
            batch_X, batch_Y, indices = getBatch(train_X, train_OH, batch_size)
            train_feed = {inputs: batch_X, targets: batch_Y}

            # run a step of the autoencoder trainer
            sess.run(optimizer, train_feed)

            if ((i % plot_per) == 0):
                e = sess.run(model.loss, train_feed)
                E.append(e)

            # report training progress
            progress = int(float(i)/float(num_iter)*100.0)
            sys.stdout.write('\rTraining: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rTraining: 100%\n\n')
        sys.stdout.flush()

        test_feed = {inputs: test_X, targets: test_OH}
        acc = sess.run(accuracy, test_feed)

        print acc

        O = np.zeros((n, latent_dim))
        for i in range(n):
            test_feed = {inputs: train_X[None, i, :]}

            out = sess.run(model.encode, test_feed)

            O[i, :] = out

        np.savetxt('train_aelatent.dat', O)

        sess.close()

    # save loss values to a csv file
    l_fp = open('loss.csv', 'w')
    for e in E:
        l_fp.write(str(e) + '\n')
    l_fp.close()

    y = np.loadtxt('loss.csv', delimiter=',')

    if plot:
        plt.plot(y)
        plt.show()

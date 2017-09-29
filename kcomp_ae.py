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

from neural.models import MultiLayerPerceptron
from neural.layers import KCompetitiveLayer
from utils.data_utils import getBatch

if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 3):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./vanilla_ae.py <train_x_fp> <out_dir>')
        sys.exit()

    # get the training data
    X = np.loadtxt(sys.argv[1])

    n, d = X.shape

    # for collecting error values
    E = []

    """ hyper parameters """
    eta = 1e-2
    latent_dim = 100

    alpha = 6.26
    ktop = latent_dim//4

    """ runtime parameters """
    num_iter = 100
    plot_per = 1
    batch_size = 128
    plot = 1
    save_states = 1

    """ tensor flow ops """
    # graph input
    inputs = tf.placeholder(tf.float32, [None, d])
    targets = tf.placeholder(tf.float32, [None, d])

    encoder = MultiLayerPerceptron([d, latent_dim],
                                   inputs,
                                   scope='mlp_encoder',
                                   out_type='tanh')

    #competitive = KCompetitiveLayer(encoder.predict, ktop, alpha)

    decoder = MultiLayerPerceptron([latent_dim, d],
                                   encoder.predict,
                                   scope='mlp_decoder',
                                   out_type='logits')

    outputs = decoder.predict

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets,
        logits=outputs
    )

    error = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(error)

    init = tf.global_variables_initializer()

    """ Tensorflow Session """
    with tf.Session() as sess:
        sess.run(init)

        # train the network
        for i in range(num_iter):
            # grab a randomly chosen batch from the training data
            batch_X, batch_Y, indices = getBatch(X, X, batch_size)
            train_feed = {inputs: batch_X, targets: batch_Y}

            # run a step of the autoencoder trainer
            sess.run(optimizer, train_feed)

            if ((i % plot_per) == 0):
                e = sess.run(error, train_feed)
                E.append(e)

            # report training progress
            progress = int(float(i)/float(num_iter)*100.0)
            sys.stdout.write('\rTraining: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rTraining: 100%\n\n')
        sys.stdout.flush()

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

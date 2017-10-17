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
    with open(sys.argv[1], 'r') as fp:
        train_X = pickle.load(fp)
    train_Y = np.loadtxt(sys.argv[2])
    with open(sys.argv[3], 'r') as fp:
        test_X = pickle.load(fp)
    test_Y = np.loadtxt(sys.argv[4])

    train_OH = label2OneHot(np.reshape(train_Y, (train_Y.shape[0], 1)))
    test_OH = label2OneHot(np.reshape(test_Y, (test_Y.shape[0], 1)))

    n = len(train_X)
    d = 300

    K = len(np.unique(train_Y))

    # for collecting error values
    E = []

    embed_dir = '/home/dylan/rpi/thesis/GoogleNews-vectors-negative300.bin'

    """ hyper parameters """
    eta = 1e-3
    latent_dim = int(sys.argv[5])

    """ runtime parameters """
    num_iter = 1000000
    plot_per = 10000
    batch_size = 1
    plot = 1
    save_states = 1

    """ tensor flow ops """
    # initializers for any weights and biases in the model
    w_init = tf.contrib.layers.xavier_initializer()
    b_init = 0.01

    # cnn - want shape [1, d, s], rnn - want shape [s, 1, 300]
    inputs_emb = tf.placeholder(tf.float32, [1, None, d])

    # placeholder for the rnn targets (one-hot encodings)
    targets = tf.placeholder(tf.float32, [1, K])

    rnn_inputs = tf.transpose(inputs_emb, [1, 0, 2])
    #cnn_inputs = tf.transpose(inputs_emb, [0, 2, 1])

    cell = tf.contrib.rnn.LSTMCell(latent_dim, state_is_tuple=True)

    outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                       inputs=inputs_emb,
                                       dtype=tf.float32,
                                       time_major=True)

    rnn_outputs = tf.reshape(outputs[0][-1], (1, latent_dim))

    outputs = tf.contrib.layers.fully_connected(rnn_outputs, K,
                                             activation_fn=tf.nn.softmax)

    loss = tf.reduce_mean((outputs - targets)**2)

    #l2_losses = tf.add_n([tf.nn.l2_loss(p) for p in tf.trainable_variables()])

    #loss += l2_losses

    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)

    init = tf.global_variables_initializer()

    """ initialize the google word2vec model """
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(embed_dir,
                                                            binary=True)

    """ Tensorflow Session """
    with tf.Session() as sess:
        sess.run(init)

        inst_E = []

        # train the network
        for i in range(num_iter):
            # grab a randomly chosen batch from the training data
            r = np.random.randint(0, n)
            batch_X = train_X[r]
            batch_Y = train_OH[None, r, :]

            # build word embedding
            s = len(batch_X)
            W = np.zeros((s, d))
            # get the sentence embedding matrix W
            for j, word in enumerate(batch_X):
                try:
                    W[j, :] = wv_model.wv[word]
                except:
                    W[j, :] = np.random.normal(0, 1, (1, d))

            W = np.reshape(W, (1, W.shape[0], W.shape[1]))

            train_feed = {inputs_emb: W, targets: batch_Y}

            # run a step of the autoencoder trainer
            sess.run(optimizer, train_feed)

            # get loss of this sample
            e = sess.run(loss, train_feed)
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

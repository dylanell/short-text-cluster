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

    n, s = train_X.shape
    d = 64

    K = len(np.unique(train_Y))

    # for collecting error values
    E = []

    """ hyper parameters """
    eta = 1e-5
    num_maps = 100
    flat_dim = num_maps * 3
    latent_dim = 50
    margin = 2.0

    """ runtime parameters """
    num_iter = 500000
    plot_per = 1000
    batch_size = 1
    plot = 1

    """ conv hyper params """
    # filter = # [height, width, input_depth, num_filters]
    c1_filt_dim = [3, d, 1, num_maps]
    c1_strides = [1, 1, 1, 1]
    c1_pad = [[0, 0],
              [c1_filt_dim[0] - 1, c1_filt_dim[0] - 1],
              [0, 0],
              [0, 0]]

    # filter = # [height, width, input_depth, num_filters]
    c2_filt_dim = [4, d, 1, num_maps]
    c2_strides = [1, 1, 1, 1]
    c2_pad = [[0, 0],
              [c2_filt_dim[0] - 1, c2_filt_dim[0] - 1],
              [0, 0],
              [0, 0]]

    # filter = # [height, width, input_depth, num_filters]
    c3_filt_dim = [5, d, 1, num_maps]
    c3_strides = [1, 1, 1, 1]
    c3_pad = [[0, 0],
              [c3_filt_dim[0] - 1, c3_filt_dim[0] - 1],
              [0, 0],
              [0, 0]]


    """ tensor flow ops """
    # initializers for any weights and biases in the model
    w_init = tf.contrib.layers.xavier_initializer()
    b_init = 0.01

    # filter weights for conv1
    c1_filt = tf.get_variable('conv1_filters',
                                 shape=c1_filt_dim,
                                 initializer=w_init)

    c1_bias = tf.get_variable("conv1_biases",
    initializer=(b_init * tf.ones([1], tf.float32)))

    # filter weights for conv1
    c2_filt = tf.get_variable('conv2_filters',
                                 shape=c2_filt_dim,
                                 initializer=w_init)

    c2_bias = tf.get_variable("conv2_biases",
    initializer=(b_init * tf.ones([1], tf.float32)))

    # filter weights for conv1
    c3_filt = tf.get_variable('conv3_filters',
                                 shape=c3_filt_dim,
                                 initializer=w_init)

    c3_bias = tf.get_variable("conv3_biases",
    initializer=(b_init * tf.ones([1], tf.float32)))

    # cnn - want shape [1, d, s], rnn - want shape [s, 1, 300]
    inputs = tf.placeholder(tf.int32, [None, None])

    # placeholder for the rnn targets (one-hot encodings)
    targets = tf.placeholder(tf.float32, [None, 1])

    # embedding matrix for all words in our vocabulary
    embeddings = tf.get_variable('embedding_weights',
                                 shape =[vocab_len, d],
                                 initializer=w_init)

    # mask to zero out vectors correspondong to nulls
    def nullMask(seq):
        used = tf.cast(tf.not_equal(seq, NULL_IDX), tf.float32)
        num = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(used, 1), -1), tf.int32))
        return used, num

    """ define the model """
    def model(inputs):
        """ embedding and reshaping """
        # get the length of this input
        mask = tf.expand_dims(nullMask(inputs)[0], [-1])
        num_samp = nullMask(inputs)[1]

        # these ops embed the word vector placeholders using the embedding weights
        inputs_emb = mask * tf.nn.embedding_lookup(embeddings, inputs)

        # reshape inputs to be like a batch of "images" with pixel depth 1,
        # therefore making them 4D tensors
        inputs_resh = tf.expand_dims(inputs_emb, -1)

        """ convolution layer 1 """
        # pad the inputs
        c1_padded = tf.pad(inputs_resh, c1_pad)

        # perform wide convolution 1
        c1_out = tf.nn.conv2d(c1_padded, c1_filt,
                                 strides=c1_strides, padding='VALID')

        c1_biased = c1_out + c1_bias

        c1_act = tf.nn.tanh(c1_biased)

        c1_pool = tf.reduce_max(c1_act, axis=1)

        """ convolution layer 2 """
        # pad the inputs
        c2_padded = tf.pad(inputs_resh, c2_pad)

        # perform wide convolution 1
        c2_out = tf.nn.conv2d(c2_padded, c2_filt,
                                 strides=c2_strides, padding='VALID')

        c2_biased = c2_out + c2_bias

        c2_act = tf.nn.tanh(c2_biased)

        c2_pool = tf.reduce_max(c2_act, axis=1)

        """ convolution layer 3 """
        # pad the inputs
        c3_padded = tf.pad(inputs_resh, c3_pad)

        # perform wide convolution 1
        c3_out = tf.nn.conv2d(c3_padded, c3_filt,
                                 strides=c3_strides, padding='VALID')

        c3_biased = c3_out + c3_bias

        c3_act = tf.nn.tanh(c3_biased)

        c3_pool = tf.reduce_max(c3_act, axis=1)


        """ fully connected layer """
        concat = tf.concat([c1_pool, c2_pool, c3_pool], axis=2)
        flat = tf.reshape(concat, [num_samp, flat_dim])

        output = tf.contrib.layers.fully_connected(flat, latent_dim,
                                                 activation_fn=tf.nn.tanh)

        # normalize the features
        output_norm = tf.nn.l2_normalize(output, dim=1)

        return output_norm

    outputs = model(inputs)

    """ contrastive loss """
    # need to use l1 dist
    diff = outputs[0, :] - outputs[1, :]
    dist = tf.norm(diff, ord=1)

    l_s = 0.5 * (dist**2)
    l_d = 0.5 * (tf.maximum(0.0, margin - dist)**2)

    loss = (1.0 - targets[0, 0])*l_s + (targets[0, 0]*l_d)

    # define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)

    # variable initializer
    init = tf.global_variables_initializer()

    """ Tensorflow Session """
    with tf.Session() as sess:
        sess.run(init)

        inst_E = []

        # train the network
        for i in range(num_iter):

            des = np.random.uniform()

            sample = np.random.randint(0, n)
            batch_X = train_X[None, sample, :]

            if (des > 0.5):
                # choose a disimilar sample
                tar = np.ones((1, 1))
                while True:
                    neighbor = np.random.randint(0, n)
                    if (train_Y[sample] != train_Y[neighbor]):
                        break
            else:
                # choose a similar sample
                tar = np.zeros((1, 1))
                while True:
                    neighbor = np.random.randint(0, n)
                    if ((train_Y[sample] == train_Y[neighbor]) and  \
                                                (sample != neighbor)):
                        break

            batch_X = np.append(batch_X, train_X[None, neighbor, :], axis=0)

            train_feed = {inputs: batch_X, targets: tar}

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
                out = sess.run(outputs, train_feed)
                if(np.any(np.isnan(out)) == True):
                    print('GOT NANS')
                    exit()

            # report training progress
            progress = int(float(i)/float(num_iter)*100.0)
            sys.stdout.write('\rTraining: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rTraining: 100%\n\n')
        sys.stdout.flush()

        n = test_X.shape[0]
        O = np.zeros((n, latent_dim))
        for i in range(n):
            test_feed = {inputs: test_X[None, i, :]}

            out = sess.run(outputs, test_feed)

            O[i, :] = out

        np.savetxt('latent.dat', O)

        sess.close()

    # save loss values to a csv file
    l_fp = open('loss.csv', 'w')
    for e in E:
        l_fp.write(str(e) + '\n')
    l_fp.close()

    y = np.loadtxt('loss.csv', delimiter=',')

    if plot:
        plt.plot(y)
        #plt.plot(range(len(E)), E)
        plt.show()
    # end

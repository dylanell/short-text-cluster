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

from neural.models import NBOW
from neural.models import LSTM
from neural.models import TextCNN
from neural.models import DynamicCNN

NULL = '__'

def validateModel(model_type):
    # check if valid output type
    valid_types = ['lstm', 'dcnn', 'tcnn', 'nbow']
    assert(model_type in valid_types), 'model \'%s\' is not valid' % model_type


if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 9):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./classify.py <data_dir> <model> <emb_dim> ' \
              '<num_iter> <batch_size> <eta> <latent_dim> <plot_flag>')
        sys.exit()

    # get the type of model were using and chek if valid
    model_type = sys.argv[2]
    validateModel(model_type)

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

    # size of inputs [n, s] = [num_samples, max_seq_len]
    n, s = train_X.shape

    # dimensionality of the word embeddings
    d = int(sys.argv[3])

    # dimensionality of the output (num classes)
    K = len(np.unique(train_Y))

    # for collecting error values
    E = []

    """ hyper parameters """
    eta = float(sys.argv[6])

    """ runtime parameters """
    num_iter = int(sys.argv[4])
    plot_per = 50
    batch_size = int(sys.argv[5])
    plot = int(sys.argv[8])

    """ model parameters """
    latent_dim = int(sys.argv[7])
    emb_dims = [vocab_len, d]

    print('\n[INFO] Model: %s' % model_type)
    print('[INFO] Data Source: %s' % data_dir)
    print('[INFO] Number of Iterations: %d' % num_iter)
    print('[INFO] Word Vector Dimension: %d' % d)
    print('[INFO] Batch Size: %d' % batch_size)
    print('[INFO] Learning Rate: %f' % eta)
    print('[INFO] Latent Dimension: %d' % latent_dim)
    print('[INFO] Plotting Loss: %r\n' % bool(plot))

    """ tensorflow ops """
    # keep probability for dropout layers
    keep_prob = tf.placeholder(tf.float32)

    # placeholder for inputs [batch_size, max_seq_len]
    inputs = tf.placeholder(tf.int32, [None, None])

    # placeholder for targets [batch_size, num_classes]
    targets = tf.placeholder(tf.float32, [None, K])

    # initialize the model (model_type)
    if (model_type=='nbow'):
        dims = [latent_dim, K]
        model = NBOW(emb_dims, dims, inputs, targets,
                     NULL_IDX, kp=keep_prob, eta=eta, out_type='softmax')
    elif (model_type=='lstm'):
        dims = [latent_dim, K]
        model = LSTM(emb_dims, dims, inputs, targets,
                     NULL_IDX, kp=keep_prob, eta=eta, out_type='softmax')
    elif (model_type=='tcnn'):
        num_maps = 100
        flat_dim = num_maps * 3
        emb_dims = [vocab_len, d]
        filt_dims = [[3, num_maps], [4, num_maps], [5, num_maps]]
        fc_dims = [latent_dim, K]
        model = TextCNN(emb_dims, filt_dims, fc_dims, inputs,
                    targets, NULL_IDX, kp=keep_prob, eta=eta,
                    out_type='softmax')
    elif (model_type=='dcnn'):
        k_top_v = 4
        filt_dims = [[7, 20], [5, 14]]
        fc_dims = [latent_dim, K]
        model = DynamicCNN(emb_dims, filt_dims, fc_dims, inputs,
                    targets, NULL_IDX, k_top_v=k_top_v, kp=keep_prob,
                    eta=eta, out_type='softmax')
    else:
        print('model \'%s\' is not valid' % model_type)

    init = tf.global_variables_initializer()

    """ Tensorflow Session """
    with tf.Session() as sess:
        sess.run(init)

        inst_E = []

        # train the network
        print('')
        for i in range(num_iter):
            # grab a randomly chosen batch from the training data
            batch_X, batch_Y, indices = getBatch(train_X, train_OH, batch_size)
            train_feed = {inputs: batch_X, targets: batch_Y, keep_prob: 0.5}

            #probe = sess.run(model.probe, train_feed)
            #print probe.shape
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
            sys.stdout.write('\r[INFO] Training: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\r[INFO] Training: 100%\n')
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
        correct = np.sum(1 * (test_Y_hat == test_Y))
        accuracy = float(correct)/float(num_test)*100.0
        print('[INFO] Accuracy: %.2f' % accuracy)

        a = np.loadtxt('accuracy_log.txt')
        a = np.append(a, accuracy)
        np.savetxt('accuracy_log.txt', a)

        O = np.zeros((n, latent_dim))
        B = np.zeros((n))
        for i in range(n):
            test_feed = {inputs: train_X[None, i, :], keep_prob: 1.0}

            out = sess.run(model.encode, test_feed)

            O[i, :] = out
            B[i] = train_Y[i]

        np.savetxt('train_latent_' + str(accuracy) + '.dat', O)
        np.savetxt('train_label_' + str(accuracy) + '.dat', B)

        sess.close()

    if plot:
        # save loss values to a csv file
        l_fp = open('loss.dat', 'w')
        for e in E:
            l_fp.write(str(e) + '\n')
        l_fp.close()

        y = np.loadtxt('loss.dat', delimiter=',')

        plt.plot(y)
        plt.show()

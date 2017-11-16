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

def getPairWiseBatch(X, Y, batch_size):
    n, d = X.shape
    batch_Y = np.random.randint(0, 2, (batch_size))
    batch_X = np.zeros((2*batch_size, d))

    # TODO: need iterator for neighbor adding
    i = 0
    for y in batch_Y:
        sample = np.random.randint(0, n)

        if (y == 1):
            # choose a disimilar sample
            while True:
                neighbor = np.random.randint(0, n)
                if (Y[sample] != Y[neighbor]):
                    break
        else:
            # choose a similar but different sample
            while True:
                neighbor = np.random.randint(0, n)
                if ((Y[sample] == Y[neighbor]) and (sample != neighbor)):
                    break

        # add sample and neighbor to batch
        batch_X[i, :] = X[None, sample, :]
        batch_X[i+1, :] = X[None, neighbor, :]

        # increment to next pairing in the batch
        i += 2

    return batch_X, batch_Y

if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 3):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./semi_kmeans.py <data_dir> <model>')
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

    # concatenate all data to make actual training set
    train_X = np.concatenate([train_X, test_X], axis=0)
    train_Y = np.concatenate([train_Y, test_Y], axis=0)

    vocab = gensim.corpora.Dictionary.load(data_dir + 'vocabulary.dat')
    vocab_len = len(vocab)

    # find the index of the NULL word '__'
    rev_vocab = dict()
    for key, value in vocab.iteritems():
        rev_vocab[value] = key
    NULL_IDX = rev_vocab[NULL]
    del vocab
    del rev_vocab

    # size of inputs [n, s] = [num_samples, max_seq_len]
    n, s = train_X.shape

    # dimensionality of the output (num classes)
    K = len(np.unique(train_Y))

    # for collecting error values
    E = []

    """ hyper parameters """
    eta = 1e-5
    num_samp = 0
    l = 595                         # number of labeled samples to use

    """ runtime parameters """
    num_iter = 1000
    plot_per = 100
    batch_size = 1
    plot = 1

    """ model parameters """
    d = 64                          # dimensionality of the word embeddings
    latent_dim = 100                # dimension of the learned embedding
    margin = 2.0
    emb_dims = [vocab_len, d]

    # if number samples set, take that many samples
    if (num_samp != 0):
        n = num_samp

    print('\n[INFO] Model: %s' % model_type)
    print('[INFO] Data Source: %s' % data_dir)
    print('[INFO] Number Sampled: %d' % n)
    print('[INFO] Number Labeled: %d' % l)
    print('[INFO] Number of Iterations: %d' % num_iter)
    print('[INFO] Word Vector Dimension: %d' % d)
    print('[INFO] Latent Vector Dimension: %d' % latent_dim)
    print('[INFO] Margin: %.3f' % margin)
    print('[INFO] Batch Size: %d' % batch_size)
    print('[INFO] Learning Rate: %f' % eta)
    print('[INFO] Plotting Loss: %r\n' % bool(plot))

    # split into labeled and unlabeled sets and ensure we have a sample
    # from every class in each
    # fist l samples of train_X are the labeled ones
    while True:
        # grab a "batch" of data the entire size of data to shuffle samples
        # with labels
        train_X, train_Y, indices = getBatch(train_X, train_Y, n)

        # choose l of these for the labeled samples
        labeled_X = train_X[:l]
        labeled_Y = train_Y[:l]
        labeled_OH = label2OneHot(np.reshape(labeled_Y,
                                            (labeled_Y.shape[0], 1)))
        labeled = indices[:l]
        unlabeled_X = train_X[l:]
        unlabeled_Y = train_Y[l:]

        # count how many unqique classes in each split
        unique_labeled = np.unique(labeled_Y).shape[0]
        unique_unlabeled = np.unique(unlabeled_Y).shape[0]

        # if both splits have samples from every class then we can continue
        if ((unique_labeled == K) and (unique_unlabeled==K)):
            break

    """ tensorflow ops """
    # keep probability for dropout layers
    keep_prob = tf.placeholder(tf.float32)

    # placeholder for inputs [batch_size, max_seq_len]
    inputs = tf.placeholder(tf.int32, [None, None])

    # placeholder for targets [batch_size, num_classes]
    targets = tf.placeholder(tf.float32, [None])

    targets_OH = tf.one_hot(tf.cast(tf.concat([targets, targets], axis=0), tf.int32), K)

    # initialize the model (model_type)
    if (model_type=='nbow'):
        dims = [latent_dim, K]
        model = NBOW(emb_dims, dims, inputs, targets_OH,
                     NULL_IDX, kp=keep_prob, eta=eta, out_type='softmax')
    elif (model_type=='lstm'):
        dims = [latent_dim, K]
        model = LSTM(emb_dims, dims, inputs, targets_OH,
                     NULL_IDX, kp=keep_prob, eta=eta, out_type='softmax')
    elif (model_type=='tcnn'):
        num_maps = 100
        flat_dim = num_maps * 3
        emb_dims = [vocab_len, d]
        filt_dims = [[3, num_maps], [4, num_maps], [5, num_maps]]
        fc_dims = [latent_dim, K]
        model = TextCNN(emb_dims, filt_dims, fc_dims, inputs,
                    targets_OH, NULL_IDX, kp=keep_prob, eta=eta,
                    out_type='softmax')
    elif (model_type=='dcnn'):
        k_top_v = 5
        filt_dims = [[8, 5]]
        fc_dims = [latent_dim, K]
        model = DynamicCNN(emb_dims, filt_dims, fc_dims, inputs,
                    targets_OH, NULL_IDX, k_top_v=k_top_v, kp=keep_prob,
                    eta=eta, out_type='softmax')
    else:
        print('model \'%s\' is not valid' % model_type)

    outputs = model.encode

    # get diff between every pair of rows using tensor slices
    diff = outputs[0::2, :] - outputs[1::2, :]
    dist = tf.norm(diff, ord=1, axis=1)

    l_s = 0.5 * (dist**2)
    l_d = 0.5 * (tf.maximum(0.0, margin - dist)**2)

    error = (1.0 - targets)*l_s + (targets*l_d)

    loss = tf.reduce_mean(error)

    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)

    init = tf.global_variables_initializer()

    """ Tensorflow Session """
    with tf.Session() as sess:
        sess.run(init)

        inst_E = []

        # train the network on the labeled data
        for i in range(num_iter):
            # grab a randomly chosen batch from the training data
            batch_X, batch_Y = getPairWiseBatch(labeled_X, labeled_Y, batch_size)
            train_feed = {inputs: batch_X, targets: batch_Y, keep_prob: 0.5}

            #print batch_Y
            #probe = sess.run(dist, train_feed)
            #print probe
            #probe = sess.run(l_s, train_feed)
            #print probe
            #probe = sess.run(l_d, train_feed)
            #print probe
            #probe = sess.run(error, train_feed)
            #print probe
            #exit()

            # run a step of the autoencoder trainer
            sess.run(model.optimize, train_feed)

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
                print e

            # report training progress
            progress = int(float(i)/float(num_iter)*100.0)
            sys.stdout.write('\r[INFO] Training: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\r[INFO] Training: 100%\n')
        sys.stdout.flush()

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

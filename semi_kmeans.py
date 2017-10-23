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

import scipy

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

def getSeed(labeled_X, labeled_Y):
    classes = np.unique(labeled_Y)
    K = classes.shape[0]

    n, d = labeled_X.shape

    seed_X = np.zeros((K, d))

    for c in classes:
        while True:
            r = np.random.randint(0, n)
            seed_X[int(c), :] = labeled_X[r, :]
            if (labeled_Y[r] == c):
                break

    return seed_X

def assignClusters(FX, MU):
    n, d = FX.shape

    K = MU.shape[0]

    R = np.zeros((n, K))

    for i in range(n):
        # calculate the distance from every centroid
        dist = np.linalg.norm(FX[None, i, :] - MU, axis=1)

        # assign label from closest centroid
        R[i, np.argmin(dist)] = 1

    return R

# src: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
def arrayRowIntersection(a,b):
   tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
   return a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]

# uses scikit learn to solve the minimum assignment problem to construct
# an optimal mapping from true labels T to current labels L
def mapping(T, L, X):
    classes = np.unique(labeled_Y)
    K = classes.shape[0]

    C = np.zeros((K, K))

    # convert L from one hot to numerical labels
    L = np.argmax(L, axis=1)

    for i in range(K):
        for j in range(K):
            Xi = X[np.where(T == i)]
            Xj = X[np.where(L == j)]

            Xi_and_Xj = arrayRowIntersection(Xi, Xj)

            len_Xi = Xi.shape[0]
            len_Xj = Xj.shape[0]
            len_Xi_and_Xj = Xi_and_Xj.shape[0]

            dis = len_Xi + len_Xj - 2*(len_Xi_and_Xj)

            C[i, j] = dis

    # solve the linear assignment problem on C with scipy
    G = scipy.optimize.linear_sum_assignment(C)

    return G[-1]


# TODO: ensure we get all classes in both splits
if __name__ == '__main__':
    # retrieve command line args
    if (len(sys.argv) < 5):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./classify.py <model> <data_dir> ' \
                    '<num_labeled> <emb_dim>')
        sys.exit()

    # get the type of model were using and chek if valid
    model_type = sys.argv[1]
    validateModel(model_type)

    # get the data
    data_dir = sys.argv[2]
    train_X = np.loadtxt(data_dir + 'train_indices.dat')
    train_Y = np.loadtxt(data_dir + 'train_label.dat')

    # number of labeled samples to use
    l = int(sys.argv[3])

    # dimension of the learned embedding
    latent_dim = int(sys.argv[4])

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

    # dimensionality of the word embeddings
    d = 64

    # dimensionality of the output (num classes)
    K = len(np.unique(train_Y))

    # for collecting error values
    E = []

    # split into labeled and unlabeled sets and ensure we have a sample
    # from every class in each
    while True:
        # grab a "batch" of data the entire size of data to shuffle samples
        # with labels
        train_X, train_Y, indices = getBatch(train_X, train_Y, n)

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


    """ hyper parameters """
    eta = 1e-3

    """ runtime parameters """
    num_iter = 1000
    pretrain_iter = 1000
    plot_per = 10
    batch_size = 32
    plot = 0

    """ model parameters """
    emb_dims = [vocab_len, d]

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


    #external_loss = tf.zeros([1])

    #external_optimize = #tf.train.AdamOptimizer(learning_rate=eta).minimize(external_loss)

    init = tf.global_variables_initializer()

    """ kmeans parameters """
    # TODO: use kmeans++
    MU = np.random.uniform(-10, 10, (K, latent_dim))

    R = np.zeros((n, K))

    """ Tensorflow Session """
    with tf.Session() as sess:
        sess.run(init)

        """ pretraining """
        for i in range(pretrain_iter):
            # grab a randomly chosen batch from the training data
            batch_X, batch_Y, indices = getBatch(labeled_X,
                                                 labeled_OH,
                                                 batch_size)
            train_feed = {inputs: batch_X, targets: batch_Y, keep_prob: 0.5}

            # run a step of the autoencoder trainer
            sess.run(model.optimize, train_feed)

            # report training progress
            progress = int(float(i)/float(num_iter)*100.0)
            sys.stdout.write('\rPre-training: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rPre-training: 100%\n\n')
        sys.stdout.flush()

        """ report pretrain error """
        inst_E = []
        for i in range(l):
            test_feed = {inputs: labeled_X[None, i, :],
                         targets: labeled_OH[None, i, :],
                         keep_prob: 1.0}
            inst_E.append(sess.run(model.loss, test_feed))
        print('Pretrain Error: %f' % np.mean(inst_E))

        """ intiailize cluster centroids with pretrained encodings """
        # get a seed batch from the labeled data from each class
        seed_X = getSeed(labeled_X, labeled_Y)
        seed_feed = {inputs: seed_X, keep_prob: 1.0}

        # feed the seed batch through the pretraiend network to get
        # initialized cluster centroids
        MU = sess.run(model.encode, seed_feed)

        print 'MU', MU.shape

        inst_E = []
        for i in range(num_iter):
            """ minimize J by changing R (f(x) and MU constant) """
            # encode all points
            FX = np.zeros((n, latent_dim))
            for i in range(n):
                encode_feed = {inputs: train_X[None, i, :], keep_prob: 1.0}
                FX[i, :] = sess.run(model.encode, encode_feed)

            print 'FX', FX.shape

            # populate R by assigning current labels to the data
            R = assignClusters(FX, MU)

            print 'R', R.shape

            # get the dissagrement cost (C) between the assignent R and label
            G = mapping(labeled_Y, R[:l], labeled_X)

            print 'G', G

            # NOTE
            exit()

            """ minimize J by changing MU (f(x) and R constant) """
            # TODO

            """ minimize J by changing f(x) (MU and R constant) """
            # TODO

            # report training progress
            progress = int(float(i)/float(num_iter)*100.0)
            sys.stdout.write('\rTraining: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rTraining: 100%\n\n')
        sys.stdout.flush()

        sess.close()

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

def updateAssignments(FX, MU):
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
# "for each truth label, what is the equivalent label in our cluster labels?"
# this is what mapping G(.) does. cluster_label = G(truth_label)
def mapping(T, L, X):
    classes = np.unique(labeled_Y)
    K = classes.shape[0]

    C = np.zeros((K, K))

    # convert L from one hot to numerical labels
    L = np.argmax(L, axis=1)

    n = T.shape[0]

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

    # map each truth label in T to the correspondng cluster label in mapped_T
    mapped_T = np.zeros_like(T)

    for i in range(n):
        mapped_T[i] = G[-1][int(T[i])]

    return mapped_T, G[-1]


# updates centroids using eq. 5 using labeled and unlabeled data
def updateCentroids(MU, FX, R, mapped_Y, alpha, margin):
    L = mapped_Y.shape[0]

    N, d = FX.shape

    K = MU.shape[0]

    W = np.zeros((L, K))

    # for each column of W (k) and row of W (n) W -> LxK
    for k in range(K):
        for n in range(L):
            # get the mapped label for labeled sample n
            gn = int(mapped_Y[n])

            # create an iterator for all labels other than gn
            not_gn = [g for g in range(K) if g != gn]

            # distance from sample fxn to centroid gn
            fxn = FX[None, n, :]
            mugn = MU[None, gn, :]
            fxn_to_mugn = np.linalg.norm(fxn - mugn)**2

            # indicator that class k is equal to gn
            I_prime_nk = delta(k, gn)

            # intiailize sums to calculate over all labels != gn
            sum_I_dprime_nkj = 0
            sum_I_tprime_nkj = 0

            # iterate through labels other than gn to get sums for
            # I_tprime_nkj, I_dprime_nkj
            for j in not_gn:
                # distance from sample fxn to centroid j
                muj = MU[None, j, :]
                fxn_to_muj = np.linalg.norm(fxn - muj)**2

                # indicator if distance to centroid j is less than distance
                # to centroid gn plus the margin
                d_prime_nj = delta(margin + fxn_to_mugn - fxn_to_muj)

                sum_I_dprime_nkj += delta(k, j) * d_prime_nj
                sum_I_tprime_nkj += (1 - delta(k, j)) * d_prime_nj

            # calculate W[n, k]
            W[n, k] = (1 - alpha) * (I_prime_nk +
                                     sum_I_dprime_nkj -
                                     sum_I_tprime_nkj)

        # build terms of the equation for muk
        # top right term of eq. (5)
        top_right = np.matmul(W[:, k, None].T, FX[:l, :])

        # bottom right term of eq. (5)
        bot_right = np.sum(W[:, k, None])

        # top left term of eq. (5)
        top_left = alpha * np.matmul(R[:, k, None].T, FX)

        # bottom left term of eq. (5)
        bot_left = alpha * np.sum(R[:, k, None])

        # calculate the updated centroid k
        muk = (top_left + top_right) / (bot_left + bot_right)

        # update kth row of MU with this new centroid
        MU[k, :] = muk

    return MU

# hybrid delta function
# outputs 1 if a == b or outputs 1 if a > 0
def delta(a, b=None):
    if b is not None:
        return 1 * (a == b)
    else:
        return 1 * (a > 0)


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

    print('Num Labeled: %d' % l)
    print('Embed Dim: %d' % latent_dim)

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
    # fist l samples of train_X are the labeled ones
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
    alpha = 0.1
    margin = 2

    """ runtime parameters """
    num_iter = 1000
    pretrain_iter = 200
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


    # placeholders for netrok loss
    MU_tf = tf.placeholder(tf.float32, [K, latent_dim])

    R_tf = tf.placeholder(tf.float32, [n, K])

    mapped_Y_tf = tf.placeholder(tf.float32, [l, 1])

    # TODO: build loss for external optimizer (only variable are network params)
    FX_stack = tf.stack([model.encode for z in range(K)], axis=2)
    MU_stack = tf.stack([tf.transpose(MU_tf) for z in range(n)], axis=0)
    S = FX_stack - MU_stack
    S_norm = tf.pow(tf.norm(S, axis=1), 2)
    term1 = alpha * tf.reduce_sum(tf.diag_part(tf.matmul(S_norm, tf.transpose(R_tf))))

    #external_optimize = #tf.train.AdamOptimizer(learning_rate=eta).minimize(external_loss)

    init = tf.global_variables_initializer()

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
            encode_feed = {inputs: train_X, keep_prob: 1.0}
            FX = sess.run(model.encode, encode_feed)

            print 'FX', FX.shape

            # update R by assigning current labels to the data
            R = updateAssignments(FX, MU)

            print 'R', R.shape

            # map truth labels to labels in our clustering
            mapped_Y, G = mapping(labeled_Y, R[:l], labeled_X)

            print 'G', G

            """ minimize J by changing MU (f(x) and R constant) """
            # update MU by computing new centroids
            MU = updateCentroids(MU, FX, R, mapped_Y, alpha, margin)

            print 'MU', MU.shape

            """ minimize J by changing f(x) (MU and R constant) """
            train_feed = {inputs: train_X,
                          MU_tf: MU,
                          R_tf: R,
                          mapped_Y_tf: mapped_Y,
                          keep_prob: 0.5}

            # TODO: run external optimizer here to update network params

            exit()

            # report training progress
            progress = int(float(i)/float(num_iter)*100.0)
            sys.stdout.write('\rTraining: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rTraining: 100%\n\n')
        sys.stdout.flush()

        sess.close()

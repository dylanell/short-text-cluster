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

from sklearn import cluster
from sklearn import metrics

from utils.data_utils import getBatch
from utils.data_utils import label2OneHot
from utils import cluster_utils as cu

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
    if (len(sys.argv) < 8):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./classify.py <data_dir> <model> ' \
                    '<num_labeled> <margin> <pre_iter> ' \
                    '<train_iter> <out_dir>')
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

    # only use max 6000 samples
    n = min(n, 6000)

    """ hyper parameters """
    eta = 1e-3
    alpha = 0.01
    margin = float(sys.argv[4])
    l = int(sys.argv[3])            # number of labeled samples to use
    d = 300                         # dimensionality of the word embeddings

    """ runtime parameters """
    num_iter = int(sys.argv[6])
    pretrain_iter = int(sys.argv[5])
    plot_per = 1
    batch_size = 128
    plot = 0

    """ model parameters """
    emb_dims = [vocab_len, d]
    latent_dim = 100   # dimension of the learned embedding

    print('\n[INFO] Model: %s' % model_type)
    print('[INFO] Data Source: %s' % data_dir)
    print('[INFO] Number Labeled: %d' % l)
    print('[INFO] Pre-Train Iterations: %d' % pretrain_iter)
    print('[INFO] Training Iterations: %d' % num_iter)
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
        num_maps = 500
        flat_dim = num_maps * 3
        emb_dims = [vocab_len, d]
        filt_dims = [[1, num_maps], [2, num_maps], [3, num_maps]]
        fc_dims = [latent_dim, K]
        model = TextCNN(emb_dims, filt_dims, fc_dims, inputs,
                    targets, NULL_IDX, kp=keep_prob, eta=eta,
                    out_type='softmax')
    elif (model_type=='dcnn'):
        k_top_v = 5
        filt_dims = [[8, 5]]
        fc_dims = [latent_dim, K]
        model = DynamicCNN(emb_dims, filt_dims, fc_dims, inputs,
                    targets, NULL_IDX, k_top_v=k_top_v, kp=keep_prob,
                    eta=eta, out_type='softmax')
    else:
        print('model \'%s\' is not valid' % model_type)


    # placeholders for netrok loss
    MU_tf = tf.placeholder(tf.float32, [K, latent_dim])
    R_tf = tf.placeholder(tf.float32, [n, K])
    mapped_Y_tf = tf.placeholder(tf.int32, [l])

    # TODO: build loss for external optimizer (only variable are network params)
    FX_stack = tf.stack([model.encode for z in range(K)], axis=2)
    MU_stack = tf.stack([tf.transpose(MU_tf) for z in range(n)], axis=0)
    S = FX_stack - MU_stack
    S_norm = tf.pow(tf.norm(S, axis=1), 2) # S_norm[n, k] = ||f(s_n) - mu_k||^2

    # get term 1 from J loss eq. (4)
    #term1 = 1.0/(float(n)) * alpha * tf.reduce_sum(R_tf * S_norm)
    term1 = alpha * tf.reduce_sum(R_tf * S_norm)

    L_norm = S_norm[:l, :]
    map_OH = tf.cast(tf.one_hot(mapped_Y_tf, K), tf.float32)
    correct_dist = L_norm * map_OH

    correct_dist_vec = tf.reshape(tf.reduce_sum(correct_dist, axis=1), (l, 1))

    not_map_OH = tf.cast(tf.equal(map_OH, 0.0), tf.float32)

    error_dist = not_map_OH * (margin + correct_dist_vec - L_norm)

    error_dist_hinge = tf.maximum(error_dist, 0.0)

    error_dist_vec = tf.reshape(tf.reduce_sum(error_dist_hinge, axis=1), (l, 1))

    # get term 2 from J loss eq. (4)
    # term2 = 1.0/(float(l)) (1.0 - alpha) * tf.reduce_sum(correct_dist_vec + error_dist_vec)
    term2 = (1.0 - alpha) * tf.reduce_sum(correct_dist_vec + error_dist_vec)

    kmeans_loss = term1 + term2

    extern_opt = tf.train.AdamOptimizer(learning_rate=eta).minimize(kmeans_loss)

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
            progress = int(float(i)/float(pretrain_iter)*100.0)
            sys.stdout.write('\rPre-training: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rPre-training: 100%\n')
        sys.stdout.flush()

        """ report pretrain error """
        inst_E = []
        for i in range(l):
            test_feed = {inputs: labeled_X[None, i, :],
                         targets: labeled_OH[None, i, :],
                         keep_prob: 1.0}
            inst_E.append(sess.run(model.loss, test_feed))
        print('Pre-train Error: %f' % np.mean(inst_E))

        """ intiailize cluster centroids with pretrained encodings """
        # get a seed batch from the labeled data from each class
        seed_X = getSeed(labeled_X, labeled_Y)
        seed_feed = {inputs: seed_X, keep_prob: 1.0}

        # feed the seed batch through the pretraiend network to get
        # initialized cluster centroids
        MU = sess.run(model.encode, seed_feed)

        #print 'MU', MU.shape

        inst_E = []
        for i in range(num_iter):
            """ minimize J by changing R (f(x) and MU constant) """
            # encode all points
            encode_feed = {inputs: train_X, keep_prob: 1.0}
            FX = sess.run(model.encode, encode_feed)

            #print 'FX', FX.shape

            # update R by assigning current labels to the data
            R = updateAssignments(FX, MU)

            #print 'R', R.shape

            # map truth labels to labels in our clustering
            mapped_Y, G = cu.mapping(labeled_Y, R[:l], labeled_X, one_hot=True)

            #print 'G', G

            """ minimize J by changing MU (f(x) and R constant) """
            # update MU by computing new centroids
            MU = updateCentroids(MU, FX, R, mapped_Y, alpha, margin)

            #print 'MU', MU.shape

            """ minimize J by changing f(x) (MU and R constant) """
            train_feed = {inputs: train_X,
                          MU_tf: MU,
                          R_tf: R,
                          mapped_Y_tf: mapped_Y,
                          keep_prob: 0.5}

            # run the optimizer
            sess.run(extern_opt, train_feed)

            # colelct the current loss
            e = sess.run(kmeans_loss, train_feed)
            inst_E.append(e)
            print e

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

        sys.stdout.write('\rTraining: 100%\n')
        sys.stdout.flush()

        # array to hold encoded outputs
        O = np.zeros((n, latent_dim))

        # array to hold labels chosen
        B = np.zeros((n))

        for i in range(n):
            test_feed = {inputs: train_X[None, i, :], keep_prob: 1.0}

            out = sess.run(model.encode, test_feed)

            O[i, :] = out
            B[i] = train_Y[i]

            # report training progress
            progress = int(float(i)/float(n)*100.0)
            sys.stdout.write('\rEncoding: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rEncoding: 100%\n')
        sys.stdout.flush()

        out_dir = sys.argv[7]

        np.savetxt(out_dir + 'train_latent.dat', O)
        np.savetxt(out_dir + 'train_label.dat', B)

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

#!/usr/bin/env python

import sys
import numpy as np
import os

def sampleDist(latent, label, num_samples, same=True):
    dist = np.zeros(num_samples)
    n = label.shape[0]
    for i in range(num_samples):
        # choose a random index r
        r = np.random.randint(n)

        if same:
            # choose random index k that is different than r but has same label
            while True:
                k = np.random.randint(n)

                if ((k != r) and (label[k] == label[r])):
                    break
        else:
            # choose random index k that is different than r and has diff label
            while True:
                k = np.random.randint(n)

                if ((k != r) and (label[k] != label[r])):
                    break

        # calculate the euclidean distance between the latent representations
        dist[i] = np.linalg.norm(latent[r] - latent[k], ord=2)

    # get average neighbor-to-neighbor diatance
    avg_dist = np.mean(dist)

    return avg_dist


if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=1000, suppress=True)

    # retrieve command line args
    if (len(sys.argv) < 2):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./statistics.py <model>')
        sys.exit()

    model_dir = sys.argv[1]

    if (model_dir == 'nbow/'):
        print('[INFO] getting NBOW performance statistics')
    elif (model_dir == 'lstm/'):
        print('[INFO] getting LSTM performance statistics')
    elif (model_dir == 'tcnn/'):
        print('[INFO] getting TCNN performance statistics')
    elif (model_dir == 'dcnn/'):
        print('[INFO] getting DCNN performance statistics')
        model_dir = 'dcnn/'
    else:
        print('[ERROR] unknown model %s; exiting' % model_dir)
        sys.exit()


    # get the q-type accuracies and best latent vectors
    qtype_a = np.loadtxt(model_dir + 'q-type/accuracy_log.txt')

    qtype_files = os.listdir(model_dir + 'q-type/')
    qtype_best_latent = sorted(qtype_files)[-1]
    qtype_best_label = qtype_best_latent.split('_')[0] + '_label_' + qtype_best_latent.split('_')[-1]
    qtype_best_latent = np.loadtxt(model_dir + 'q-type/' + qtype_best_latent)
    qtype_best_label = np.loadtxt(model_dir + 'q-type/' + qtype_best_label)

    # get the stackoverflow accuracies
    stk_a = np.loadtxt(model_dir + 'stk-ovflw/accuracy_log.txt')

    stk_files = os.listdir(model_dir + 'stk-ovflw/')
    stk_best_latent = sorted(stk_files)[-1]
    stk_best_label = stk_best_latent.split('_')[0] + '_label_' + stk_best_latent.split('_')[-1]
    stk_best_latent = np.loadtxt(model_dir + 'stk-ovflw/' + stk_best_latent)
    stk_best_label = np.loadtxt(model_dir + 'stk-ovflw/' + stk_best_label)

    # get the agnews accuracies
    agnews_a = np.loadtxt(model_dir + 'ag-news/accuracy_log.txt')

    agnews_files = os.listdir(model_dir + 'ag-news/')
    agnews_best_latent = sorted(agnews_files)[-1]
    agnews_best_label = agnews_best_latent.split('_')[0] + '_label_' + agnews_best_latent.split('_')[-1]
    agnews_best_latent = np.loadtxt(model_dir + 'ag-news/' + agnews_best_latent)
    agnews_best_label = np.loadtxt(model_dir + 'ag-news/' + agnews_best_label)

    # get output statistics
    avg_qtype_a = np.mean(qtype_a)
    std_qtype_a = np.std(qtype_a)
    avg_qtype_neigh_dist = sampleDist(qtype_best_latent, qtype_best_label, 100, same=True)
    avg_qtype_non_neigh_dist = sampleDist(qtype_best_latent, qtype_best_label, 100, same=False)

    avg_stk_a = np.mean(stk_a)
    std_stk_a = np.std(stk_a)
    avg_stk_neigh_dist = sampleDist(stk_best_latent, stk_best_label, 100, same=True)
    avg_stk_non_neigh_dist = sampleDist(stk_best_latent, stk_best_label, 100, same=False)

    avg_agnews_a = np.mean(agnews_a)
    std_agnews_a = np.std(agnews_a)
    avg_agnews_neigh_dist = sampleDist(agnews_best_latent, agnews_best_label, 100, same=True)
    avg_agnews_non_neigh_dist = sampleDist(agnews_best_latent, agnews_best_label, 100, same=False)

    # print statistics
    print('[INFO] Q-Type Accuracy: %.2f +- %.2f' % (avg_qtype_a, std_qtype_a))
    print('[INFO] StackOverflow Accuracy: %.2f +- %.2f' % (avg_stk_a, std_stk_a))
    print('[INFO] AG-News Accuracy: %.2f +- %.2f' % (avg_agnews_a, std_agnews_a))

    # get averages of all neigh-to-neigh distances
    avg_neigh_dist = (avg_qtype_neigh_dist + avg_stk_neigh_dist + avg_agnews_neigh_dist)/3.0
    avg_non_neigh_dist = (avg_qtype_non_neigh_dist + avg_stk_non_neigh_dist + avg_agnews_non_neigh_dist)/3.0

    print_args = (avg_neigh_dist, avg_non_neigh_dist)
    print('[INFO] Neigh/Non-Neigh Dist: [%.2f, %.2f] ' % print_args)

#!/usr/bin/env python

import sys
import numpy as np
import os


if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=1000, suppress=True)

    # retrieve command line args
    if (len(sys.argv) < 2):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./statistics.py <data_dir>')
        sys.exit()

    data_dir = sys.argv[1]

    if (data_dir == 'q-type/'):
        print('[INFO] getting Question-Type performance statistics')
    elif (data_dir == 'stk-ovflw/'):
        print('[INFO] getting StackOverflow performance statistics')
    elif (data_dir == 'ag-news/'):
        print('[INFO] getting AG-News performance statistics')
    else:
        print('[ERROR] %s not a valid dataset; exiting' % data_dir)
        sys.exit()

    # get scores for each model on thie chosen dataset
    nbow_ami = np.loadtxt('nbow/' + data_dir + 'ami_log.txt')
    nbow_f = np.loadtxt('nbow/' + data_dir + 'f_log.txt')

    lstm_ami = np.loadtxt('lstm/' + data_dir + 'ami_log.txt')
    lstm_f = np.loadtxt('lstm/' + data_dir + 'f_log.txt')

    tcnn_ami = np.loadtxt('tcnn/' + data_dir + 'ami_log.txt')
    tcnn_f = np.loadtxt('tcnn/' + data_dir + 'f_log.txt')

    dcnn_ami = np.loadtxt('dcnn/' + data_dir + 'ami_log.txt')
    dcnn_f = np.loadtxt('dcnn/' + data_dir + 'f_log.txt')

    avg_nbow_ami = np.mean(nbow_ami)
    std_nbow_ami = np.std(nbow_ami)
    avg_nbow_f = np.mean(nbow_f)
    std_nbow_f = np.std(nbow_f)

    avg_lstm_ami = np.mean(lstm_ami)
    std_lstm_ami = np.std(lstm_ami)
    avg_lstm_f = np.mean(lstm_f)
    std_lstm_f = np.std(lstm_f)

    avg_tcnn_ami = np.mean(tcnn_ami)
    std_tcnn_ami = np.std(tcnn_ami)
    avg_tcnn_f = np.mean(tcnn_f)
    std_tcnn_f = np.std(tcnn_f)

    avg_dcnn_ami = np.mean(dcnn_ami)
    std_dcnn_ami = np.std(dcnn_ami)
    avg_dcnn_f = np.mean(dcnn_f)
    std_dcnn_f = np.std(dcnn_f)

    print('[INFO] Semi-Sup Performance on %s' % data_dir)

    print('[INFO] NBOW AMI: %.3f +- %.3f' % (avg_nbow_ami, std_nbow_ami))
    print('[INFO] NBOW F: %.3f +- %.3f' % (avg_nbow_f, std_nbow_f))

    print('[INFO] LSTM AMI: %.3f +- %.3f' % (avg_lstm_ami, std_lstm_ami))
    print('[INFO] LSTM F: %.3f +- %.3f' % (avg_lstm_f, std_lstm_f))

    print('[INFO] TCNN AMI: %.3f +- %.3f' % (avg_tcnn_ami, std_tcnn_ami))
    print('[INFO] TCNN F: %.3f +- %.3f' % (avg_tcnn_f, std_tcnn_f))

    print('[INFO] DCNN AMI: %.3f +- %.3f' % (avg_dcnn_ami, std_dcnn_ami))
    print('[INFO] DCNN F: %.3f +- %.3f' % (avg_dcnn_f, std_dcnn_f))

#!/usr/bin/env python

"""
Compares classifier A with classifier B with a confidence interval.

"""

import sys
import numpy as np
import os

import scipy.stats as stats

if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=1000, suppress=True)

    # retrieve command line args
    if (len(sys.argv) < 4):
        print('[ERROR] not enough cmd line arguments')
        print('[USAGE] ./t-test.py <acc_a> <acc_b> <conf>')
        sys.exit()

    acc_a = np.loadtxt(sys.argv[1])
    acc_b = np.loadtxt(sys.argv[2])
    conf = float(sys.argv[3])

    delta = acc_a - acc_b

    mu_delta = np.mean(delta)
    var_delta = np.var(delta)

    K = len(delta)

    Z_delta = np.sqrt(K) * mu_delta / np.sqrt(var_delta)

    #print Z_delta
    #print interval

    interval = stats.t.interval(conf, K-1)

    if ((Z_delta >= interval[0]) and (Z_delta <= interval[1])):
        print('No difference b/w A and B with %.2f confidence' % conf)
    elif (Z_delta < interval[0]):
        print('B better than A with %.2f confidence' % conf)
    else:
        print('A better than B with %.2f confidence' % conf)

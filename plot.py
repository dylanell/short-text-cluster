#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

l = np.loadtxt('loss.dat')

plt.plot(range(l.shape[0]), l)
plt.show()

# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#
import numpy as np
from matplotlib import pyplot as plt

with open('results.txt', 'r') as f:
    data = np.array([row.split() for row in f.read().split('\n')[:-1]]).astype(float)

plt.plot(data[:, -1], data[:, -2], 'k.', ms=10)
plt.show()

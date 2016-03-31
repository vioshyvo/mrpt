# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#
import numpy as np
from matplotlib import pyplot as plt

with open('results.txt', 'r') as f:
    data = np.array([row.split() for row in f.read().split('\n')[:-1]]).astype(float)

print len(data)
fig = plt.figure()
ax = fig.add_subplot(111)

# set your ticks manually
ax.plot(data[:, -1], data[:, -2], 'k.')#'o', mec='#EE6699', mfc='w', mew=2)
ax.yaxis.set_ticks(np.arange(0,1.1,0.1))
ax.grid(True)
plt.xlabel('Query time (s)')
plt.ylabel('Accuracy in 10NN search (MNIST data)')

plt.show()

#for d in data:
#    if d[-1] < 0.05 and d[-2] > 0.9:
#        print d[:-2].astype(int)
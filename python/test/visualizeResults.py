# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#
import numpy as np
from matplotlib import pyplot as plt

with open('votingAndPqResults.txt', 'r') as f:
    data = np.array([[x for x in row.split() if x is not 'None'] for row in f.read().split('\n')][:-1]).astype(float)

fig = plt.figure()
ax = fig.add_subplot(111)

# set your ticks manually
ax.plot(data[:, -1], data[:, -2], '.', color='#99DDDD')


with open('results.txt', 'r') as f:
    data = np.array([[x for x in row.split() if x is not 'None'] for row in f.read().split('\n')][:-1])
    data = np.array([x for x in data if x[4] == 'None'])
    data = np.concatenate((data[:, :4], data[:, 5:]), axis=1).astype(float)

print len(data)
# set your ticks manually
ax.plot(data[:, -1], data[:, -2], '.', color='#AA0000', ms=10)

ax.yaxis.set_ticks(np.arange(0,1.1,0.1))
ax.grid(True)
plt.axis((0,0.4,0,1))
plt.xlabel('Query time (s)')
plt.ylabel('Accuracy in 10NN search (MNIST data)')

#plt.savefig('mrptvsvoting.pdf', format='pdf', dpi=1000)
plt.show()


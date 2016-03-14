# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import time
import sys

import scipy.spatial.distance as ssd
import numpy as np

from utils.mnist_utils import *
from mrpt import *



########################################################################################################################
"""
BASIC SETTINGS
"""
if len(sys.argv) == 5:
    n0, n_trees, n_queries, k = [int(a) for a in sys.argv[1:5]]
elif len(sys.argv) == 2:
    n0, n_trees = [int(a) for a in sys.argv[1:3]]
    n_queries = 100
    k = 10
else:
    n0 = 32
    n_trees = 32
    n_queries = 100
    k = 10

print 'Looking for '+str(k)+' nearest neighbors. Averaging over '+str(n_queries)+' queries.'
print 'Building index with T='+str(n_trees)+', n0='+str(n0)+'.'
########################################################################################################################
"""
LOAD MNIST DATA
"""
imgs, labels = mnist_read_train_data()
test_imgs, test_labels = mnist_read_test_data()
queries = [test_imgs[i] for i in np.random.permutation(len(test_imgs))[:n_queries]]
N, dim = imgs.shape

########################################################################################################################
"""
BUILD INDEX
"""

start_time = time.time()
index = MRPTIndex(imgs, n0=n0, n_trees=n_trees)
print 'Index build time: '+str(time.time()-start_time)

########################################################################################################################
"""
MRPT QUERY TIMING
"""

neighbors_approximate = list()
start_time = time.time()
for img in queries:
    neighbors_approximate.append(index.ann(img, k))
print 'Avg mrpt query time: ' + str((time.time()-start_time)/n_queries)

########################################################################################################################
"""
LINEAR SEARCH AND COMPARISON
"""
neighbors_exact = list()
start_time = time.time()
for img in queries:
    neighbors_exact.append(np.argsort(ssd.cdist([img], imgs)[0])[:k])
print 'Avg linear query time: '+str((time.time()-start_time)/n_queries)

accuracy = [len(set(neighbors_approximate[i]).intersection(neighbors_exact[i]))/float(k) for i in range(n_queries)]
print 'MRPT (T='+str(n_trees)+', n0='+str(n0)+') accuracy: ' + str(np.mean(accuracy))

########################################################################################################################

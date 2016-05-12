# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from cppmrpt import *
from utils.mnist_utils import *
import numpy as np
from numpy import random as nr
from scipy.spatial import distance as ssd
from matplotlib import pyplot as plt
import time


def mrpt_example(n0, n_trees, n_extra_branches, n_elected):
    """
    This function demonstrates the use of MRPT on the MNIST data set. The function prints information about its
    progress. Please be patient, the total running time may be several minutes depending on the parameters.
    :param n0: The leaf size of the random projection trees in the MRPT index
    :param n_trees: The number of RP trees in the MRPT index
    :param n_extra_branches: The number of extra branches explored in ann-queries
    :param n_elected: The number of elected neighbors-candidates, that go into the exact linear search
    :return: -
    """
    print 'Loading data, solving exact neighbors...'
    n_queries = 100
    train_images = mnist_read_train_data()[0]
    test_images = mnist_read_test_data()[0]
    test_images = test_images[nr.permutation(len(test_images))[:n_queries]]  # Picking 100 random objects as queries
    e_neighbors = np.argsort(ssd.cdist(test_images, train_images), axis=1)[:, :10]

    print 'Building index...'
    t = time.time()
    index = MRPTIndex(train_images, n0=n0, n_trees=n_trees)
    print 'Took ' + '%.2f' % (time.time()-t) + 'sec'

    print 'Doing ' + '%d' % n_queries + ' queries...'
    t = time.time()
    neighbors = np.zeros((n_queries, 10))
    for i in range(n_queries):
        neighbors[i] = index.ann(test_images[i], 10, n_extra_branches=n_extra_branches, votes_required=n_elected)
    print 'Took '+ '%.2f' % (time.time()-t) + 'sec'

    correct_neighbors = np.zeros(n_queries)
    for i in range(n_queries):
        correct_neighbors[i] = len(np.intersect1d(e_neighbors[i], neighbors[i]))
    plt.hist(correct_neighbors, bins=np.arange(-0.5, 11.5, 1))
    plt.title('MRPT: n0='+ '%d' % n0 + ', n_trees=' + '%d' % n_trees+', n_extra_branches=' + '%d' % n_extra_branches+', n_elected=' + '%d' % n_elected)
    plt.xlabel('Number of correct neighbors (max 10)')
    plt.ylabel('Number of test queries')
    plt.show()

    print '%.1f ' % (1.0*sum(correct_neighbors)/n_queries) + 'neighbors (out of 10) correct on average'


# An OK choice of parameters with no bells and whistles employed. Should get ~90% performance.
print '=========================\nRun 1: Basic parameters'
mrpt_example(n0=32, n_trees=75, n_extra_branches=0, n_elected=0)

# A good choice of advanced parameters for MNIST. Should get to ~90% performance.
print '=========================\nRun 2: Advanced parameters'
mrpt_example(n0=512, n_trees=20, n_extra_branches=100, n_elected=200)


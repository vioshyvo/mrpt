# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from mrpt import *
import numpy as np
from utils.mnist_utils import *
import time


def explore_mnist(iters=1):
    trainimgs, trainlbls = mnist_read_train_data('../../datasets/mnist/train-images.idx3-ubyte',
                                          '../../datasets/mnist/train-labels.idx1-ubyte')
    testimgs, testlbls = mnist_read_test_data('../../datasets/mnist/t10k-images.idx3-ubyte',
                                          '../../datasets/mnist/t10k-labels.idx1-ubyte')
    query_indexes, e_neighbors = load_exact_results()
    queries = testimgs[query_indexes]

    for i in range(iters):
        n0, n_trees, degree, n_extra_branches, n_elected = choose_params(trainimgs, 10)
        print n0, n_trees, degree, n_extra_branches, n_elected
        index = MRPTIndex(trainimgs, n0, n_trees, degree, use_saved=False)

        a_neighbors = []
        start_time = time.time()
        for query in queries:
            a_neighbors.append(index.ann(query, 10, n_extra_branches, n_elected))
        avg_query_time = (time.time()-start_time)/100

        correct_approximates = [len(np.intersect1d(a_neighbors[i], e_neighbors[i])) for i in range(len(a_neighbors))]
        accuracy = sum(correct_approximates)/1000.0

        with open('results.txt', "a") as f:
            f.write(str(n0)+' '+str(n_trees)+' '+str(degree)+' '+str(n_extra_branches)+' '+str(n_elected)+' ' +
                    str(accuracy)+' '+str(avg_query_time)+'\n')


def load_exact_results():
    d = np.load('trueresults.npz')
    query_indexes = d['arr_0']
    neighbors = d['arr_1']
    return query_indexes, neighbors


def choose_params(data, k):
    n0 = pow(2, np.random.randint(0, 12))
    n_trees = int(np.random.beta(2, 5)*min(300, len(data)/n0))
    degree = 2
    n_extra_branches = int(np.random.beta(2, 2)*1000)
    n_elected = k + int(np.random.beta(2, 5)*1000)
    return n0, n_trees, degree, n_extra_branches, n_elected

explore_mnist(3)
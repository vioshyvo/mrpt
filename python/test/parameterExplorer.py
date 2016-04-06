# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import sys
sys.path.append('/cs/work/home/teempitk/mrpt/python')
sys.path.append('/Users/teemu/Developer/mrpt/python')
from mrpt import *
import numpy as np
from utils.mnist_utils import *
import time


def explore_news(iters=1):
    data = np.load('../../datasets/news/newsdata_train.npy')
    queries = np.load('../../datasets/news/newsdata_test.npy')
    exact_results = np.load('../../datasets/news/newsdata_true_neighbors.npy')

    for i in range(iters):
        n0, n_trees, degree, n_extra_branches, n_elected = choose_params_news(data, 10)
        print n0, n_trees, degree, n_extra_branches, n_elected
        index = MRPTIndex(data, n0, n_trees, degree, use_saved=False)

        a_neighbors = []
        start_time = time.time()
        for query in queries:
            a_neighbors.append(index.ann(query, 10, n_extra_branches, n_elected))
        avg_query_time = (time.time()-start_time)/100

        correct_approximates = [len(np.intersect1d(a_neighbors[i], exact_results[i])) for i in range(len(a_neighbors))]
        accuracy = sum(correct_approximates)/1000.0

        with open('news_results.txt', "a") as f:
            f.write(str(n0)+' '+str(n_trees)+' '+str(degree)+' '+str(n_extra_branches)+' '+str(n_elected)+' ' +
                    str(accuracy)+' '+str(avg_query_time)+'\n')


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

        with open('rptResults.txt', "a") as f:
            f.write(str(n0)+' '+str(n_trees)+' '+str(degree)+' '+str(n_extra_branches)+' '+str(n_elected)+' ' +
                    str(accuracy)+' '+str(avg_query_time)+'\n')


def load_exact_results():
    d = np.load('trueresults.npz')
    query_indexes = d['arr_0']
    neighbors = d['arr_1']
    return query_indexes, neighbors


def choose_params_news(data, k):
    n0 = 512
    n_trees = 33
    n_elected = 149
    degree = 2
    n_extra_branches = 111
    # SINGLE RPT
    #n0 = pow(2, np.random.randint(0, 16))
    #n_trees = 1
    #degree = 2
    #n_extra_branches = 0
    #n_elected = None
    return n0, n_trees, degree, n_extra_branches, n_elected


def choose_params(data, k):
    # SINGLE RPT
    n0 = pow(2, np.random.randint(0, 16))
    n_trees = 1
    degree = 2
    n_extra_branches = 0
    n_elected = None

    # Just voting
    #n0 = pow(2, np.random.randint(0, 14))
    #n_trees = int(np.random.beta(2, 5)*300)
    #degree = 2
    #n_extra_branches = 0
    #n_elected = k + int(np.random.beta(2, 5)*1000)

    # Normal mrpt
    #n0 = pow(2, np.random.randint(0, 12))
    #n_trees = int(np.random.beta(2, 5)*min(300, len(data)/n0))
    #degree = 2
    #n_extra_branches = 0
    #n_elected = None

    # voting + pq
    #n0 = pow(2, np.random.randint(0, 12))
    #n_trees = int(np.random.beta(2, 5)*min(300, len(data)/n0))
    #degree = 2
    #n_extra_branches = int(np.random.beta(2, 5)*500)
    #n_elected = k + int(np.random.beta(2, 5)*1000)
    return n0, n_trees, degree, n_extra_branches, n_elected
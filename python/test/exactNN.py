# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import scipy.spatial.distance as ssd
from utils.mnist_utils import *
import numpy as np

imgs, lbls = mnist_read_train_data('../../datasets/mnist/train-images.idx3-ubyte',
                                   '../../datasets/mnist/train-labels.idx1-ubyte')
testimgs, testlbls = mnist_read_test_data('../../datasets/mnist/t10k-images.idx3-ubyte',
                                          '../../datasets/mnist/t10k-labels.idx1-ubyte')

query_indexes = np.random.permutation(len(testimgs))[:100]
queries = testimgs[query_indexes]

neighbors = []

for query in queries:
    neighbors.append(np.argsort(ssd.cdist([query], imgs)[0])[:10])

np.savez('trueresults', query_indexes, neighbors)

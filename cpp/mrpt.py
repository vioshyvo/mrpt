# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import numpy as np

import mrptlib


class MRPTIndex(object):
    """
    Wraps the extension module written in C++
    """
    def __init__(self, data, depth, n_trees,
                 projection_sparsity=None, metric='euclidean', shape=None, sparse=False, mmap=False):
        """
        Initializes an MRPT index object.
        :param data: Input data either as a NxDim numpy ndarray or as a filepath to a binary file containing the data
        :param depth: The depth of the trees
        :param n_trees: The number of trees used in the index
        :param projection_sparsity: Expected ratio of non-zero components in a projection matrix
        :param metric: Distance metric to use, currently euclidean or angular
        :param shape: Shape of the data as a tuple (N, dim). Needs to be specified only if loading the data from a file.
        :param sparse: Set to True if the data should be treated as sparse
        :param mmap: If true, the data is mapped into memory. Has effect only if the data is loaded from a file.
        :return:
        """
        if isinstance(data, np.ndarray):
            if len(data) == 0 or len(data.shape) != 2:
                raise ValueError("Invalid data matrix")
            if not data.flags['C_CONTIGUOUS'] or not data.flags['ALIGNED']:
                raise ValueError("The data matrix has to be C_CONTIGUOUS and ALIGNED")
            n_samples, dim = data.shape
        elif isinstance(data, str):
            if not isinstance(shape, tuple) or len(shape) != 2:
                raise ValueError("You must specify the shape of the data as a tuple (N, dim) "
                                 "when loading data from a binary file")
            n_samples, dim = shape
        else:
            raise ValueError("Data must be either an ndarray or a filepath")

        max_depth = np.ceil(np.log2(n_samples))
        if not 1 <= depth <= max_depth:
            raise ValueError("Depth should be in range [1, %d]" % max_depth)

        if n_trees < 1:
            raise ValueError("Number of trees must be positive")

        if projection_sparsity == 'auto':
            projection_sparsity = 1. / np.sqrt(dim)
        elif projection_sparsity is None:
            projection_sparsity = 1
        elif not 0 < projection_sparsity <= 1:
            raise ValueError("Sparsity should be in (0, 1]")

        if metric not in ('euclidean', 'angular'):
            raise ValueError("Metric must be euclidean or angular")
        metric_val = 1 if metric == 'angular' else 0

        self.index = mrptlib.MrptIndex(data, n_samples, dim, depth, n_trees, projection_sparsity, metric_val, sparse, mmap)
        self.built = False

    def build(self):
        """
        Builds the MRPT index.
        :return:
        """
        self.index.build()
        self.built = True

    def save(self, path):
        """
        Saves the MRPT index to a file.
        :param path: Filepath to the location of the saved index.
        :return:
        """
        if not self.built:
            raise RuntimeError("Cannot save index before building")
        self.index.save(path)

    def load(self, path):
        """
        Loads the MRPT index from a file.
        :param path: Filepath to the location of the index.
        :return:
        """
        self.index.load(path)

    def ann(self, q, k, n_extra_branches=0, votes_required=1):
        """
        The MRPT approximate nearest neighbor query.
        :param q: The query object, i.e. the vector whose nearest neighbors are searched for
        :param k: The number of neighbors the user wants the query to return
        :param n_extra_branches: The number of extra branches to be explored by the priority queue trick
        :param votes_required: The number of votes an object has to get to be included in the linear search part of the query.
        :return: The indices of the approximate nearest neighbors in the original input data given to the constructor.
        """
        return self.index.ann(q, k, votes_required, n_extra_branches)

    def parallel_ann(self, Q, k, n_extra_branches=0, votes_required=1):
        """
        Parallel version of the MRPT approximate nearest neighbor query for performing several
        queries at once. The queries are given as a numpy matrix where each row contains a query.
        :param Q: The query matrix, i.e. a matrix where each row is a vector whose nearest neighbors are searched for
        :param k: The number of neighbors the user wants the query to return
        :param n_extra_branches: The number of extra branches to be explored by the priority queue trick
        :param votes_required: The number of votes an object has to get to be included in the linear search part of the query.
        :return: Matrix of indices where each row contains the indices of the approximate nearest neighbors in the original
                 input data for the corresponding query.
        """
        return self.index.parallel_ann(Q, k, votes_required, n_extra_branches)

    def exact_search(self, Q, k):
        """
        Performs an exact nearest neighbor query for several queries in parallel. The queries are
        given as a numpy matrix where each row contains a query. Useful for measuring accuracy.
        :param q: The query object, i.e. the vector whose nearest neighbors are searched for
        :param k: The number of neighbors the user wants the query to return
        :return: Matrix of indices where each row contains the indices of the nearest neighbors in the original
                 input data for the corresponding query.
        """
        return self.index.exact_search(Q, k)

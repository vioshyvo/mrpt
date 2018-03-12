import os
import numpy as np

import mrptlib


class MRPTIndex(object):
    """
    Wraps the extension module written in C++
    """
    def __init__(self, data, depth, n_trees, projection_sparsity='auto', shape=None, mmap=False, sparse=False):
        """
        Initializes an MRPT index object.
        :param data: Input data either as a NxDim numpy ndarray or as a filepath to a binary file containing the data
        :param depth: The depth of the trees
        :param n_trees: The number of trees used in the index
        :param projection_sparsity: Expected ratio of non-zero components in a projection matrix
        :param shape: Shape of the data as a tuple (N, dim). Needs to be specified only if loading the data from a file.
        :param mmap: If true, the data is mapped into memory. Has effect only if the data is loaded from a file.
        :param sparse: Set to True if the data should be treated as sparse
        :return:
        """
        if isinstance(data, np.ndarray):
            if len(data) == 0 or len(data.shape) != 2:
                raise ValueError("The data matrix should be non-empty and two-dimensional")
            if data.dtype != np.float32:
                raise ValueError("The data matrix should have type float32")
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

        if mmap and os.name == 'nt':
            raise ValueError("Memory mapping is not available on Windows")

        if projection_sparsity < 1 and sparse:
            raise ValueError("Combining sparse data and sparse projections is unsupported")

        self.index = mrptlib.MrptIndex(data, n_samples, dim, depth, n_trees, projection_sparsity, mmap, sparse)
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
        self.built = True

    def ann(self, q, k, votes_required=1):
        """
        The MRPT approximate nearest neighbor query.
        :param q: The query object, i.e. the vector whose nearest neighbors are searched for
        :param k: The number of neighbors the user wants the query to return
        :param votes_required: The number of votes an object has to get to be included in the linear search part of the query.
        :return: The indices of the approximate nearest neighbors in the original input data given to the constructor.
        """
        if not self.built:
            raise RuntimeError("Cannot query before building index")
        return self.index.ann(q, k, votes_required)

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

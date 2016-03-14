# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from rptree import *
import scipy.spatial.distance as ssd


class MRPTIndex(object):
    """
    The classic-style MRPT index, which is basically just a collection of RP-trees. Each query is performed in each of
    the trees, and the results are combined to find the best k approximate neighbors.
    """
    def __init__(self, data, n0=32, n_trees=32, n_elected=-1):
        self.trees = [RPTree(data, n0) for t in range(n_trees)]
        self.data = data
        self.n_elected = n_elected

    def ann(self, obj, k):
        if self.n_elected > 0:
            return self.vann(obj, k)
        neighborhood = set()
        for tree in self.trees:
            neighborhood = neighborhood.union(tree.find_leaf(obj))
        neighborhood = list(neighborhood)
        return [neighborhood[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in neighborhood])[0])[:k]]

    def vann(self, obj, k):
        """
        The voting-enhanced MRPT-ANN query. In queries each potential approximate neighbor suggested by a tree counts as
        a vote. Only the objects with the highest number votes are actually compared at the end of the search.
        Has the potential to provide big improvements in query time.
        """
        votes = np.zeros(len(self.data))
        for tree in self.trees:
            for vote in tree.find_leaf(obj):
                votes[vote] += 1

        elected = np.argsort(votes)[len(votes)-1:len(votes)-1-self.n_elected:-1]
        return [elected[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in elected])[0])[:k]]


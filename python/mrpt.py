# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from rptree import *
import scipy.spatial.distance as ssd
import hashlib as hl


class MRPTIndex(object):
    """
    The MRPT index is basically just a collection of RP trees. Query results are formed by combining the results of
    single trees. The constructor builds a user-specified number of trees with a user-specified maximum leaf-size. The
    current version can use pre-built saved trees to speed up -- if there are trees built for the same data set with the
    same leaf size those will be used. If the use_saved option is allowed but the index needs more trees than there
    exists in the pre-built collection, as many new trees will be built as needed. The new trees are also added to the
    collection.
    """
    def __init__(self, data, n0=10, n_trees=32, use_saved=False):
        self.data = data
        if use_saved:
            self.trees = []
            save_path = 'saved_trees/'+hl.sha1(data.view(np.uint8)).hexdigest()[:8]+'/'+str(n0)
            self.trees = load_trees(save_path, n_trees)
            for t in range(len(self.trees), n_trees):
                t = RPTree(data, n0)
                self.trees.append(t)
                save_tree(t, save_path)
        else:
            self.trees = [RPTree(data, n0) for t in range(n_trees)]

    def ann(self, obj, k=10):
        """
        The classic-style MRPT query which is performed in each of the trees, and the results are combined to find the
        best k approximate neighbors.
        :param obj: The vector whose neighbors are being searched for
        :param k: The number of neighbors
        :return: The indices of the approximate neighbors in the data set
        """
        neighborhood = set()
        for tree in self.trees:
            neighborhood = neighborhood.union(tree.find_leaf(obj))
        neighborhood = list(neighborhood)
        return [neighborhood[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in neighborhood])[0])[:k]]

    def vann(self, obj, k=10, n_elected=500):
        """
        The voting-enhanced MRPT query. In queries each potential approximate neighbor suggested by a tree counts as
        a vote. Only the objects with the highest number votes are actually compared at the end of the search.
        Has the potential to provide big improvements in query time.
        :param obj: The vector whose neighbors are being searched for
        :param k: The number of neighbors
        :param n_elected: The number of data objects whose distances to the query are really computed
        :return: The indices of the approximate neighbors in the data set
        """
        votes = np.zeros(len(self.data))
        for tree in self.trees:
            for vote in tree.find_leaf(obj):
                votes[vote] += 1
        elected = np.argsort(votes)[len(votes)-1:len(votes)-1-n_elected:-1]
        return [elected[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in elected])[0])[:k]]


# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from rptree import *
import scipy.spatial.distance as ssd
import cPickle


class MRPTIndex:

    def __init__(self, data, n0=10, n_trees=50):
        self.trees = [RPTree(data, n0) for t in range(n_trees)]
        self.data = data

    def ann(self, obj, k):
        neighborhood = set()
        for tree in self.trees:
            neighborhood = neighborhood.union(tree.find_leaf(obj))
        neighborhood = list(neighborhood)
        return [neighborhood[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in neighborhood])[0])[:k]]

'''
# Need a method to save without the data!!! --update: still saving in single trees---
def save(index, filename='mrpt.idx'):
    with open(filename, 'w') as f:
        cPickle.dump(index.trees, f)


def load(data, filename='mrpt.idx'):
    with open(filename, 'r') as f:
        index = MRPTIndex(data, n0=0, n_trees=0)
        index.trees = cPickle.load(f)  # Bad convention. Create method set_trees for mrptindex...
        return index
'''
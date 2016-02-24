# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from rptree import *
import scipy.spatial.distance as ssd


class mrpt_index:

    def __init__(self, data, n0=10, T=10):
        self.trees = [RPTree(data, n0) for t in range(T)]
        self.data = data

    def ann(self, obj, k):
        pool = set()
        for tree in self.trees:
            pool=pool.union(tree.find_leaf(obj))
        pool = list(pool)
        return [pool[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in pool])[0])[:k]]

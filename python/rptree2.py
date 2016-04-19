# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import math
import numpy as np
from collections import deque


class RPTree(object):
    """
    A LIGHTER VERSION OF A RP-TREE. BENEFITS ARE QUESTIONABLE.
    A Random Projection Tree is a spatial index data structure used in specific data mining and indexing tasks. Our
    implementation has been built with approximate nearest neighbor search (ANN) problems in mind. A tree is built by
    dividing the data space by random hyperplanes into small cells. In ANN-problems we usually achieve substantial
    improvements by quickly choosing just a subset of the data located in a single cell where the actual brute-force NN
    search is then performed, instead of using the whole data set.
    """

    def __init__(self, data, n0):
        """
        Sets the main attributes and calls the build_tree routine which actually creates the nodes and builds the tree.
        :param data: The data for which the index is built
        :param n0: The maximum leaf size of the tree
        """
        self.seed = np.random.randint(0, int(1e9))
        self.tree_height = int(math.ceil(np.log2(len(data)/float(n0))))
        self.splits = np.zeros(shape=(self.tree_height, 2**self.tree_height))
        np.random.seed(self.seed)
        projections = np.dot(data, np.random.normal(size=(data.shape[1], self.tree_height)))

        prev_level = [np.arange(len(data))]
        next_level = []
        for i in range(self.tree_height):
            for j in range(len(prev_level)):
                idx = prev_level[j]
                split = np.median(projections[idx, i])
                self.splits[i, j] = split
                left_idx = projections[idx, i] < split
                next_level.append(idx[left_idx])
                next_level.append(idx[np.logical_not(left_idx)])
            prev_level = next_level
            next_level = []
        self.leaves = prev_level

    def full_tree_traversal(self, obj):
        """
        The function places the query object 'obj' to a leaf. The function re-creates the same random vectors that were
        used in tree construction, computes the projections of the query vector and using the split information stored
        in the nodes places the query vector into a single leaf.
        :param obj: The query object, has to be given as a row vector
        :return: The indexes of the leaf, gap information and projection values from the path to leaf (for the extra
        branches trick)
        """
        # Restore rng settings, compute projections to random basis
        np.random.seed(self.seed)
        projections = np.dot(obj, np.random.normal(size=(len(obj), self.tree_height)))

        # Move down the tree according to the projections and split values stored in the tree
        indexes, gaps = self.partial_tree_traversal(0, projections, 0)
        return indexes, gaps, projections

    def partial_tree_traversal(self, node, projections, tree_level):
        """
        The funtion traverses the tree down starting from the node given as parameter.
        :param node: The node where to start
        :param projections: Projections (pre-computed)
        :param tree_level: The level on which 'node' is
        :return: The indices of the leaf, the gaps and nodes where the search did not go
        """
        gaps = []
        for lvl in range(tree_level, self.tree_height):
            node *= 2
            gap = projections[lvl]-self.splits[lvl, node]
            if gap > 0:
                node += 1
                gaps.append((abs(gap), node-1, lvl+1))
            else:
                gaps.append((abs(gap), node+1, lvl+1))
        return self.leaves[node], gaps
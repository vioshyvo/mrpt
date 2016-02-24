# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import math
import numpy as np
from collections import deque


class RPTree:
    """
    This class implements the random projection tree data structure used for indexing a data set for quick approximate
    nearest neighbor queries.
    """

    def __init__(self, data, n0, seed=None):

        if seed is None:
            seed = np.random.randint(0, 1e9)
        self.seed = seed
        self.dim = data.shape[1]
        self.root = Node(range(data.shape[0]))
        self.tree_depth = 1

        self.build_tree(data, n0)

    def build_tree(self, data, n0):
        np.random.seed(self.seed)
        self.tree_depth = int(np.log2(len(data)/float(n0))+1)
        vectors = np.random.normal(size=(self.tree_depth, self.dim))
        all_projections = np.dot(vectors, data.T)
        queue = deque([self.root])
        level_size = 0
        level_capacity = 1
        curr_level = 0
        while len(queue) > 0:
            node = queue.popleft()
            indxs = node.get_indxs()
            node_size = len(indxs)

            projections = [all_projections[curr_level, i] for i in indxs]
            order = np.argsort(projections)

            # Create node objects for children
            left = Node([indxs[i] for i in order[:node_size / 2]])
            right = Node([indxs[i] for i in order[node_size / 2:]])
            division = (projections[order[node_size / 2]] + projections[
                order[int(math.ceil(node_size / 2.0) - 1)]]) / 2.0
            node.set_children(left, right, division)

            # Add new nodes to queue to be split if necessary
            if node_size / 2 > n0:
                queue.append(left)
                queue.append(right)

            level_size += 1
            if level_size == level_capacity:
                level_size = 0
                level_capacity *= 2
                curr_level += 1

    def find_leaf(self, obj):
        np.random.seed(self.seed)
        projections = np.dot(np.random.normal(size=(self.tree_depth, self.dim)), obj)

        node = self.root
        i = 0
        while node.left is not None:
            if projections[i] < node.division:
                node = node.left
            else:
                node = node.right
            i += 1
        return node.indxs


class Node:
    """
    This class describes a single node of the rp-tree.

    """
    def __init__(self, indxs):
        self.indxs = indxs  # Can be removed from inner nodes for memory efficiency
        self.left = None  # Not required for child nodes for memory efficiency
        self.right = None  # Not required for child nodes for memory efficiency
        self.division = None  # Not required for child nodes for memory efficiency

    def set_children(self, left, right, division):
        self.left = left
        self.right = right
        self.division = division

    def get_indxs(self):
        return self.indxs


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
        self.tree_depth = int(np.log2(len(data)/float(n0))+1)
        self.build_tree(data, n0)

    def build_tree(self, data, n0):
        """
        This method builds the random projection tree using projections on random vectors.
        """
        # Restore rng settings for reproducibility and compute projections to random basis
        np.random.seed(self.seed)
        all_projections = np.dot(np.random.normal(size=(self.tree_depth, self.dim)), data.T) # Possible to do without transpose

        # Main while loop that builds the tree one level at a time
        level_size = 0
        level_capacity = 1
        curr_level = 0
        queue = deque([self.root])
        while len(queue) > 0:
            # Pop next node to be processed
            node = queue.popleft()
            indexes = node.get_indexes()
            node_size = len(indexes)

            # Sort the projections and define the split
            projections = [all_projections[curr_level, i] for i in indexes]
            order = np.argsort(projections)
            division = (projections[order[node_size / 2]] + projections[
                order[int(math.ceil(node_size / 2.0) - 1)]]) / 2.0

            # Create new nodes, add to queue if their size requires a split
            left = Node([indexes[i] for i in order[:node_size / 2]])
            right = Node([indexes[i] for i in order[node_size / 2:]])
            node.set_children(left, right, division)
            if node_size / 2 > n0:
                queue.append(left)
                queue.append(right)

            # For keeping track on which tree level the loop operates
            level_size += 1
            if level_size == level_capacity:
                level_size = 0
                level_capacity *= 2
                curr_level += 1

    def find_leaf(self, obj):
        """
        This function routes the query data object to a leaf and returns the index object indices of that leaf
        """
        # Restore rng settings, compute projections to random basis
        np.random.seed(self.seed)
        projections = deque(np.dot(np.random.normal(size=(self.tree_depth, self.dim)), obj))

        # Move down the tree according to the projections and split values stored in the tree
        node = self.root
        while node.left is not None:
            if projections.popleft() < node.division:
                node = node.left
            else:
                node = node.right
        return node.get_indexes()


class Node:
    """
    This class defines the structure of the rp-tree nodes

    """
    def __init__(self, indexes):
        self.indexes = indexes
        self.left = None
        self.right = None
        self.division = None

    def set_children(self, left, right, division):
        self.left = left
        self.right = right
        self.division = division

    def get_indexes(self):
        return self.indexes


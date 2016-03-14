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
    This class implements the random projection tree data structure used for indexing a data set for quick approximate
    nearest neighbor queries.
    """

    def __init__(self, data, n0):
        self.dim = data.shape[1]
        self.seed = np.random.randint(0, int(1e9))
        self.n0 = n0
        self.n_random_vectors = math.ceil(np.log2(len(data)/float(n0)))

        self.root = Node()
        self.build_tree(data, n0)

    def build_tree(self, data, n0):
        """
        This method builds the random projection tree using projections on random vectors.
        """
        # Restore rng settings for reproducibility and compute projections to random basis
        np.random.seed(self.seed)
        all_projections = np.dot(data, np.random.normal(size=(self.dim, self.n_random_vectors)))

        # Main while loop that builds the tree one level at a time
        level_size = 0
        level_capacity = 1
        curr_level = 0
        queue = deque([(self.root, range(len(data)))])
        while len(queue) > 0:
            # Pop next node to be processed
            node, indexes = queue.popleft()
            node_size = len(indexes)

            # Sort the projections and define the split
            projections = [all_projections[i, curr_level] for i in indexes]
            order = np.argsort(projections)
            node.set_split((projections[order[node_size / 2]] + projections[
                order[int(math.ceil(node_size / 2.0) - 1)]]) / 2.0)

            # Create new nodes, add to queue if their size requires a split
            l_indexes = [indexes[i] for i in order[:math.ceil(node_size/2.0)]]
            r_indexes = [indexes[i] for i in order[math.ceil(node_size/2.0):]]

            if node_size >= 2*n0 + 1:
                left = Node()
                node.set_left(left)
                queue.append((left, l_indexes))
                if node_size > 2*n0 + 1:
                    right = Node()
                    node.set_right(right)
                    queue.append((right, r_indexes))
                else:
                    node.set_right(Leaf(r_indexes))
            else:
                node.set_left(Leaf(l_indexes))
                node.set_right(Leaf(r_indexes))

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
        projections = deque(np.dot(obj, np.random.normal(size=(self.dim, self.n_random_vectors))))

        # Move down the tree according to the projections and split values stored in the tree
        node = self.root
        while node.left is not None:
            if projections.popleft() < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.get_indexes()

    def get_n0(self):
        return self.n0


class Node(object):
    """
    This class defines the structure of the rp-tree nodes

    """
    def __init__(self):
        self.left = None
        self.right = None
        self.split_threshold = None
        self.indexes = None

    def set_indexes(self, indexes):
        self.indexes = indexes

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def set_split(self, split):
        self.split_threshold = split


class Leaf(Node):
    def __init__(self, indexes):
        super(Leaf, self).__init__()
        self.indexes = indexes

    def get_indexes(self):
        return self.indexes

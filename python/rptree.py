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
    This class contains the implementation of a single random projection tree. Only median splits are used for now.
    """

    def __init__(self, data, indxs, n0, seed=None):
        if seed is None:
            seed = np.random.randint(0, 1e9)
        self.seed = seed
        self.dim = data.shape[1]
        self.root = Node(indxs)

        queue = deque([self.root])
        np.random.seed(self.seed)

        # The following keep track on the level of the tree and are used to indicate when a new random vector is needed
        level_size = 1
        level_capacity = 1

        while len(queue) > 0:
            if level_size == level_capacity:
                level_size = 0
                level_capacity *= 2
                vector = np.random.normal(size=self.dim)
            level_size += 1

            # Pop next node from the queue
            node = queue.popleft()
            indxs = node.get_indxs()
            size = len(indxs)
            # Compute and sort the projections
            projections = [np.dot(vector, obj) for obj in [data[i] for i in indxs]]

            order = np.argsort(projections)

            # Create node objects for children
            left = Node([indxs[i] for i in order[:size/2]])
            right = Node([indxs[i] for i in order[size/2:]])
            division = (projections[order[size/2]] + projections[order[int(math.ceil(size/2.0) - 1)]])/2.0
            node.set_children(left, right, division)

            # Add new nodes to queue to be split if necessary
            if len(projections)/2 > n0:
                queue.append(left)
                queue.append(right)

    def query(self, obj):
        np.random.seed(self.seed)
        node = self.root
        while node.left is not None:
            vector = np.random.normal(size=self.dim)
            projection = np.dot(obj, vector)
            if projection < node.division:
                node = node.left
            else:
                node = node.right
        return node.indxs


class Node:
    """
    This class describes a single node of the rp-tree.

    """
    def __init__(self, indxs):
        self.indxs = indxs # Can be removed from inner nodes
        self.left = None # Not required for child nodes
        self.right = None # Not required for child nodes
        self.division = None # Not required for child nodes

    def set_children(self, left, right, division):
        self.left = left
        self.right = right
        self.division = division

    def get_indxs(self):
        return self.indxs


# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import math
import numpy as np
from collections import deque
import os
import cPickle


class RPTree(object):
    """
    A Random Projection Tree is a spatial index data structure used in specific data mining and indexing tasks. Our
    implementation has been built with approximate nearest neighbor search (ANN) problems in mind. A tree is built by
    dividing the data space by random hyperplanes into small cells. In ANN-problems we usually achieve substantial
    improvements by quickly choosing just a subset of the data located in a single cell where the actual brute-force NN
    search is then performed.
    """

    def __init__(self, data, n0):
        self.seed = np.random.randint(0, int(1e9))
        self.tree_height = math.ceil(np.log2(len(data)/float(n0)))
        self.root = InnerNode()
        self.build_tree(data, n0)

    def build_tree(self, data, n0):
        """
        An iterative method for building the random projection tree structure. The tree is built level-by-level by
        using a queue to handle the order in which the nodes are processed. The method is significantly faster to
        recursive building. This implementation uses same random vectors in each tree branch for running
        time efficiency. Note that although all projections are computed in a single matrix multiplication, the
        projection vector is different on each level.
        """
        # Restore rng settings for reproducibility and compute projections to random basis
        np.random.seed(self.seed)
        all_projections = np.dot(data, np.random.normal(size=(data.shape[1], self.tree_height)))

        # Main while loop that builds the tree one level at a time
        queue = deque([(self.root, range(len(data)))])
        tracker = FullBinaryTreeLevelTracker()
        while len(queue) > 0:
            # Pop next node to be processed
            node, indexes = queue.popleft()
            node_size = len(indexes)

            # Sort the projections and define the split
            projections = [all_projections[i, tracker.get_level()] for i in indexes]
            order = np.argsort(projections)
            node.set_split((projections[order[node_size / 2]] + projections[
                order[int(math.ceil(node_size / 2.0) - 1)]]) / 2.0)

            # Split the indexes to child nodes. In case of uneven division the extra object goes to the left branch
            l_indexes = [indexes[i] for i in order[:math.ceil(node_size/2.0)]]
            r_indexes = [indexes[i] for i in order[math.ceil(node_size/2.0):]]

            # Set references to children, add child nodes to queue if further splits are required (node size > n0)
            if len(l_indexes) > n0:
                left = InnerNode()
                node.set_left(left)
                queue.append((left, l_indexes))
            else:
                node.set_left(LeafNode(l_indexes))

            if len(r_indexes) > n0:
                right = InnerNode()
                node.set_right(right)
                queue.append((right, r_indexes))
            else:
                node.set_right(LeafNode(r_indexes))

            tracker.object_added()  # Corresponds to adding _node_, not its children, thus called only once

    def find_leaf(self, obj):
        """
        The function re-creates the same random vectors that were used in tree construction, computes the projections of
        the query vector and using the split information stored in the nodes places the query vector into a single leaf.
        The query vector has to be given as a 1-dimensional array.
        """
        # Restore rng settings, compute projections to random basis
        np.random.seed(self.seed)
        projections = deque(np.dot(obj, np.random.normal(size=(len(obj), self.tree_height))))

        # Move down the tree according to the projections and split values stored in the tree
        node = self.root
        while not hasattr(node, 'get_indexes'):
            if projections.popleft() < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.get_indexes()


def save_tree(tree, path):
    """
    The other main function in this file, used to store single rp-trees to disk.
    :param tree: The tree to be saved
    :param datasetname: Name of the data set the tree is built for
    """
    if not os.path.exists(path):
        os.makedirs(path)
    ordinal = 0
    while os.path.isfile(path + '/t' + str(ordinal) + '.idx'):
        ordinal += 1
    filename = path + '/t' + str(ordinal) + '.idx'
    with open(filename, 'w') as f:
        cPickle.dump(tree, f)


def load_trees(path, n_trees):
    """
    The other main function in this file. Loads trees from disk.
    :param path: The path where the trees are loaded
    :param n_trees: The number of trees loaded
    :return: A list containing the trees. Empty if no such directory.
    """
    trees = []
    if os.path.exists(path):
        files = os.listdir(path)
        for i in range(min(n_trees, len(files))):
            with open(path+'/'+files[i], 'r') as f:
                trees.append(cPickle.load(f))
    return trees


class FullBinaryTreeLevelTracker(object):
    """
    This class keeps track on the height of an almost full binary tree while adding objects. Useful in construction.
    """
    def __init__(self):
        self.level = 0
        self.level_capacity = 1
        self.level_occupancy = 0

    def object_added(self):
        self.level_occupancy += 1
        if self.level_occupancy == self.level_capacity:
            self.level_occupancy = 0
            self.level_capacity *= 2
            self.level += 1

    def get_level(self):
        return self.level


class InnerNode(object):
    """
    This class defines the structure of the inner rp-tree nodes. The only information stored here are pointers to the
    child nodes and the split value used in this node to divide the data objects to the children.

    """
    def __init__(self):
        self.left = None
        self.right = None
        self.split_threshold = None

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def set_split(self, split):
        self.split_threshold = split


class LeafNode(object):
    """
    This class describes a leaf-node of the rp-tree. The only values stored are the indexes of the data objects that
    belong to the leaf. All information on tree structure is stored in inner nodes.
    """
    def __init__(self, indexes):
        self.indexes = indexes

    def get_indexes(self):
        return self.indexes

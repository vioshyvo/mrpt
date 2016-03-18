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
    A Random Projection Tree is a spatial index data structure used in specific data mining and indexing tasks. Our
    implementation has been built with approximate nearest neighbor search (ANN) problems in mind. A tree is built by
    dividing the data space by random hyperplanes into small cells. In ANN-problems we usually achieve substantial
    improvements by quickly choosing just a subset of the data located in a single cell where the actual brute-force NN
    search is then performed, instead of using the whole data set.
    """

    def __init__(self, data, n0, degree=2):
        """
        Sets the main attributes and calls the routine which actually creates the nodes and builds the tree.
        :param data: The data for which the index is built
        :param n0: The maximum leaf size of the tree
        :param degree: The maximum number of children that each internal node of the tree has
        """
        self.seed = np.random.randint(0, int(1e9))
        self.degree = degree
        self.tree_height = math.ceil(math.log(len(data)/float(n0), degree))
        self.root = _Node()
        self._build_tree(data, n0)

    def _build_tree(self, data, n0):
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
        queue = deque([(self.root, np.arange(len(data)))])
        tracker = _FullTreeLevelTracker(self.degree)
        while len(queue) > 0:
            # Pop next node to be processed
            node, indexes = queue.popleft()

            # Divide the indexes into equal sized chunks (one for each child) and compute the split boundaries
            indexes_divided, node.splits = self._chunkify(indexes, all_projections[indexes, tracker.level], n0)

            # Set references to children, add child nodes to queue if further splits are required (node size > n0)
            for node_indexes in indexes_divided:
                if len(node_indexes) > n0:
                    child = _Node()
                    node.children.append(child)
                    queue.append((child, node_indexes))
                else:
                    node.children.append(node_indexes)

            tracker.object_added()  # Corresponds to adding _node_, not its children, thus called only once

    def _chunkify(self, indexes, projections, n0):
        """
        Divides the list 'indexes' into 'self.degree' equal-sized chunks. If the split is not even, the extra elements
        are added to the list entries on the left. The function aims at building as full leaf nodes as possible, so in
        case each child is not of size at least n0, the method may result in less than 'self.degree' chunks. Think of
        the case n0=10, self.degree=5, len(indexes)=12. Instead of splitting into 5 chunks of sizes 2-3, the function
        returns two chunks with sizes 6 and 5. Thus the leaf sizes are generally closer to n0 and the running time and
        performance stay more predictable.
        :param indexes: The list of indexes to be chunkified
        :param projections: The projections corresponding to the indexes
        :param n0: The maximum leaf size
        :return: The chunks as a list of lists, the projection values that separate these chunks
        """
        indexes = indexes[np.argsort(projections)]
        projections = np.sort(projections)
        n = len(indexes)

        n_chunks = min(int(math.ceil(n/float(n0))), self.degree)
        min_chunk_size = int(math.floor(n/n_chunks))
        chunk_sizes = np.repeat([min_chunk_size], n_chunks)
        chunk_sizes[range(n - min_chunk_size*n_chunks)] += 1
        chunk_bounds = np.cumsum([0] + chunk_sizes)

        return ([indexes[chunk_bounds[i]:chunk_bounds[i+1]] for i in range(n_chunks)],
                [(projections[i-1] + projections[i])/2 for i in chunk_bounds[1:-1]])

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
        while hasattr(node, 'splits'):
            projection = projections.popleft()
            child_index = len(node.splits)
            for i in range(len(node.splits)):
                if projection < node.splits[i]:
                    child_index = i
                    break
            node = node.children[child_index]
        return node


class _FullTreeLevelTracker(object):
    """
    To be documented ...
    """
    def __init__(self, degree=2):
        self.level = 0
        self.level_capacity = 1
        self.level_occupancy = 0
        self.degree = degree

    def object_added(self):
        """
        Must be called every time a node is added to the tree, updates the attributes accordingly.
        """
        self.level_occupancy += 1
        if self.level_occupancy == self.level_capacity:
            self.level_occupancy = 0
            self.level_capacity *= self.degree
            self.level += 1


class _Node(object):
    """
    This class defines the structure of the inner rp-tree nodes. The only information stored here are pointers to the
    child nodes and the split value used in this node to divide the data objects to the children.

    """
    def __init__(self):
        """
        A node is initially just a place holder so no values ...
        """
        self.children = []
        self.splits = []

# encoding: utf-8
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import numpy as np
from scipy.spatial.distance import cdist
from Queue import PriorityQueue, Empty
from collections import deque
from math import log, ceil, floor

# The following imports used only for saving/loading trees
import os
import cPickle
from hashlib import sha1


class MRPTIndex(object):
    """
    The MRPT index is a data structure that allows us to answer approximate nearest neighbor queries quickly. The index
    is basically just a collection of random projection trees. This class implements the basic MRPT index as well as a
    number of small tricks to improve performance.
    """
    def __init__(self, data, n0, n_trees, degree=2, use_saved=False):
        """
        The initializer builds the MRPT index. It has to be called before ANN-queries can be made.
        :param data: The data fro which the index is built. At the moment this has to be done at once, no objects can be
        added later.
        :param n0: The maximum leaf size used in all of the trees of the index.
        :param n_trees: The number of trees used in the index.
        :param degree: Defines the degree of inner random projection tree nodes (the number of children). Use the
        default value unless you really know what you are doing.
        :param use_saved: Saves and loads trees for reuse instead of building new ones every time an index is created.
        :return: The index object.
        """
        self.data = data
        if use_saved:
            self.trees = []
            # The directory name is a hash of the data set contents to avoid using trees built for wrong data
            save_path = 'saved_trees/'+sha1(data.view(np.uint8)).hexdigest()[:8]+'/'+str(n0)+'/'+str(degree)
            self.trees = MRPTIndex.load_trees(save_path, n_trees)
            for t in range(len(self.trees), n_trees):
                t = RPTree(data, n0, degree=degree)
                self.trees.append(t)
                MRPTIndex.save_trees([t], save_path)
        else:
            self.trees = [RPTree(data, n0) for t in range(n_trees)]

    def ann(self, obj, k, extra_branches=0, n_elected=0):
        """
        The mrpt approximate nearest neighbor query. By default the function implements a basic query - the query object
        is routed to one leaf in each tree, and the nearest neighbors are searched in the union of these leaves. The
        extra parameters allow some tricks to improve performance. The parameter extra_brances allows the query to
        explore extra branches where the projection value was close to the split. The n_elected parameter causes the
        query to use the voting trick - not all objects in the union of leaves are searched, but the appearance counts
        as a vote and only the most voted objects get 'elected' to be searched. Thus an object must be returned by
        several trees to be included in the linear search.
        :param obj: The object whose neighbors are being searched
        :param k: The number of neighbors being searched
        :param extra_branches: The number of extra branches allowed in the index
        :param n_elected: The number of elected objects in the voting trick
        :return: The approximate neighbors. In extreme situations not strictly k, but slightly less (eg. 1 tree case)
        """
        priority_queue = PriorityQueue()
        all_projections = []
        votes = np.zeros(len(self.data))

        # First traverse each tree from root to leaf
        for tree_id in range(len(self.trees)):
            indexes, gaps, projections = self.trees[tree_id].full_tree_traversal(obj)
            votes[indexes] += 1
            all_projections.append(projections)
            for (gap_width, node, level) in gaps:
                priority_queue.put((gap_width, node, level, tree_id))

        # Optional branching trick: traverse down from #extra_branches nodes with the smallest d(projection, split)
        for i in range(extra_branches):
            try:
                gap_width, node, level, tree = priority_queue.get(block=False)
                indexes, gaps = RPTree.partial_tree_traversal(node, all_projections[tree][level:], level)
                votes[indexes] += 1
                for gap in gaps:
                    priority_queue.put((gap[0], gap[1], gap[2], tree))
            except Empty:
                print 'More branches than leaves. Will skip the extras.'

        # Decide which nodes to include in the brute force search
        if n_elected > 0:   # Optional voting trick
            elected = np.argsort(votes)[::-1][:n_elected]
        else:  # Basic mrpt
            elected = np.nonzero(votes)[0]

        # Find the nearest neighbors in the subset of objects
        return elected[np.argsort(cdist([obj], self.data[elected])[0])[:k]]

    @staticmethod
    def save_trees(trees, path):
        """
        A function for saving the trees to disk.
        :param trees: The trees to be saved
        :param path: The path where the trees are saved
        """
        if not os.path.exists(path):
            os.makedirs(path)
        ordinal = 0
        for tree in trees:
            while os.path.isfile(path + '/t' + str(ordinal) + '.idx'):
                ordinal += 1
            filename = path + '/t' + str(ordinal) + '.idx'
            with open(filename, 'w') as f:
                cPickle.dump(tree, f)

    @staticmethod
    def load_trees(path, n_trees):
        """
        A function for reading saved trees from the disk.
        :param path: The path from where the trees are loaded
        :param n_trees: The number of trees the user wants loaded
        :return: A list containing the trees. Empty if no such directory exists. If not enough trees exist the function
        will return as many as it finds.
        """
        trees = []
        if os.path.exists(path):
            files = os.listdir(path)
            for i in range(min(n_trees, len(files))):
                with open(path+'/'+files[i], 'r') as f:
                    trees.append(cPickle.load(f))
        return trees


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
        Sets the main attributes and calls the build_tree routine which actually creates the nodes and builds the tree.
        :param data: The data for which the index is built
        :param n0: The maximum leaf size of the tree
        :param degree: The maximum number of children that each internal node of the tree has
        """
        self.seed = np.random.randint(0, int(1e9))
        self.degree = degree
        self.tree_height = int(ceil(log(len(data)/float(n0), degree)))
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

        n_chunks = min(int(ceil(n/float(n0))), self.degree)
        min_chunk_size = int(floor(n/n_chunks))
        chunk_sizes = np.repeat([min_chunk_size], n_chunks)
        chunk_sizes[range(n - min_chunk_size*n_chunks)] += 1
        chunk_bounds = np.cumsum(np.concatenate(([0], chunk_sizes)))

        return ([indexes[chunk_bounds[i]:chunk_bounds[i+1]] for i in range(n_chunks)],
                [(projections[i-1] + projections[i])/2 for i in chunk_bounds[1:-1]])

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
        indexes, gaps = self.partial_tree_traversal(self.root, projections, 0)
        return indexes, gaps, projections

    @staticmethod
    def partial_tree_traversal(node, projections, tree_level):
        """
        Moves down to a leaf starting from the specified node.
        """
        gaps = []
        for projection in projections:
            # Leaves as
            if not hasattr(node, 'splits'):
                break

            # Find the child where the query object belongs.
            child_index = len(node.splits)
            for i in range(len(node.splits)):
                if projection < node.splits[i]:
                    child_index = i
                    break

            # Store the distances to splits for the priority queue trick.
            for i in range(len(node.splits)):
                gap = abs(projection - node.splits[i])
                if i < child_index:
                    gaps.append((gap, node.children[i], tree_level + 1))
                elif i >= child_index:
                    gaps.append((gap, node.children[i+1], tree_level + 1))

            # Move down the tree for next iteration round
            node = node.children[child_index]
            tree_level += 1

        return node, gaps


class _FullTreeLevelTracker(object):
    """
    A _FullTreeLevelTracker object keeps track on the height of a tree with a fixed degree in internal nodes. Every time
    an object is added to the tree the object_added -method need to be called. The height can be read from the 'level'
    attribute.
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
    The class describes the structure of a single internal node. Only the split values used at this node and the links
    to the child nodes need to be stored. (The random vector can be generated without explicitly storing it.). Notice
    also that leaf nodes are not _Node-objects but just simple lists of data object indices.
    """
    def __init__(self):
        self.children = []
        self.splits = []

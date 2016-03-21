# encoding: utf-8
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from rptree import *
import scipy.spatial.distance as ssd
import hashlib as hl
import os
import cPickle
from Queue import PriorityQueue


class MRPTIndex(object):
    """
    The MRPT index is basically just a collection of RP trees. Query results are formed by combining the results of
    single trees. The constructor builds a user-specified number of trees with a user-specified maximum leaf-size. The
    current version can use pre-built saved trees to speed up -- if there are trees built for the same data set with the
    same leaf size those will be used. If the use_saved option is allowed but the index needs more trees than there
    exists in the pre-built collection, as many new trees will be built as needed. The new trees are also added to the
    collection.
    """
    def __init__(self, data, n0=10, n_trees=32, degree=2, use_saved=False):
        self.data = data
        if use_saved:
            self.trees = []
            save_path = 'saved_trees/'+hl.sha1(data.view(np.uint8)).hexdigest()[:8]+'/'+str(n0)+'/'+str(degree)
            self.trees = MRPTIndex.load_trees(save_path, n_trees)
            for t in range(len(self.trees), n_trees):
                t = RPTree(data, n0, degree=degree)
                self.trees.append(t)
                MRPTIndex.save_trees([t], save_path)
        else:
            self.trees = [RPTree(data, n0) for t in range(n_trees)]

    def ann(self, obj, k=10, p=0):
        """
        Ann with backtrack. Will replace ann?
        :param obj:
        :param k:
        :param p:
        :return:
        """
        p_queue = PriorityQueue()
        all_projections = []
        neighborhood = set()

        for i in range(len(self.trees)):
            ((indexes, gaps), projections) = self.trees[i].find_leaf(obj)
            neighborhood = neighborhood.union(indexes)
            all_projections.append(projections)
            for gap in gaps:
                p_queue.put((gap[0], gap[1], gap[2], i))

        for i in range(p):
            gap, node, level, tree = p_queue.get()
            indexes, gaps = RPTree.move_down_from_node(node, all_projections[tree][level:], level)
            neighborhood = neighborhood.union(indexes)
            for gap in gaps:
                p_queue.put((gap[0], gap[1], gap[2], tree))

        neighborhood = list(neighborhood)
        return [neighborhood[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in neighborhood])[0])[:k]]

    def vann(self, obj, k=10, n_elected=200, p=0):
        """
        Ann with backtrack. Will replace ann?
        :param obj:
        :param k:
        :param p:
        :return:
        """
        p_queue = PriorityQueue()
        all_projections = []
        votes = np.zeros(len(self.data))

        for i in range(len(self.trees)):
            ((indexes, gaps), projections) = self.trees[i].find_leaf(obj)
            for vote in indexes:
                votes[vote] += 1
            all_projections.append(projections)
            for gap in gaps:
                p_queue.put((gap[0], gap[1], gap[2], i))

        for i in range(p):
            gap, node, level, tree = p_queue.get()
            indexes, gaps = RPTree.move_down_from_node(node, all_projections[tree][level:], level)
            for vote in indexes:
                votes[vote] += 1
            for gap in gaps:
                p_queue.put((gap[0], gap[1], gap[2], tree))

        elected = np.argsort(votes)[len(votes)-1:len(votes)-1-n_elected:-1]
        return [elected[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in elected])[0])[:k]]

    def ann_old(self, obj, k=10):
        """
        The classic-style MRPT query which is performed in each of the trees, and the results are combined to find the
        best k approximate neighbors.
        :param obj: The vector whose neighbors are being searched for
        :param k: The number of neighbors
        :return: The indices of the approximate neighbors in the data set
        """
        neighborhood = set()
        for tree in self.trees:
            neighborhood = neighborhood.union(tree.find_leaf(obj))
        neighborhood = list(neighborhood)
        return [neighborhood[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in neighborhood])[0])[:k]]

    def vann_old(self, obj, k=10, n_elected=500):
        """
        The voting-enhanced MRPT query. In queries each potential approximate neighbor suggested by a tree counts as
        a vote. Only the objects with the highest number votes are actually compared at the end of the search.
        Has the potential to provide big improvements in query time.
        :param obj: The vector whose neighbors are being searched for
        :param k: The number of neighbors
        :param n_elected: The number of data objects whose distances to the query are really computed
        :return: The indices of the approximate neighbors in the data set
        """
        votes = np.zeros(len(self.data))
        for tree in self.trees:
            ((indexes, gaps), projections) = tree.find_leaf(obj)
            for vote in indexes:
                votes[vote] += 1
        elected = np.argsort(votes)[len(votes)-1:len(votes)-1-n_elected:-1]
        return [elected[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in elected])[0])[:k]]

    @staticmethod
    def save_trees(trees, path):
        """
        The other main function in this file, used to store single rp-trees to disk.
        :param trees: The trees to be saved
        :param datasetname: Name of the data set the trees are built for
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

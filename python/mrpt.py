# encoding: utf-8
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from rptree import *
import scipy.spatial.distance as ssd
from Queue import PriorityQueue

# The following imports used only for saving/loading trees
import os
import cPickle
import hashlib as hl


class MRPTIndex(object):
    """
    The MRPT index is basically just a collection of RP trees. Query results are formed by combining the results of
    single trees. The constructor builds a user-specified number of trees with a user-specified maximum leaf-size. The
    current version can use pre-built saved trees to speed up -- if there are trees built for the same data set with the
    same leaf size those will be used. If the use_saved option is allowed but the index needs more trees than there
    exists in the pre-built collection, as many new trees will be built as needed. The new trees are also added to the
    collection.
    """
    def __init__(self, data, n0, n_trees, degree=2, use_saved=False):
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

    def ann(self, obj, k, extra_branches=0, n_elected=None):
        """
        The mrpt approximate nearest neighbor query. By default the function implements a basic query - the query object
        is routed to one leaf in each tree, and the nearest neighbors are searched in the union of these leaves. The
        extra parameters allow some performance-improving tricks. The parameter extra_brances allows the query to
        explore extra branches where the projection value was close to the split. The n_elected parameter causes the
        query to use the voting trick - not all objects in the union of leaves are searched, but only those that were
        returned by several trees.
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
        for i in range(len(self.trees)):
            ((indexes, gaps), projections) = self.trees[i].full_tree_traversal(obj)
            votes[indexes] += 1
            all_projections.append(projections)
            for gap in gaps:
                priority_queue.put((gap[0], gap[1], gap[2], i))  # gap_size, link_to_node, level_in_tree, tree_id

        # Optional branching trick: traverse down from #extra_branches nodes with the smallest d(projection, split)
        for i in range(extra_branches):
            gap, node, level, tree = priority_queue.get()
            indexes, gaps = RPTree.partial_tree_traversal(node, all_projections[tree][level:], level)
            votes[indexes] += 1
            for gap in gaps:
                priority_queue.put((gap[0], gap[1], gap[2], tree))

        # Decide which nodes to include in the brute force search
        if n_elected is not None:   # Optional voting trick
            elected = np.argsort(votes)[len(votes)-1:len(votes)-1-n_elected:-1]
        else:  # Basic mrpt
            elected = np.nonzero(votes)[0]

        # Find the nearest neighbors in the subset of objects
        return [elected[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in elected])[0])[:k]]

    @staticmethod
    def save_trees(trees, path):
        """
        The other main function in this file, used to store single rp-trees to disk.
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
        The other main function in this file. Loads trees from disk.
        :param path: The path from where the trees are loaded
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

# encoding: utf-8
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from rptree import *
import scipy.spatial.distance as ssd
import Queue

# The following imports used only for saving/loading trees
import os
import cPickle
import hashlib as hl


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
            save_path = 'saved_trees/'+hl.sha1(data.view(np.uint8)).hexdigest()[:8]+'/'+str(n0)+'/'+str(degree)
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
        priority_queue = Queue.PriorityQueue()
        all_projections = []
        votes = np.zeros(len(self.data))

        # First traverse each tree from root to leaf
        for tree_id in range(len(self.trees)):
            indexes, gaps, projections = self.trees[tree_id].full_tree_traversal(obj)
            votes[indexes] += 1
            all_projections.append(projections)
            for (gap_width, node, level) in gaps:
                priority_queue.put((gap_width, node, level, tree_id))  # gap_size, link_to_node, level_in_tree, tree_id

        # Optional branching trick: traverse down from #extra_branches nodes with the smallest d(projection, split)
        for i in range(extra_branches):
            try:
                gap_width, node, level, tree = priority_queue.get(block=False)
                indexes, gaps = RPTree.partial_tree_traversal(node, all_projections[tree][level:], level)
                votes[indexes] += 1
                for gap_width in gaps:
                    priority_queue.put((gap_width[0], gap_width[1], gap_width[2], tree))
            except Queue.Empty:
                print 'More branches than leaves. Will skip the extras.'

        # Decide which nodes to include in the brute force search
        if n_elected > 0:   # Optional voting trick
            elected = np.argsort(votes)[len(votes)-1:len(votes)-1-n_elected:-1]
        else:  # Basic mrpt
            elected = np.nonzero(votes)[0]

        # Find the nearest neighbors in the subset of objects
        return [elected[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in elected])[0])[:k]]

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

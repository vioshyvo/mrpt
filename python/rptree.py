# -*- coding: utf-8 -*-
#
#  Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import numpy as np
import scipy.spatial.distance as ssd


class rptree(object):

    # Values that control random vector generation
    a = 283
    b = 293

    # This method recursively builds the tree structure for the given data
    def __init__(self, data, indices, n0, split_function, seed=None, root=1):

        # If the node is bigger than the allowed maximum leaf size
        if len(indices) > n0:

            # Set random seed and store only if self is root
            if root == 1:
                seed = np.random.randint(0, 1e9)
                self.seed = seed
            np.random.seed(seed)

            # Generate the random vector and compute projections
            randomvector = np.random.normal(size=(1,len(data[0])))
            projections = np.dot(randomvector, np.array([data[index] for index in indices]).T)[0]

            # Decide the split point
            self.splitPoint = split_function(projections)

            # Find out the number of objects that go into each subtree
            sizeofleft = 0
            for i in range(len(projections)):
                if projections[i]<self.splitPoint:
                    sizeofleft += 1

            # Build two lists containing the division of data indices into two subsets, one for each subtree
            leftIndices = np.array([0]*sizeofleft)
            rightIndices = np.array([0]*(len(projections)-sizeofleft))
            l=0
            r=0
            for i in range(len(projections)):
                if projections[i] < self.splitPoint:
                    leftIndices[l] = indices[i]
                    l += 1
                else:
                    rightIndices[r]=indices[i]
                    r += 1

            # Recursively build the subtrees
            self.left = rptree(data, leftIndices, n0, split_function, seed+self.a, 0)
            self.right = rptree(data, rightIndices, n0, split_function, seed+self.b, 0)

        # This node becomes a leaf
        else:
            self.members = indices

    # This method places the query object in a leaf and returns the leaf members
    def query(self, queryobject):

        # Moving down the tree to a leaf
        tree = self
        seed = tree.seed

        while not hasattr(tree, 'members'):

            # restore rng settings, generate vector, compute projection
            np.random.seed(seed)
            randomvector = np.random.normal(size=(1, len(queryobject)))
            projection = np.dot(randomvector, queryobject)

            # Decide into which branch the query object belongs
            if projection < tree.splitPoint:
                tree = tree.left
                seed += self.a
            else:
                tree = tree.right
                seed += self.b

        # Return the indices of the objects in the leaf
        return tree.members


def approximate_knn(trees, queryObject, data, k):
    searchset = np.array([])
    for tree in trees:
        searchset = np.concatenate((searchset,tree.query(queryObject)), axis=1)
    searchset = list(set(searchset))
    print np.array([data[index] for index in searchset])
    indices = np.argsort(ssd.cdist(np.array([queryObject]), np.array([data[index] for index in searchset])))[0,:k]
    print indices
    return [searchset[index] for index in indices]

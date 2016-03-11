# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from mrpt import *


class VotingMRPTIndex(MRPTIndex):

    def __init__(self, data, n0=1000, n_trees=32, tuning=350):
        super(VotingMRPTIndex, self).__init__(data, n0, n_trees)
        self.tuning = tuning

    def ann(self, obj, k):
        neighborhood = []
        for tree in self.trees:
            neighborhood = np.append(neighborhood, tree.find_leaf(obj))
        votes = np.zeros(len(self.data))
        for neighbor in neighborhood:
            votes[neighbor] += 1

        elected = np.argsort(votes)[::-1]
        elected = elected[:self.tuning]

        return [elected[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in elected])[0])[:k]]
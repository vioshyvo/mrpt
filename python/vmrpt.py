# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from mrpt import *


class VMRPTIndex(MRPTIndex):

    def __init__(self, data, n0=1000, n_trees=32, tuning=350):
        super(VMRPTIndex, self).__init__(data, n0, n_trees)
        self.tuning = tuning

    def ann(self, obj, k):
        neighborhood = []
        votes = np.zeros(len(self.data))
        for tree in self.trees:
            for vote in tree.find_leaf(obj):
                votes[vote] += 1

        elected = np.argsort(votes)[len(votes)-1:len(votes)-1-self.tuning:-1]

        return [elected[i] for i in np.argsort(ssd.cdist([obj], [self.data[i] for i in elected])[0])[:k]]
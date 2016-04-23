# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import mrptlib


class MRPTIndex(object):
    """
    Wraps the extension module written in C++ and ensures that the arguments are given as lists, not ndarrays.
    """
    def __init__(self, X, n0, n_trees):
        self.index = mrptlib.MrptIndex(X.tolist(), n0, n_trees)

    def ann(self, q, k, elect):
        return self.index.ann(q.tolist(), k, elect)

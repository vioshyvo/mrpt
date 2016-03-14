# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import cPickle
import os


# Constructs a new unique filename
def new_filename(filename):
    ordinal = 0
    while os.path.isfile(filename+str(ordinal) + '.idx'):
        ordinal += 1
    return filename+str(ordinal) + '.idx'


# Checks that the directories needed exist and creates them if not
def create_dirs(datasetname, n0):
    if not os.path.exists('saved_trees/' + datasetname+'/' + str(n0)):
        os.makedirs('saved_trees' + '/' + datasetname + '/' + str(n0))


# Saves a single tree
def save(tree, datasetname):
    create_dirs(datasetname, tree.get_n0())
    filename = new_filename('saved_trees/' + datasetname + '/' + str(tree.get_n0()) + '/' + 't')
    with open(filename, 'w') as f:
        cPickle.dump(tree, f)


# Loads number rp-trees with the desired n0.
def load(n0, number):
    raise NotImplementedError('Method not yet implemented!')

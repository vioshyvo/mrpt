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
    if not os.path.exists('saved_trees/' + datasetname + '/' + n0):
        os.makedirs('saved_trees' + '/' + datasetname + '/' + n0)


# Saves a single tree
def save(tree, datasetname, n0):
    """
    The main function in this file, used to store single rp-trees to disk.
    :param tree: The tree to be saved
    :param datasetname: Name of the data set the tree is built for
    """
    create_dirs(datasetname, str(n0))
    filename = new_filename('saved_trees/' + datasetname + '/' + str(n0) + '/' + 't')
    with open(filename, 'w') as f:
        cPickle.dump(tree, f)


# Loads number rp-trees with the desired n0. NOTE! THE DIRECTORY MUST NOT CONTAIN ANY OTHER FILES!!!
def load(datasetname, n0, n_trees):
    dirname = 'saved_trees/'+datasetname+'/'+str(n0)+'/'
    filenames = os.listdir(dirname)
    trees = []
    ordinal = 0
    for i in range(min(n_trees, len(filenames))):
        with open(dirname+filenames[ordinal], 'r') as f:
            trees.append(cPickle.load(f))
    return trees

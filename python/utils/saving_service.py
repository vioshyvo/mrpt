# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import cPickle
import os


# Saves a single tree
def save(tree, path):
    """
    The other main function in this file, used to store single rp-trees to disk.
    :param tree: The tree to be saved
    :param datasetname: Name of the data set the tree is built for
    """
    create_dirs(path)
    filename = new_filename(path + '/' + 't')
    with open(filename, 'w') as f:
        cPickle.dump(tree, f)


def new_filename(filename):
    """
    Used to construct a unique filename. Finds the smallest ordinal not already used in a tree name.
    :param filename: The first part of the filename
    :return:
    """
    ordinal = 0
    while os.path.isfile(filename+str(ordinal) + '.idx'):
        ordinal += 1
    return filename+str(ordinal) + '.idx'


def create_dirs(path):
    """
    Checks and creates directories for trees
    :param path: Path where the trees are saved
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load(path, n_trees):
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

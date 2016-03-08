# Python implementation of the MRPT algorithm

Dependencies
============

Numpy, Scipy

Get started
===========

To create a MRPT index instance from a matrix of row vectors, simply call: 
	
	from mrpt import *
	index = MRPTIndex(data)

And that's it! Sit back and enjoy lightning-fast approximate nearest neighbor queries by calling:

	neighbors = index.ann(query_object, k)

The query returns the indexes of the approximate nearest neighbors in the original data set.

Advanced
========

The algorithm can also take optional inputs, the number of random projection trees used in the index `n_trees` and the maximum leaf size 
of the trees `n0`. Simply put, increasing either value improves the accuracy of the approximation, but makes the algorithm (both index 
building and queries) run slower.

	index = MRPTIndex(data, n0=32, n_trees=64)

Extras
======
The file `mnist_utils.py` contains python functions to read and visualize the popular mnist data set.

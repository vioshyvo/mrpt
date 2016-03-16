# Python implementation of the MRPT algorithm

The python version of the algorithm is intended mostly for research purposes. Our C++ implementation naturally runs way faster, and even with python you are likely better off using the C++ version with python wrappers. However, we find prototyping in python to be way more convenient and this version is usually the most up-to-date. Also note that even this version can provide reasonable approximations way faster than an exact brute-force approach using numpy.

##Dependencies

Numpy, Scipy

##Get started

To create a MRPT index instance from a matrix of row vectors, simply call: 
	
	from mrpt import *
	index = MRPTIndex(data)

And that's it! Sit back and enjoy lightning-fast approximate nearest neighbor queries by calling:

	neighbors = index.ann(query_object, k)

The query returns the indexes of the approximate nearest neighbors in the original data set.

##Advanced
The algorithm has some optional parameters for more advanced use. Read the following for details.

###Number of trees and leaf size

The algorithm has optional parameters `n0` and `n_trees`, which stand for the maximum tree leaf size and the number of trees, respectively. Using a bigger leaf size makes buiding the index slightly faster, but queries get slower. Increasing `n_trees` will make both index construction and queries slower. However, increasing either of the parameters increses the accuracy of the NN-approximation. Finding the right balance between `n0` and `n_trees` is an active research topic.

###Speed up index construction with pre-built trees

The index construction now supports saving and loading trees. When calling the `MRPTIndex` constructor, pass an optional parameter `use_saved=True` to enable this. The constructor will use previously built trees saved in a database to reduce construction time. Any new trees will also be added to the database.

###Speed up queries by employing a voting trick

When employing the voting trick, the index is built exactly in the same manner as in typical MRPT. To understand the method we need to go a bit deeper to the normal query first. When answering a query, each of the `n_trees` trees in the index returns all the objects in a single leaf, resulting in at most `n_trees*n0` objects. We compute the nearest neighbors wthin these objects in a brute-force manner. 

The voting-scheme is based on the intuitive idea of the actually nearest neighbors likely being returned by several trees. In a voting approximate nearest neighbor query (function `vann`) we take the objects returned by the trees as votes, and only compute the distances to the objects with the most votes. The number of these objects is determined by an extra parameter `n_elected` of the `vann` function. Notice that this trick allows us to use a lot bigger values of `n0`.

	voting_index = MRPTIndex(data, n0=1000, n_trees=64, use_saved=True)
    neighbors = index.vann(query_object, k=10, n_elected=100)

##Extras
The file `mnist_utils.py` contains python functions to read and visualize the popular mnist data set.

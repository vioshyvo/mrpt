# MRPT in python 2.7

This directory contains the Python implementation of our method. Although this version is usually the most up-to-date, notice that the C++ version runs way faster in real-life applications. Even this version can still provide excellent approximations way faster than an exact brute force approach using numpy.

##Dependencies

Numpy, Scipy

##Get started

To use MRPT for nearest neighbor queries, we first need to construct the index. This can be achieved by: 
	
	from mrpt import *
	index = MRPTIndex(data, n0=<value>, n_trees=<value>)

`data` has to be given in a NxDIM array, `n0` and `n_trees`, as explained below, are parameters that affect both accuracy and running time of queries. After the index has been constructed, approximate nearest neighbors can be found with the `ann`-function:

	neighbors = index.ann(query_object, <k>),

where `query_object` is a single row vector whose neighbors are being searched and `k` is the number of neighbors. The function returns the indexes (row numbers) of the approximate nearest neighbors in the original data set.

###Number of trees and leaf size

The algorithm has parameters `n0` and `n_trees`, which stand for the maximum tree leaf size and the number of trees, respectively. Using a bigger leaf size makes buiding the index slightly faster, but queries get slower. Increasing `n_trees` will make both index construction and queries slower. However, increasing either of the parameters increses the accuracy of the NN-approximation. Finding the right balance between `n0` and `n_trees` is an active research topic.

##Advanced
The algorithm has optional parameters for more advanced use. Read the following for details.

### Experiment with node degrees
The code allows the user to use multiway (non-binary) splits, however, THIS IS STRONGLY DISCOURAGED. In our initial tests even splitting to 3 or 4 causes accuracy to decrease alarmingly and the corresponding decrease in query time is not sufficient to cover this loss. The node degree can be passed by the optional constructor parameter `degree`.

###Speed up index construction with pre-built trees

The index construction supports saving and loading trees. When calling the `MRPTIndex` constructor, pass an optional parameter `use_saved=True` to enable this. The constructor will use previously built trees saved in a database to reduce construction time. Any new trees will also be added to the database. Notice that the trees can take quite a lot of space.

###Speed up queries by employing a voting trick

For the next two tricks, the index is built exactly in the same manner as in typical MRPT. The extra parameters are passed to the `ann`-function instead. 

To understand the voting trick we need to go a bit deeper to the normal query first. When answering a query, each of the `n_trees` trees in the index returns all the objects in a single leaf, resulting in at most `n_trees*n0` objects. We compute the nearest neighbors within these objects in a brute-force manner. 

The voting-scheme is based on the intuitive idea of the actually nearest neighbors likely being returned by several trees. In an approximate nearest neighbor query with voting we take the objects returned by the trees as votes, and only compute the distances to the objects with the most votes. The number of these objects is determined by an extra parameter `n_elected` of the `ann` function. Notice that this trick allows us to use a lot bigger values of `n0`.

### Route queries to extra branches

Why does a query have to be routed to just one leaf per tree? Actually it does not! Using multiple branches in the existing trees is actually a good way to avoid the need to build new ones.

This trick maintains a priority queue of the difference between the projection value of the query object and the split value in the node, for all nodes in the path the query traverses from root to leaf, in all trees. Once all trees have been traversed once from root to leaf, we take as many extra branches from the priority queue as we want. The number of these extra branches is declared by the `extra_branches` -parameter of the `ann` function.

## MNIST handwritten digits example
The file `mnist_utils.py` contains python functions to read and visualize the popular [mnist data](http://yann.lecun.com/exdb/mnist/) set. You may need to fix the paths to data sets etc. 

Once you have the data, you can try the following examples for 10-NN search. For reference, finding the neighbors with bruteforce with the following command takes approximately 300-400ms on my system (a 2013 Macbook Air 2*1.3GHz Core i5):

    neighbors = numpy.argsort(scipy.spatial.distance.cdist([query_img], mnist_train_imgs)[0])[:10]

### Basic MRPT

The following parameters for the basic MRPT should get 9 out of 10 neighbors right on average. On my system the query time is about 200ms.
    
    index = MRPTIndex(data, n0=32, n_trees=75)
    neighbors = index.ann(query_obj, 10)

### MRPT with voting and extra branches

We can do better using the additional tricks described above. The following parameters should get roughly 9 out of 10 neighbors right and the queries run in less than 40ms on my system.

    index = MRPTIndex(data, n0=512, n_trees=30)
    neighbors = index.ann(query_obj, 10, extra_branches=140, n_elected=200)

The code in `example.py` executes the above commands and prints some basic stats.


# Questions?
teemu.pitkanen'at'cs.helsinki.fi


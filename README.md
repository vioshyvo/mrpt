# MRPT - fast nearest neighbor search with random projection
![NN search in 3 RP-trees](mrpt-image.jpg)


MRPT is a library for approximate nearest neighbor search written in C++11. According to [our experiments](https://github.com/ejaasaari/mrpt-comparison/) MRPT is currently the *fastest* alternative to reach high recall levels in common benchmark data sets.

In the offline phase of the algorithm MRPT indexes the data with a collection of *random projection trees*. In the online phase the index structure allows us to answer queries in superior time. More technical details can be found in the pre-print version of [our paper](https://arxiv.org/pdf/1509.06957.pdf).

A simple demo in Python (runs in approx 20 seconds): 
~~~~
import numpy as np
from scipy.spatial.distance import cdist
from time import time
import mrpt

# Generate synthetic test data
k = 10; n_queries = 100
data = np.dot(np.random.rand(1e5,5), np.random.rand(5,100)).astype('float32')
queries = np.dot(np.random.rand(n_queries,5), np.random.rand(5,100)).astype('float32')

# Solve exact nearest neighbors with standard methods from scipy and numpy for reference
exact_search_time = time()
exact_neighbors = np.zeros((n_queries, k))
for i in range(n_queries):
    exact_neighbors[i] = np.argsort(cdist([queries[i]], data))[0,:k]
exact_search_time = time() - exact_search_time
    
# Offline phase: Indexing the data.
indexing_time = time()
index = mrpt.MRPTIndex(data, depth=7, n_trees=300)
index.build()
indexing_time = time() - indexing_time
    
# Online phase: Approximate nearest neighbor search using the index.
approximate_search_time = time()
approximate_neighbors = np.zeros((n_queries, k))
for i in range(n_queries):
    approximate_neighbors[i] = index.ann(queries[i], k, votes_required=4)
approximate_search_time = time() - approximate_search_time
        
# Printing some stats.
print 'Indexing time: %1.3f seconds' %indexing_time
print '%d approximate queries time: %1.3f seconds' %(n_queries, approximate_search_time)
print '%d exact queries time: %1.3f seconds' %(n_queries, exact_search_time)
        
correct_neighbors = 0
for i in range(n_queries):
    correct_neighbors += len(np.intersect1d(exact_neighbors[i], approximate_neighbors[i]))
print 'Average recall: %1.2f' %(float(correct_neighbors)/(n_queries*k))
~~~~
            
An example output:
~~~~
Indexing time: 5.993 seconds
100 approximate queries time: 0.230 seconds
100 exact queries time: 11.776 seconds
Average recall: 0.97
~~~~

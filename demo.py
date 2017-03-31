import numpy as np
from scipy.spatial.distance import cdist
from time import time
import mrpt

# Generate synthetic test data
k = 10; n_queries = 100
data = np.dot(np.random.rand(int(1e5),5), np.random.rand(5,100)).astype('float32')
queries = np.dot(np.random.rand(n_queries,5), np.random.rand(5,100)).astype('float32')

# Solve exact nearest neighbors with standard methods from scipy and numpy for reference
exact_search_time = time()
exact_neighbors = np.zeros((n_queries, k))
for i in range(n_queries):
    exact_neighbors[i] = np.argsort(cdist([queries[i]], data))[0,:k]
exact_search_time = time() - exact_search_time

# Offline phase: Indexing the data. This might take some time.
indexing_time = time()
index = mrpt.MRPTIndex(data, depth=5, n_trees=100)
index.build()
indexing_time = time() - indexing_time

# Online phase: Finding nearest neighbors stupendously fast.
approximate_search_time = time()
approximate_neighbors = np.zeros((n_queries, k))
for i in range(n_queries):
    approximate_neighbors[i] = index.ann(queries[i], k, votes_required=4)
approximate_search_time = time() - approximate_search_time

# Print some stats
print ('Indexing time: %1.3f seconds' %indexing_time)
print ('%d approximate queries time: %1.3f seconds' %(n_queries, approximate_search_time))
print ('%d exact queries time: %1.3f seconds' %(n_queries, exact_search_time))

correct_neighbors = 0
for i in range(n_queries):
    correct_neighbors += len(np.intersect1d(exact_neighbors[i], approximate_neighbors[i]))
print ('Average recall: %1.2f.' %(float(correct_neighbors)/(n_queries*k)))

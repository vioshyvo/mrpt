# Python implementation for the MRPT algotihm

To creata a MRPT index instance, simply call 

> index = MRPTIndex(data_to_be_indexed)

A k-nn query can then be made by

> neighbors = index.ann(query_object, k)

The query returns the indices of the approximate nearest neighbors in the original data set.

#ifndef KNN_H
#define	KNN_H

// find k nearest neighbors from data for the query point
// X = data matrix, row = data point, col = dimension
// q = query point as a row matrix
// k = number of neighbors searched for
// indices = indices of the points in the original matrix where search is made
// return : indices of nearest neighbors in data matrix X as a column vector
uvec knnCpp_indices(const fmat& X, const frowvec& q, uword k, uvec indices);

// find k nearest neighbors from data for the query point
// X = transposed data matrix, row = dimension, col = data point
// q = query point as a column matrix
// k = number of neighbors searched for
// indices = indices of the points in the original matrix where search is made
// return : indices of nearest neighbors in data matrix X as a column vector
uvec knnCpp_T_indices(const fmat& X, const fvec& q, uword k, uvec indices);

// find k nearest neighbors from data for the query point
// X = data matrix, row = data point, col = dimension
// q = query point as a row matrix
// k = number of neighbors searched for
// return : indices of nearest neighbors in data matrix X as a column vector
uvec knnCpp(const fmat& X, const frowvec& q, int k);

// find k nearest neighbors from data for the query point
// X = *transposed* data matrix, row = dimension, col = data point
// q = query point as a column matrix
// k = number of neighbors searched for
// return : indices of nearest neighbors (cols of transposed data matrix X) as a column vector
uvec knnCppT(const fmat& X, const fvec& q, int k);

#endif	/* KNN_H */


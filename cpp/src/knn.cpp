#include <armadillo>

using namespace arma;

// find k nearest neighbors from data for the query point
// X = data matrix, row = data point, col = dimension
// q = query point as a row matrix
// k = number of neighbors searched for
// return : indices of nearest neighbors in data matrix X as a column vector

uvec knnCpp_indices(const fmat& X, const frowvec& q, uword k, uvec indices) {
    // std::cout << "q.head(5): " << q.head(5);
    int n_rows = indices.size();
    // std::cout << "n_rows: " << n_rows << std::endl;
    fvec distances = fvec(n_rows);
    for (int i = 0; i < n_rows; i++)
        distances[i] = sum(pow((X.row(indices(i)) - q), 2));
    
    if(k == 1) {
        uvec ret(1);
        distances.min(ret[0]);
        return ret;
    }

    uvec sorted_indices = indices(sort_index(distances));
    // std::cout << "sorted_indices:\n" << sorted_indices; 
    return sorted_indices.size() > k ? sorted_indices.head(k) : sorted_indices;
}


// find k nearest neighbors from data for the query point
// X = data matrix, row = data point, col = dimension
// q = query point as a row matrix
// k = number of neighbors searched for
// return : indices of nearest neighbors in data matrix X as a column vector

uvec knnCpp_T_indices(const fmat& X, const fvec& q, uword k, uvec indices) {
    int n_cols = indices.size();
    fvec distances = fvec(n_cols);
    for (int i = 0; i < n_cols; i++)
        distances[i] = sum(pow((X.col(indices(i)) - q), 2));
    
    if(k == 1) {
        uvec ret(1);
        distances.min(ret[0]);
        return ret;
    }
    
    uvec sorted_indices = indices(sort_index(distances));
    return sorted_indices.size() > k ? sorted_indices.head(k): sorted_indices;
}



// find k nearest neighbors from data for the query point
// X = data matrix, row = data point, col = dimension
// q = query point as a row matrix
// k = number of neighbors searched for
// return : indices of nearest neighbors in data matrix X as a column vector

uvec knnCpp(const fmat& X, const frowvec& q, int k) {
    int n_rows = X.n_rows;
    fvec distances = fvec(n_rows);
    for (int i = 0; i < n_rows; i++)
        distances[i] = sum(pow((X.row(i) - q), 2));
    
    if(k == 1) {
        uvec ret(1);
        distances.min(ret[0]);
        return ret + 1;
    }
    
    uvec sorted_indices = sort_index(distances);
    return sorted_indices.head(k) + 1;
}


// find k nearest neighbors from data for the query point
// X = *transposed* data matrix, row = dimension, col = data point
// q = query point as a column matrix
// k = number of neighbors searched for
// return : indices of nearest neighbors (cols of transposed data matrix X) as a column vector

uvec knnCppT(const fmat& X, const fvec& q, int k) {
    int n_cols = X.n_cols;
    fvec distances = fvec(n_cols);
    for (int i = 0; i < n_cols; i++)
        distances[i] = sum(pow((X.col(i) - q), 2));
    
    if(k == 1) {
        uvec ret(1);
        distances.min(ret[0]);
        return ret;
    }
    
    uvec sorted_indices = sort_index(distances);
    return sorted_indices.subvec(0, k - 1);
}


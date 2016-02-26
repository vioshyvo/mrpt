#include <iostream>
#include "armadillo"
#include <ctime>

using namespace arma;

#include "knn.h"
#include "Mrpt.h"


Mrpt::Mrpt(const fmat& X_, int n_trees_, int n_0_, std::string id_) : X(X_), n_trees(n_trees_), n_0(n_0_), id(id_){
    n_rows = X.n_cols; // X is transposed
    dim = X.n_rows;
    depth = ceil(log2(n_rows / n_0));
    n_pool = n_trees * depth;
    n_array = pow(2, depth + 1);
    trees = fmat();
    leaf_labels = umat();
    random_matrix = fmat();
}

void Mrpt::read_trees() {
    trees.load(id + "_trees.mat");
    random_matrix.load(id + "_random_matrix.mat");
    leaf_labels.load(id + "_leaf_labels.mat");
}


std::vector<double> Mrpt::grow() {
    trees = zeros<fmat>(n_array, n_trees);
    leaf_labels = umat(n_rows, n_trees);
    std::vector<double> times(2);
    uvec indices = linspace<uvec>(0, n_rows - 1, n_rows);

    // generate the random matrix and project the data set onto it
    clock_t begin = clock();
    random_matrix = conv_to<fmat>::from(randn(n_pool, dim));
    projected_data = random_matrix * X;
    // std::cout << "n_pool: " << n_pool << ", n_rows: " << n_rows << ", dim: " << dim << std::endl;
    // std::cout << "projected_data.n_rows: " << projected_data.n_rows << ", projected_data.n_cols: " << projected_data.n_cols << std::endl;
    clock_t end = clock();
    times[0] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);

    // grow the trees
    begin = clock();
    for (int n_tree = 0; n_tree < n_trees; n_tree++) {
        first_idx = n_tree * depth;
        grow_subtree(indices, 0, 0, n_tree); // all rows of data, 0 = level of the tree, 0 = first index in the array that stores the tree, n_tree:th tree

    }
    end = clock();
    times[1] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);

    trees.save(id + "_trees.mat");
    random_matrix.save(id + "_random_matrix.mat");
    leaf_labels.save(id + "_leaf_labels.mat");
    return times;
}


uvec Mrpt::query(const fvec& q, int k) {
    fvec projected_query = random_matrix * q; // query vector q is passed as a reference to a col vector
    std::vector<int> idx_canditates(n_trees * n_0);
    int j = 0;


    for (int n_tree = 0; n_tree < n_trees; n_tree++) {
        const uvec& col_leaf_labels = leaf_labels.unsafe_col(n_tree);
        const fvec& tree = trees.unsafe_col(n_tree);

        double split_point = tree[0];
        int idx_left, idx_right;
        int idx_tree = 0;

        while (split_point) {
            idx_left = 2 * idx_tree + 1;
            idx_right = idx_left + 1;
            idx_tree = projected_query(j++) <= split_point ? idx_left : idx_right;
            split_point = tree[idx_tree];
        }

        uvec idx_one_tree = find(col_leaf_labels == idx_tree);
        idx_canditates.insert(idx_canditates.begin(), idx_one_tree.begin(), idx_one_tree.end());
    }

    auto last = std::unique(idx_canditates.begin(), idx_canditates.end());
    idx_canditates.erase(last, idx_canditates.end());

    return knnCpp_T_indices(X, q, k, conv_to<uvec>::from(idx_canditates));
}


uvec Mrpt::query_canditates(const fvec& q, int k) {
    fvec projected_query = random_matrix * q; // query vector q is passed as a reference to a col vector
    std::vector<int> idx_canditates(n_trees * n_0);
    int j = 0;

    // std::cout << "projected_query.size(): " << projected_query.size() << ", idx_canditates.size(): " << idx_canditates.size() << std::endl;
    for (int n_tree = 0; n_tree < n_trees; n_tree++) {
        // std::cout << "n_tree: " << n_tree << ", n_trees: " << n_trees << ", j: " << j << std::endl;

        const uvec& col_leaf_labels = leaf_labels.unsafe_col(n_tree);
        const fvec& tree = trees.unsafe_col(n_tree);

        // std::cout << "tree[0]: " << tree[0] << std::endl;

        double split_point = tree[0];
        int idx_left, idx_right;
        int idx_tree = 0;

        while (split_point) {
            idx_left = 2 * idx_tree + 1;
            idx_right = idx_left + 1;
            idx_tree = projected_query(j++) <= split_point ? idx_left : idx_right;
            split_point = tree[idx_tree];
            // std::cout << "idx_left: " << idx_left << ", idx_right: " << idx_right << ", split_point: " << split_point << std::endl;
            // bool temp = split_point == 0;
            // std::cout << "split_point == 0: " <<  temp << std::endl;
        }

        uvec idx_one_tree = find(col_leaf_labels == idx_tree);
        idx_canditates.insert(idx_canditates.begin(), idx_one_tree.begin(), idx_one_tree.end());
    }

    auto last = std::unique(idx_canditates.begin(), idx_canditates.end());
    idx_canditates.erase(last, idx_canditates.end());
    return conv_to<uvec>::from(idx_canditates);
}


void Mrpt::matrix_multiplication(const fvec& q) {
    fvec projected_query = random_matrix * q;
}


void Mrpt::grow_subtree(const uvec &indices, int tree_level, int i, uword n_tree) {
    int n = indices.size();
    int idx_left = 2 * i + 1;
    int idx_right = idx_left + 1;

    if (n <= n_0) {
        uvec idx_tree = {n_tree};
        leaf_labels(indices, idx_tree) = zeros<uvec>(n) + i;
        return;
    }

    uvec level = {first_idx + tree_level};
    frowvec projection = projected_data(level, indices);
    uvec ordered = sort_index(projection); // indices??

    int split_point = n % 2 ? n / 2 : n / 2 - 1; // median split
    int idx_split_point = ordered(split_point);
    int idx_split_point2 = ordered(split_point + 1);

    trees(i, n_tree) = n % 2 ? projection(idx_split_point) : (projection(idx_split_point) + projection(idx_split_point2)) / 2;
    uvec left_indices = ordered.subvec(0, split_point);
    uvec right_indices = ordered.subvec(split_point + 1, n - 1);

    grow_subtree(indices.elem(left_indices), tree_level + 1, idx_left, n_tree);
    grow_subtree(indices.elem(right_indices), tree_level + 1, idx_right, n_tree);

}

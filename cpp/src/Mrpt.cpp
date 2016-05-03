#include <iostream>
#include "armadillo"
#include <ctime>
#include <cstdlib>
#include <queue>

using namespace arma;

#include "knn.h"
#include "Mrpt.h"

/**
 * The constructor of the index
 * @param X_ - The data for which the index will be built
 * @param n_trees_ - The number of trees to be used in the index
 * @param n_0_ - The maximum leaf size to be used in the index
 * @param id_ - A name used for filenames when saving
 */
Mrpt::Mrpt(const fmat& X_, int n_trees_, int n_0_, std::string id_) : X(X_), n_trees(n_trees_), n_0(n_0_), id(id_){
    n_samples = X.n_cols; // X is transposed
    dim = X.n_rows;
    depth = ceil(log2(n_samples / n_0));
    n_pool = n_trees * depth;
    n_array = pow(2, depth + 1);
    split_points = fmat();
    random_matrix = fmat();
}

/**
 * A function for reading previously built trees from files.
 * @return -
 */
//void Mrpt::read_trees() {
//    trees.load(id + "_trees.mat");
//    random_matrix.load(id + "_random_matrix.mat");
//    leaf_labels.load(id + "_leaf_labels.mat");
//}


/**
 * The function that actually builds the trees.
 * @return 
 */
std::vector<double> Mrpt::grow() {
    split_points = zeros<fmat>(n_array, n_trees);
    std::vector<double> times(2);
    uvec indices = linspace<uvec>(0, n_samples - 1, n_samples);

    // generate the random matrix and project the data set onto it
    clock_t begin = clock();
    random_matrix = conv_to<fmat>::from(randn(n_pool, dim));
    projected_data = random_matrix * X;
    clock_t end = clock();
    times[0] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);
    
    // Grow the trees
    begin = clock();
    for (int n_tree = 0; n_tree < n_trees; n_tree++) {       
        first_idx = n_tree * depth;
        std::vector<uvec> t = grow_subtree(indices, 0, 0, n_tree); // all rows of data, 0 = level of the tree, 0 = first index in the array that stores the tree, n_tree:th tree
        tree_leaves.push_back(t);
    }
    end = clock();
    times[1] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);
    
    // Save tree info
    //split_points.save(id + "_trees.mat");
    //random_matrix.save(id + "_random_matrix.mat");
    //leaf_labels.save(id + "_leaf_labels.mat"); // Cannot save like this in current tree format
    return times;
}

/**
 * Builds a single random projection tree. The tree is constructed by recursively
 * projecting the data on a random vector and splitting into two by the median.
 * @param indices - The indices left in this branch
 * @param tree_level - The level in tree where the recursion is at
 * @param i - The index within the tree where we are at
 * @param n_tree - The index of the tree within the index
 * @return The leaves as a vector of arma::uvecs
 */
std::vector<uvec> Mrpt::grow_subtree(const uvec &indices, int tree_level, int i, uword n_tree) {
    int n = indices.size();
    int idx_left = 2 * i + 1;
    int idx_right = idx_left + 1;

    if (tree_level == depth) {
        std::vector<uvec> v;
        v.push_back(indices);
        return v;
    }

    uvec level = {first_idx + tree_level};
    frowvec projection = projected_data(level, indices);
    uvec ordered = sort_index(projection);

    int split_point = n % 2 ? n / 2 : n / 2 - 1; // median split
    int idx_split_point = ordered(split_point);
    int idx_split_point2 = ordered(split_point + 1);

    split_points(i, n_tree) = n % 2 ? projection(idx_split_point) : (projection(idx_split_point) + projection(idx_split_point2)) / 2;
    uvec left_indices = ordered.subvec(0, split_point);
    uvec right_indices = ordered.subvec(split_point + 1, n - 1);

    std::vector<uvec> v = grow_subtree(indices.elem(left_indices), tree_level + 1, idx_left, n_tree);
    std::vector<uvec> w = grow_subtree(indices.elem(right_indices), tree_level + 1, idx_right, n_tree);
    
    for (int j = 0; j < w.size(); j++){
        v.push_back(w[j]);
    }
    return v;
}


/**
 * This function finds the k approximate nearest neighbors of the query object 
 * q. The accuracy of the query depends on both the parameters used for index 
 * construction and additional parameters given to this function. This 
 * function implements two tricks to improve performance. The voting trick 
 * interprets each index object in leaves returned by tree traversals as votes,
 * and only performs the final linear search with the 'elect' most voted 
 * objects. The priority queue trick keeps track of nodes where the split value
 * was close to the projection so that we can split the tree traversal to both
 * subtrees if we want.
 * @param q - The query object whose neighbors the function finds.
 * @param k - The number of neighbors the user wants the function to return
 * @param elect - The number of most voted objects elected to linear search
 * @param branches - The number of extra branches explored in priority queue trick
 * @return The indices of the k approximate nearest neighbors in the original
 * data set for which the index was built.
 */
uvec Mrpt::query(const fvec& q, int k, int elect, int branches) {
    
    fvec projected_query = random_matrix * q;
    uvec votes = zeros<uvec>(n_samples);
    std::priority_queue<Gap, std::vector<Gap>, std::greater<Gap>> pq;
    
    /*
     * The following loops over all trees, and routes the query to exactly one 
     * leaf in each.
     */
    int j = 0; // Used to find the correct projection value, increases through all trees
    for (int n_tree = 0; n_tree < n_trees; n_tree++) {
        const fvec& tree = split_points.unsafe_col(n_tree);

        double split_point = tree[0];
        int idx_left, idx_right;
        int idx_tree = 0;

        while (split_point) {
            idx_left = 2 * idx_tree + 1;
            idx_right = idx_left + 1;
            if (projected_query(j) <= split_point) {
                idx_tree = idx_left;
                pq.push(Gap(n_tree, idx_right, j+1, split_point-projected_query(j)));
            } else {
                idx_tree = idx_right;
                pq.push(Gap(n_tree, idx_left, j+1, projected_query(j)-split_point));
            }
            j++;
            split_point = tree[idx_tree];
        }
        const uvec& idx_one_tree = tree_leaves[n_tree][idx_tree - pow(2, depth) + 1];
        for (int i = 0; i < idx_one_tree.size(); i++){
            votes[idx_one_tree[i]]++;
        }
    }
    
    /*
     * The following loop routes the query to extra leaves in the trees handled
     * already once above. The extra branches are popped from the priority queue
     * and handled just as new root-to-leaf queries.
     */
    for (int i = 0; i < branches; i++){
        if (pq.empty()) break;
        Gap gap = pq.top();
        pq.pop();
        
        const fvec& tree = split_points.unsafe_col(gap.tree);
        int idx_tree = gap.node;
        int idx_left, idx_right;
        j = gap.level;
        double split_point = tree[idx_tree];

        while (split_point) {
            idx_left = 2 * idx_tree + 1;
            idx_right = idx_left + 1;
            if (projected_query(j) <= split_point) {
                idx_tree = idx_left;
                pq.push(Gap(gap.tree, idx_right, j+1, split_point-projected_query(j)));
            } else {
                idx_tree = idx_right;
                pq.push(Gap(gap.tree, idx_left, j+1, projected_query(j)-split_point));
            }
            j++;
            split_point = tree[idx_tree];
        }

        const uvec& idx_one_tree = tree_leaves[gap.tree][idx_tree - pow(2, depth) + 1];
        for (int i = 0; i < idx_one_tree.size(); i++){
            votes[idx_one_tree[i]]++;
        } 
    } 
    
    // Compute the actual NNs within the 'elect' objects with most votes
    uvec elected = sort_index(votes, "descend");
    elected.resize(elect);
    return knnCpp_T_indices(X, q, k, elected);
}


/**
 * This function implements the barebones MRPT algorithm and finds the k 
 * approximate nearest neighbors of the query object q using multiple random 
 * projection trees. The accuracy of the query depends on the parameters used 
 * for index construction. Separated from the function above for optimal performance.
 * @param q - The query object whose neighbors the function finds.
 * @param k - The number of neighbors the user wants the function to return
 * @return The indices of the k approximate nearest neighbors in the original
 * data set for which the index was built.
 */
uvec Mrpt::query(const fvec& q, int k) {
    fvec projected_query = random_matrix * q; // query vector q is passed as a reference to a col vector
    std::vector<int> idx_canditates(n_trees * n_0);
    int j = 0;


    for (int n_tree = 0; n_tree < n_trees; n_tree++) {
        const fvec& tree = split_points.unsafe_col(n_tree);

        double split_point = tree[0];
        int idx_left, idx_right;
        int idx_tree = 0;

        while (split_point) {
            idx_left = 2 * idx_tree + 1;
            idx_right = idx_left + 1;
            idx_tree = projected_query(j++) <= split_point ? idx_left : idx_right;
            split_point = tree[idx_tree];
        }
        
        const uvec& idx_one_tree = tree_leaves[n_tree][idx_tree - pow(2, depth) + 1];
        //uvec idx_one_tree = find(col_leaf_labels == idx_tree);
        idx_canditates.insert(idx_canditates.begin(), idx_one_tree.begin(), idx_one_tree.end());
    }

    std::sort(idx_canditates.begin(), idx_canditates.end());
    auto last = std::unique(idx_canditates.begin(), idx_canditates.end());
    idx_canditates.erase(last, idx_canditates.end());

    return knnCpp_T_indices(X, q, k, conv_to<uvec>::from(idx_canditates));
}


/**
 * No idea what this is used for ...
 */
//uvec Mrpt::query_canditates(const fvec& q, int k) {
//    fvec projected_query = random_matrix * q; // query vector q is passed as a reference to a col vector
//    std::vector<int> idx_canditates(n_trees * n_0);
//    int j = 0;
//
//    // std::cout << "projected_query.size(): " << projected_query.size() << ", idx_canditates.size(): " << idx_canditates.size() << std::endl;
//    for (int n_tree = 0; n_tree < n_trees; n_tree++) {
//        // std::cout << "n_tree: " << n_tree << ", n_trees: " << n_trees << ", j: " << j << std::endl;
//
//        const uvec& col_leaf_labels = leaf_labels.unsafe_col(n_tree);
//        const fvec& tree = trees.unsafe_col(n_tree);
//
//        // std::cout << "tree[0]: " << tree[0] << std::endl;
//
//        double split_point = tree[0];
//        int idx_left, idx_right;
//        int idx_tree = 0;
//
//        while (split_point) {
//            idx_left = 2 * idx_tree + 1;
//            idx_right = idx_left + 1;
//            idx_tree = projected_query(j++) <= split_point ? idx_left : idx_right;
//            split_point = tree[idx_tree];
//            // std::cout << "idx_left: " << idx_left << ", idx_right: " << idx_right << ", split_point: " << split_point << std::endl;
//            // bool temp = split_point == 0;
//            // std::cout << "split_point == 0: " <<  temp << std::endl;
//        }
//
//        uvec idx_one_tree = find(col_leaf_labels == idx_tree);
//        idx_canditates.insert(idx_canditates.begin(), idx_one_tree.begin(), idx_one_tree.end());
//    }
//
//    std::sort(idx_canditates.begin(), idx_canditates.end());
//    auto last = std::unique(idx_canditates.begin(), idx_canditates.end());
//    idx_canditates.erase(last, idx_canditates.end());
//    return conv_to<uvec>::from(idx_canditates);
//}


//void Mrpt::matrix_multiplication(const fvec& q) {
//    fvec projected_query = random_matrix * q;
//}

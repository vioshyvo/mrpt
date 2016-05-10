/********************************************************
 * Ville Hyvönen & Teemu Pitkänen                       *
 * HIIT / University of Helsinki                        *
 * ville.o.hyvonen<at>helsinki.fi                       *
 * teemu.pitkanen<at>cs.helsinki.fi                     *
 * 2016                                                 *
 ********************************************************/

#include "armadillo"
#include <cstdlib>
#include <queue>
#include "Mrpt.h"

using namespace arma;

/**
 * The constructor of the index. The inputs are the data for which the index 
 * will be built and additional parameters that affect the accuracy of the NN
 * approximation. Concisely, larger n_trees_ or n_0_ values improve accuracy but
 * slow down the queries. A general rule for the right balance is not known. The
 * constructor does not actually build the trees, but that is done by a separate
 * function 'grow' that has to be called before queries can be made. 
 * @param X_ - The data to be indexed. Samples as columns, features as rows.
 * @param n_trees_ - The number of trees to be used in the index.
 * @param n_0_ - The maximum leaf size to be used in the index.
 * @param id_ - A name used for filenames when saving.
 */
Mrpt::Mrpt(const fmat& X_, int n_trees_, int n_0_, std::string id_) : X(X_), n_trees(n_trees_), n_0(n_0_), id(id_){
    n_samples = X.n_cols; 
    dim = X.n_rows;
    if (n_0 == 1) // n_0==1 => leaves have sizes 1 or 2 (b/c 0.5 is impossible)
        depth = floor(log2(n_samples));
    else
        depth = ceil(log2(n_samples / n_0));
    n_pool = n_trees * depth;
    n_array = pow(2, depth + 1);
    split_points = fmat();
    random_matrix = fmat();
}

/**
 * The function whose call starts the actual index construction. Initializes 
 * arrays to store the tree structures and computes all the projections needed
 * later. Then repeatedly calls method grow_subtree that builds a single 
 * RP-tree.
 */
void Mrpt::grow() {
    split_points = zeros<fmat>(n_array, n_trees);
    uvec indices = linspace<uvec>(0, n_samples - 1, n_samples);

    // generate the random matrix and project the data set onto it
    random_matrix = conv_to<fmat>::from(randn(n_pool, dim));
    projected_data = random_matrix * X;
    
    // Grow the trees
    for (int n_tree = 0; n_tree < n_trees; n_tree++) {       
        first_idx = n_tree * depth;
        std::vector<uvec> t = grow_subtree(indices, 0, 0, n_tree); 
        tree_leaves.push_back(t);
    }
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
    v.insert( v.end(), w.begin(), w.end() );
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
        for (int idx : idx_one_tree)
            votes[idx]++;
    }
    
    /*
     * The following loop routes the query to extra leaves in the same trees 
     * handled already once above. The extra branches are popped from the 
     * priority queue and routed down the tree just as new root-to-leaf queries.
     */
    for (int b = 0; b < branches; b++){
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
        for (int idx : idx_one_tree)
            votes[idx]++;
    } 
    
    // Compute the actual NNs within the 'elect' objects with most votes
    uvec elected = sort_index(votes, "descend");
    elected.resize(elect);
    return exact_knn(X, q, k, elected);
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
uvec Mrpt::query(const fvec& q, int k, int elect) {
    
    fvec projected_query = random_matrix * q;
    uvec votes = zeros<uvec>(n_samples);
    
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
            } else {
                idx_tree = idx_right;
            }
            j++;
            split_point = tree[idx_tree];
        }
        const uvec& idx_one_tree = tree_leaves[n_tree][idx_tree - pow(2, depth) + 1];
        for (int idx : idx_one_tree)
            votes[idx]++;
    }

    // Compute the actual NNs within the 'elect' objects with most votes
    uvec elected = sort_index(votes, "descend");
    elected.resize(elect);
    return exact_knn(X, q, k, elected);
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
    fvec projected_query = random_matrix * q;
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
        idx_canditates.insert(idx_canditates.begin(), idx_one_tree.begin(), idx_one_tree.end());
    }

    std::sort(idx_canditates.begin(), idx_canditates.end());
    auto last = std::unique(idx_canditates.begin(), idx_canditates.end());
    idx_canditates.erase(last, idx_canditates.end());

    return exact_knn(X, q, k, conv_to<uvec>::from(idx_canditates));
}


/**
 * find k nearest neighbors from data for the query point
 * @param X - data matrix, row = data point, col = dimension
 * @param q - query point as a row matrix
 * @param k - number of neighbors searched for
 * @param indices - indices of the points in the original matrix where search is made
 * @return - indices of nearest neighbors in data matrix X as a column vector
 */
uvec exact_knn(const fmat& D, const fvec& q, uword k, uvec indices) {
    int n_cols = indices.size();
    fvec distances = fvec(n_cols);
    for (int i = 0; i < n_cols; i++)
        distances[i] = sum(pow((D.col(indices(i)) - q), 2));

    if(k == 1) {
        uvec ret(1);
        distances.min(ret[0]);
        return ret;
    }

    uvec sorted_indices = indices(sort_index(distances));
    return sorted_indices.size() > k ? sorted_indices.head(k): sorted_indices;
}
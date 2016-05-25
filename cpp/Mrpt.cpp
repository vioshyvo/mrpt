/********************************************************
 * Ville Hyvönen & Teemu Pitkänen                       *
 * HIIT / University of Helsinki                        *
 * ville.o.hyvonen<at>helsinki.fi                       *
 * teemu.pitkanen<at>cs.helsinki.fi                     *
 * 2016                                                 *
 ********************************************************/

#include <queue>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include "Mrpt.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

using namespace Eigen;

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
 * @param density - Expected ratio of non-zero components in a projection matrix.
 * @param metric - Distance metric to use, currently euclidean or angular.
 * @param id_ - A name used for filenames when saving.
 */
Mrpt::Mrpt(const MatrixXf &X_, int n_trees_, int n_0_, float density_,
           std::string metric_, std::string id_)
           : X(X_), n_trees(n_trees_), n_0(n_0_), density(density_), id(id_) {
    n_samples = X.cols();
    dim = X.rows();

    if (n_0 == 1) // n_0==1 => leaves have sizes 1 or 2 (b/c 0.5 is impossible)
        depth = floor(log2(n_samples));
    else
        depth = ceil(log2(n_samples / n_0));

    n_pool = n_trees * depth;
    n_array = pow(2, depth + 1);
    X_norms = X.colwise().squaredNorm();

    if (metric_ == "angular")
        metric = ANGULAR;
    else
        metric = EUCLIDEAN;
}

/**
 * The function whose call starts the actual index construction. Initializes 
 * arrays to store the tree structures and computes all the projections needed
 * later. Then repeatedly calls method grow_subtree that builds a single 
 * RP-tree.
 */
void Mrpt::grow() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // generate the random matrix and project the data set onto it
    if (density < 1) {
        sparse_random_matrix = buildSparseRandomMatrix(n_pool, dim, density, gen);
        projected_data.noalias() = sparse_random_matrix * X;
    } else {
        dense_random_matrix = buildDenseRandomMatrix(n_pool, dim, gen);
        projected_data.noalias() = dense_random_matrix * X;
    }

    split_points = MatrixXf::Zero(n_array, n_trees);
    VectorXi indices = VectorXi::LinSpaced(n_samples, 0, n_samples - 1);

    // Grow the trees
    for (int n_tree = 0; n_tree < n_trees; n_tree++) {
        first_idx = n_tree * depth;
        std::vector<VectorXi> t = grow_subtree(indices, 0, 0, n_tree); 
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
 * @return The leaves as a vector of arma::VectorXis
 */
std::vector<VectorXi> Mrpt::grow_subtree(const VectorXi &indices, int tree_level, int i, unsigned n_tree) {
    int n = indices.size();
    int idx_left = 2 * i + 1;
    int idx_right = idx_left + 1;

    if (tree_level == depth) {
        std::vector<VectorXi> v;
        v.push_back(indices);
        return v;
    }

    VectorXf projections = VectorXf(n);
    for (int i = 0; i < n; ++i)
        projections(i) = projected_data(first_idx + tree_level, indices(i));

    // sort indices of the projections based on their values
    VectorXi ordered = VectorXi::LinSpaced(n, 0, n - 1);
    std::sort(ordered.data(), ordered.data() + ordered.size(),
              [&projections](size_t i1, size_t i2) {return projections(i1) < projections(i2);});

    int split_point = n % 2 ? n / 2 : n / 2 - 1; // median split
    int idx_split_point = ordered(split_point);
    int idx_split_point2 = ordered(split_point + 1);

    split_points(i, n_tree) = n % 2 ? projections(idx_split_point) :
                              (projections(idx_split_point) + projections(idx_split_point2)) / 2;
    VectorXi left_indices = ordered.head(split_point + 1);
    VectorXi right_indices = ordered.tail(n - split_point - 1);

    VectorXi left_elems = VectorXi(left_indices.size());
    VectorXi right_elems = VectorXi(right_indices.size());

    for (int i = 0; i < left_indices.size(); ++i)
        left_elems(i) = indices(left_indices(i));
    for (int i = 0; i < right_indices.size(); ++i)
        right_elems(i) = indices(right_indices(i));

    std::vector<VectorXi> v = grow_subtree(left_elems, tree_level + 1, idx_left, n_tree);
    std::vector<VectorXi> w = grow_subtree(right_elems, tree_level + 1, idx_right, n_tree);
    v.insert(v.end(), w.begin(), w.end());
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
 * @param votes_required - The number of votes required for an object to be included in the linear search step
 * @param branches - The number of extra branches explored in the priority queue trick
 * @return The indices of the k approximate nearest neighbors in the original
 * data set for which the index was built.
 */
VectorXi Mrpt::query(const VectorXf &q, int k, int votes_required, int branches) {
    VectorXf projected_query = VectorXf(n_pool);

    if (density < 1)
        projected_query.noalias() = sparse_random_matrix * q;
    else
        projected_query.noalias() = dense_random_matrix * q;

    VectorXi votes = VectorXi::Zero(n_samples);
    std::priority_queue<Gap, std::vector<Gap>, std::greater<Gap>> pq;

    /*
     * The following loops over all trees, and routes the query to exactly one 
     * leaf in each.
     */
    int j = 0; // Used to find the correct projection value, increases through all trees
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
        int idx_tree = 0, idx_left, idx_right;
        float split_point = split_points(0, n_tree);

        for (int d = 0; d < depth; ++d) {
            idx_left = 2 * idx_tree + 1;
            idx_right = idx_left + 1;
            if (projected_query(j) <= split_point) {
                idx_tree = idx_left;
                pq.push(Gap(n_tree, idx_right, j + 1, split_point - projected_query(j)));
            } else {
                idx_tree = idx_right;
                pq.push(Gap(n_tree, idx_left, j + 1, projected_query(j) - split_point));
            }
            j++;
            split_point = split_points(idx_tree, n_tree);
        }

        const VectorXi &idx_one_tree = tree_leaves[n_tree][idx_tree - pow(2, depth) + 1];
        for (int i = 0; i < idx_one_tree.size(); ++i)
            votes(idx_one_tree(i))++;
    }

    /*
     * The following loop routes the query to extra leaves in the same trees 
     * handled already once above. The extra branches are popped from the 
     * priority queue and routed down the tree just as new root-to-leaf queries.
     */
    for (int b = 0; b < branches; ++b) {
        if (pq.empty()) break;
        Gap gap = pq.top();
        pq.pop();

        j = gap.level;
        int idx_tree = gap.node, idx_left, idx_right;
        float split_point = split_points(0, gap.tree);

        while (j % depth) {
            idx_left = 2 * idx_tree + 1;
            idx_right = idx_left + 1;
            if (projected_query(j) <= split_point) {
                idx_tree = idx_left;
                pq.push(Gap(gap.tree, idx_right, j + 1, split_point - projected_query(j)));
            } else {
                idx_tree = idx_right;
                pq.push(Gap(gap.tree, idx_left, j + 1, projected_query(j) - split_point));
            }
            j++;
            split_point = split_points(idx_tree, gap.tree);
        }

        const VectorXi &idx_one_tree = tree_leaves[gap.tree][idx_tree - pow(2, depth) + 1];
        for (int i = 0; i < idx_one_tree.size(); ++i)
            votes(idx_one_tree(i))++;
    }

    VectorXi elected(n_samples);
    for (int l = 0; l < 2; ++l) {
        j = 0;

        for (int i = 0; i < n_samples; ++i) {
            if (votes(i) >= votes_required)
                elected(j++) = i;
        }

        if (j >= k) break;

        /* 
         * If not enough samples had at least votes_required
         * votes, find the minimum amount of votes needed such
         * that the final search set size has at least k samples
         */
        int vote_count[n_trees + branches + 1] = {0};
        for (int i = 0; i < n_samples; ++i)
            vote_count[votes(i)]++;

        int n_elect = 0, min_votes = n_trees + branches;
        for (; min_votes; --min_votes) {
            n_elect += vote_count[min_votes];
            if (n_elect >= k) break;
        }

        votes_required = min_votes;
    }

    VectorXi indices = elected.head(j);
    return exact_knn(X, X_norms, q, k, indices, metric);
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
VectorXi Mrpt::query(const VectorXf &q, int k) {
    VectorXf projected_query = VectorXf(n_pool);

    if (density < 1)
        projected_query.noalias() = sparse_random_matrix * q;
    else
        projected_query.noalias() = dense_random_matrix * q;

    VectorXi votes = VectorXi::Zero(n_samples);
    int j = 0;

    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
        float split_point = split_points(0, n_tree);
        int idx_tree = 0, idx_left, idx_right;

        for (int d = 0; d < depth; ++d) {
            idx_left = 2 * idx_tree + 1;
            idx_right = idx_left + 1;
            idx_tree = projected_query(j++) <= split_point ? idx_left : idx_right;
            split_point = split_points(idx_tree, n_tree);
        }

        const VectorXi &idx_one_tree = tree_leaves[n_tree][idx_tree - pow(2, depth) + 1];
        for (int i = 0; i < idx_one_tree.size(); ++i)
            votes(idx_one_tree(i))++;
    }

    j = 0;
    VectorXi elected(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        if (votes(i) >= 1) elected(j++) = i;
    }

    VectorXi indices = elected.head(j);
    return exact_knn(X, X_norms, q, k, indices, metric);
}

/**
 * find k nearest neighbors from data for the query point
 * @param X - data matrix, row = data point, col = dimension
 * @param q - query point as a row matrix
 * @param k - number of neighbors searched for
 * @param indices - indices of the points in the original matrix where search is made
 * @return - indices of nearest neighbors in data matrix X as a column vector
 */
VectorXi exact_knn(const MatrixXf &D, const VectorXf &D_norms, const VectorXf &q, unsigned k,
                   const VectorXi &indices, Metric metric) {
    unsigned n_cols = indices.size();

    VectorXf distances(n_cols);
    if (metric == EUCLIDEAN) {
        #pragma omp parallel for
        for (unsigned i = 0; i < n_cols; ++i)
            distances(i) = D_norms(indices(i)) - 2 * q.dot(D.col(indices(i)));
    } else {
        #pragma omp parallel for
        for (unsigned i = 0; i < n_cols; ++i)
            distances(i) = -q.dot(D.col(indices(i)));
    }

    if (k == 1) {
        MatrixXf::Index index;
        distances.minCoeff(&index);
        VectorXi ret(1);
        ret(0) = (int) indices(index);
        return ret;
    }

    VectorXi idx = VectorXi::LinSpaced(distances.size(), 0, distances.size() - 1);
    std::partial_sort(idx.data(), idx.data() + k, idx.data() + idx.size(),
                      [&distances](size_t i1, size_t i2) {return distances(i1) < distances(i2);});

    VectorXi result(k);
    for (unsigned i = 0; i < k; ++i) result(i) = indices(idx(i));
    return result;
}

/**
 * Builds a random sparse matrix for use in random projection. The components of
 * the matrix are drawn from the distribution
 *
 * -1  w.p. 1 / 2s
 *  0  w.p. 1 - 1 / s
 * +1  w.p. 1 / 2s
 *
 * where s = 1 / density.
 * @param rows - The number of rows in the resulting matrix.
 * @param cols - The number of columns in the resulting matrix.
 * @param density - Expected ratio of non-zero components in the resulting matrix.
 * @param gen - A random number engine.
 */
SparseMatrix<float> buildSparseRandomMatrix(int rows, int cols, float density, std::mt19937 &gen) {
    std::uniform_real_distribution<float> uni_dist(0, 1);
    SparseMatrix<float> random_matrix = SparseMatrix<float>(rows, cols);

    std::vector<Triplet<float> > triplets;
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            if (uni_dist(gen) > density) continue;
            float value = uni_dist(gen) <= 0.5 ? -1 : 1;
            triplets.push_back(Triplet<float>(j, i, value));
        }
    }

    random_matrix.setFromTriplets(triplets.begin(), triplets.end());
    random_matrix.makeCompressed();
    return random_matrix;
}

/*
 * Builds a random dense matrix for use in random projection. The components of
 * the matrix are drawn from the standard normal distribution.
 * @param rows - The number of rows in the resulting matrix.
 * @param cols - The number of rows in the resulting matrix.
 * @param gen - A random number engine.
 */
MatrixXf buildDenseRandomMatrix(int rows, int cols, std::mt19937 &gen) {
    std::normal_distribution<float> normal_dist(0, 1);
    return MatrixXf::Zero(rows, cols).unaryExpr(
            [&normal_dist, &gen](float _) -> float { return normal_dist(gen); });
}

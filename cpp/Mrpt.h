/********************************************************
 * Ville Hyvönen & Teemu Pitkänen                       *
 * HIIT / University of Helsinki                        *
 * ville.o.hyvonen<at>helsinki.fi                       *
 * teemu.pitkanen<at>cs.helsinki.fi                     *
 * 2016                                                 *
 ********************************************************/

#ifndef CPP_MRPT_H_
#define CPP_MRPT_H_

#include <algorithm>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

using namespace Eigen;

class Mrpt {
 public:
    /**
    * The constructor of the index. The inputs are the data for which the index
    * will be built and additional parameters that affect the accuracy of the NN
    * approximation. Concisely, larger n_trees_ or smaller depth values improve
    * accuracy but slow down the queries. A general rule for the right balance is
    * not known. The constructor does not actually build the trees, but that is
    * done by a separate function 'grow' that has to be called before queries can
    * be made.
    * @param X_ - Pointer to a matrix containing the data.
    * @param n_trees_ - The number of trees to be used in the index.
    * @param depth_ - The depth of the trees.
    * @param density_ - Expected ratio of non-zero components in a projection matrix.
    */


    Mrpt(const Map<const MatrixXf> *X_, int n_trees_, int depth_, float density_, bool test_ = true) :
        X(X_),
        Y(NULL),
        n_samples(X_->cols()),
        dim(X_->rows()),
        n_trees(n_trees_),
        depth(depth_),
        density(density_),
        n_pool(n_trees_ * depth_),
        n_array(1 << (depth_ + 1)),
        sparse(false),
        test(test_)
    { }

    Mrpt(const SparseMatrix<float> *Y_, int n_trees_, int depth_, float density_, bool test_ = true) :
        X(NULL),
        Y(Y_),
        n_samples(Y_->cols()),
        dim(Y_->rows()),
        n_trees(n_trees_),
        depth(depth_),
        density(density_),
        n_pool(n_trees_ * depth_),
        n_array(1 << (depth_ + 1)),
        sparse(true),
        test(test_)
    { }

    ~Mrpt() {}

    /**
    * The function whose call starts the actual index construction. Initializes
    * arrays to store the tree structures and computes all the projections needed
    * later. Then repeatedly calls method grow_subtree that builds a single RP-tree.
    */
    void grow() {
        X_norms = VectorXf(n_samples);
        if (sparse) {
            for (int i = 0; i < n_samples; ++i) {
                X_norms(i) = Y->col(i).squaredNorm();
            }
        } else {
            X_norms.noalias() = X->colwise().squaredNorm();
        }

        // generate the random matrix
        density < 1 ? build_sparse_random_matrix() : build_dense_random_matrix();

        split_points = MatrixXf(n_array, n_trees);
        VectorXi indices(n_samples);
        std::iota(indices.data(), indices.data() + n_samples, 0);

        tree_leaves = std::vector<std::vector<VectorXi>>(n_trees);

        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            SparseMatrix<float> tree_projections_sparse;
            MatrixXf tree_projections;
            std::vector<VectorXi> t;

            if (sparse) {
                    if(density < 1) {
                        tree_projections_sparse = sparse_random_matrix.middleRows(n_tree * depth, depth) * *Y;
                        t = grow_subtree(indices, 0, 0, n_tree, tree_projections_sparse);
                    } else {
                        tree_projections.noalias() = dense_random_matrix.middleRows(n_tree * depth, depth) * *Y;
                        t = grow_subtree(indices, 0, 0, n_tree, tree_projections);
                    }


            } else {
                if (density < 1)
                    tree_projections.noalias() = sparse_random_matrix.middleRows(n_tree * depth, depth) * *X;
                else
                    tree_projections.noalias() = dense_random_matrix.middleRows(n_tree * depth, depth) * *X;

                t = grow_subtree(indices, 0, 0, n_tree, tree_projections);
            }

            tree_leaves[n_tree] = t;
        }

    }

    int sparse_query(const SparseMatrix<float>::ColXpr &q, int k, int votes_required, int *out) const {
        int max_leaf_size = n_samples / (1 << depth) + 1;
        VectorXi elected(n_trees * max_leaf_size);

        SparseVector<float> projected_query_sparse;
        VectorXf projected_query(n_pool);
        VectorXf dense_projected_query;
        int n_elected;

        if(density < 1) {
            projected_query_sparse = sparse_random_matrix * q;
            dense_projected_query = VectorXf::Zero(projected_query.rows());

            for (SparseVector<float>::InnerIterator it(projected_query_sparse); it; ++it)
                dense_projected_query[it.index()] = it.value();

            n_elected = sparse_elect(dense_projected_query, k, votes_required, elected.data()); // dense --> sparse
        } else {
            projected_query = dense_random_matrix * q;
            n_elected = sparse_elect(projected_query, k, votes_required, elected.data()); // dense --> sparse
        }

        return sparse_exact_knn(q, k, elected, n_elected, out);
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
    * @param q - The query object whose neighbors the function finds
    * @param k - The number of neighbors the user wants the function to return
    * @param votes_required - The number of votes required for an object to be included in the linear search step
    * @param out - The output buffer
    * @return
    */
    int query(const Map<VectorXf> &q, int k, int votes_required, int *out) const {
        int max_leaf_size = n_samples / (1 << depth) + 1;
        VectorXi elected(n_trees * max_leaf_size);

        VectorXf projected_query(n_pool);
        if (density < 1)
            projected_query.noalias() = sparse_random_matrix * q;
        else
            projected_query.noalias() = dense_random_matrix * q;

        int n_elected = elect(projected_query, k, votes_required, elected.data());
        return exact_knn(q, k, elected, n_elected, out);
    }

    int elect(const VectorXf &projected_query, int k, int votes_required, int *out) const {
        int found_leaves[n_trees];

        /*
        * The following loops over all trees, and routes the query to exactly one
        * leaf in each.
        */
        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            int idx_tree = 0;
            for (int d = 0; d < depth; ++d) {
                const int j = n_tree * depth + d;
                const int idx_left = 2 * idx_tree + 1;
                const int idx_right = idx_left + 1;
                const float split_point = split_points(idx_tree, n_tree);
                if (projected_query(j) <= split_point) {
                    idx_tree = idx_left;
                } else {
                    idx_tree = idx_right;
                }
            }
            found_leaves[n_tree] = idx_tree - (1 << depth) + 1;
        }

        int n_elected = 0;
        VectorXi votes = VectorXi::Zero(n_samples);

        // count votes
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            const VectorXi &idx_one_tree = tree_leaves[n_tree][found_leaves[n_tree]];
            const int nn = idx_one_tree.size(), *data = idx_one_tree.data();
            for (int i = 0; i < nn; ++i, ++data) {
                if (++votes(*data) == votes_required) {
                    out[n_elected++] = *data;
                }
            }
        }

        if(!test) {
            if (n_elected < k) {
               /*
                * If not enough samples had at least votes_required
                * votes, find the maximum amount of votes needed such
                * that the final search set size has at least k samples
                */
                VectorXf::Index max_index;
                votes.maxCoeff(&max_index);
                int max_votes = votes(max_index);

                VectorXi vote_count = VectorXi::Zero(max_votes + 1);
                for (int i = 0; i < n_samples; ++i)
                    vote_count(votes(i))++;

                for (int would_elect = 0; max_votes; --max_votes) {
                    would_elect += vote_count(max_votes);
                    if (would_elect >= k) break;
                }

                for (int i = 0; i < n_samples; ++i) {
                    if (votes(i) >= max_votes && votes(i) < votes_required)
                        out[n_elected++] = i;
                }
            }
        }

        return n_elected;
    }


int sparse_elect(VectorXf &projected_query, int k, int votes_required, int *out) const { // SparseVector<float> --> VectorXf
        int found_leaves[n_trees];

        /*
        * The following loops over all trees, and routes the query to exactly one
        * leaf in each.
        */
        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            int idx_tree = 0;
            for (int d = 0; d < depth; ++d) {
                const int j = n_tree * depth + d;
                const int idx_left = 2 * idx_tree + 1;
                const int idx_right = idx_left + 1;
                const float split_point = split_points(idx_tree, n_tree);
                if (projected_query(j) <= split_point) { // projected_query.coeffRef() <-- projected_query(j)
                    idx_tree = idx_left;
                } else {
                    idx_tree = idx_right;
                }
            }
            found_leaves[n_tree] = idx_tree - (1 << depth) + 1;
        }

        int n_elected = 0;
        VectorXi votes = VectorXi::Zero(n_samples);

        // count votes
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            const VectorXi &idx_one_tree = tree_leaves[n_tree][found_leaves[n_tree]];
            const int nn = idx_one_tree.size(), *data = idx_one_tree.data();
            for (int i = 0; i < nn; ++i, ++data) {
                if (++votes(*data) == votes_required) {
                    out[n_elected++] = *data;
                }
            }
        }

        if(!test) {
            if (n_elected < k) {
                /*
                * If not enough samples had at least votes_required
                * votes, find the maximum amount of votes needed such
                * that the final search set size has at least k samples
                */
                VectorXf::Index max_index;
                votes.maxCoeff(&max_index);
                int max_votes = votes(max_index);

                VectorXi vote_count = VectorXi::Zero(max_votes + 1);
                for (int i = 0; i < n_samples; ++i)
                    vote_count(votes(i))++;

                for (int would_elect = 0; max_votes; --max_votes) {
                    would_elect += vote_count(max_votes);
                    if (would_elect >= k) break;
                }

                for (int i = 0; i < n_samples; ++i) {
                    if (votes(i) >= max_votes && votes(i) < votes_required)
                        out[n_elected++] = i;
                }
            }
        }

        return n_elected;
    }

    /**
    * find k nearest neighbors from data for the query point
    * @param q - query point as a vector
    * @param k - number of neighbors searched for
    * @param indices - indices of the points in the original matrix where the search is made
    * @param out - output buffer
    * @return
    */
    int exact_knn(const Map<VectorXf> &q, int k, const VectorXi &indices, int n_elected, int *out) const {
        VectorXf distances(n_elected);

        if (sparse) {
            #pragma omp parallel for
            for (int i = 0; i < n_elected; ++i) {
                distances(i) = X_norms(indices(i)) - 2 * Y->col(indices(i)).dot(q);
            }
        } else {
            #pragma omp parallel for
            for (int i = 0; i < n_elected; ++i) {
                distances(i) = X_norms(indices(i)) - 2 * X->col(indices(i)).dot(q);
            }
        }

        if (k == 1) {
            MatrixXf::Index index;
            distances.minCoeff(&index);
            out[0] = indices(index);
            return 1;
        }

        int mmin = std::min(k, n_elected);

        VectorXi idx(n_elected);
        std::iota(idx.data(), idx.data() + n_elected, 0);
        std::nth_element(idx.data(), idx.data() + mmin, idx.data() + n_elected,
                         [&distances](int i1, int i2) {return distances(i1) < distances(i2);});

        for (int i = 0; i < mmin; ++i) out[i] = indices(idx(i));
        return mmin;
    }

    int sparse_exact_knn(const SparseMatrix<float>::ColXpr &q,
                          int k, const VectorXi &indices, int n_elected, int *out) const {
        VectorXf distances(n_elected);

        #pragma omp parallel for
        for (int i = 0; i < n_elected; ++i) {
            distances(i) = X_norms(indices(i)) - 2 * Y->col(indices(i)).dot(q);
        }

        if (k == 1) {
            MatrixXf::Index index;
            distances.minCoeff(&index);
            out[0] = indices(index);
            return 1;
        }

        int mmin = std::min(k, n_elected);


        VectorXi idx(n_elected);
        std::iota(idx.data(), idx.data() + n_elected, 0);
        std::partial_sort(idx.data(), idx.data() + mmin, idx.data() + n_elected,
                          [&distances](int i1, int i2) {return distances(i1) < distances(i2);});

        for (int i = 0; i < mmin; ++i) out[i] = indices(idx(i));
        return mmin;
}

    /**
    * Saves the index to a file.
    * @param path - Filepath to the output file.
    * @return True if saving succeeded, false otherwise.
    */
    bool save(const char *path) const {
        FILE *fd;
        if ((fd = fopen(path, "wb")) == NULL)
            return false;

        fwrite(split_points.data(), sizeof(float), n_array * n_trees, fd);

        // save tree leaves
        for (int i = 0; i < n_trees; ++i) {
            int sz = tree_leaves[i].size();
            fwrite(&sz, sizeof(int), 1, fd);
            for (int j = 0; j < sz; ++j) {
                int lsz = tree_leaves[i][j].size();
                fwrite(&lsz, sizeof(int), 1, fd);
                fwrite(tree_leaves[i][j].data(), sizeof(int), lsz, fd);
            }
        }

        // save random matrix
        if (density < 1) {
            int non_zeros = sparse_random_matrix.nonZeros();
            fwrite(&non_zeros, sizeof(int), 1, fd);
            for (int k = 0; k < sparse_random_matrix.outerSize(); ++k) {
                for (SparseMatrix<float, RowMajor>::InnerIterator it(sparse_random_matrix, k); it; ++it) {
                    float val = it.value();
                    int row = it.row(), col = it.col();
                    fwrite(&row, sizeof(int), 1, fd);
                    fwrite(&col, sizeof(int), 1, fd);
                    fwrite(&val, sizeof(float), 1, fd);
                }
            }
        } else {
            fwrite(dense_random_matrix.data(), sizeof(float), n_pool * dim, fd);
        }

        fclose(fd);
        return true;
    }

    /**
    * Loads the index from a file.
    * @param path - Filepath to the index file.
    * @return True if loading succeeded, false otherwise.
    */
    bool load(const char *path) {
        FILE *fd;
        if ((fd = fopen(path, "rb")) == NULL)
            return false;

        split_points = MatrixXf(n_array, n_trees);
        fread(split_points.data(), sizeof(float), n_array * n_trees, fd);

        // load tree leaves
        tree_leaves = std::vector<std::vector<VectorXi>>(n_trees);
        for (int i = 0; i < n_trees; ++i) {
            int sz;
            fread(&sz, sizeof(int), 1, fd);
            std::vector<VectorXi> leaves(sz);
            for (int j = 0; j < sz; ++j) {
                int leaf_size;
                fread(&leaf_size, sizeof(int), 1, fd);
                VectorXi samples(leaf_size);
                fread(samples.data(), sizeof(int), leaf_size, fd);
                leaves[j] = samples;
            }
            tree_leaves[i] = leaves;
        }

        // load random matrix
        if (density < 1) {
            int non_zeros;
            fread(&non_zeros, sizeof(int), 1, fd);

            sparse_random_matrix = SparseMatrix<float>(n_pool, dim);
            std::vector<Triplet<float>> triplets;
            for (int k = 0; k < non_zeros; ++k) {
                int row, col;
                float val;
                fread(&row, sizeof(int), 1, fd);
                fread(&col, sizeof(int), 1, fd);
                fread(&val, sizeof(float), 1, fd);
                triplets.push_back(Triplet<float>(row, col, val));
            }

            sparse_random_matrix.setFromTriplets(triplets.begin(), triplets.end());
            sparse_random_matrix.makeCompressed();
        } else {
            dense_random_matrix = Matrix<float, Dynamic, Dynamic, RowMajor>(n_pool, dim);
            fread(dense_random_matrix.data(), sizeof(float), n_pool * dim, fd);
        }

        fclose(fd);
        return true;
    }

 private:
    /**
    * Builds a single random projection tree. The tree is constructed by recursively
    * projecting the data on a random vector and splitting into two by the median.
    * @param indices - The indices left in this branch
    * @param tree_level - The level in tree where the recursion is at
    * @param i - The index within the tree where we are at
    * @param n_tree - The index of the tree within the index
    * @param tree_projections - Precalculated projection values for the current tree
    * @return The leaves as a vector of VectorXis
    */
    std::vector<VectorXi> grow_subtree(const VectorXi &indices, int tree_level, int i, int n_tree, const MatrixXf &tree_projections) {
        int n = indices.size();
        int idx_left = 2 * i + 1;
        int idx_right = idx_left + 1;

        if (tree_level == depth) {
            std::vector<VectorXi> v;
            v.push_back(indices);
            return v;
        }

        VectorXf projections(n);
        for (int i = 0; i < n; ++i)
            projections(i) = tree_projections(tree_level, indices(i));

        // sort indices of the projections based on their values
        VectorXi ordered(n);
        std::iota(ordered.data(), ordered.data() + n, 0);
        std::sort(ordered.data(), ordered.data() + ordered.size(),
                [&projections](int i1, int i2) {return projections(i1) < projections(i2);});

        int split_point = (n % 2) ? n / 2 : n / 2 - 1; // median split
        int idx_split_point = ordered(split_point);
        int idx_split_point2 = ordered(split_point + 1);

        split_points(i, n_tree) = (n % 2) ? projections(idx_split_point) :
                                  (projections(idx_split_point) + projections(idx_split_point2)) / 2;
        VectorXi left_indices = ordered.head(split_point + 1);
        VectorXi right_indices = ordered.tail(n - split_point - 1);

        VectorXi left_elems = VectorXi(left_indices.size());
        VectorXi right_elems = VectorXi(right_indices.size());

        for (int i = 0; i < left_indices.size(); ++i)
            left_elems(i) = indices(left_indices(i));
        for (int i = 0; i < right_indices.size(); ++i)
            right_elems(i) = indices(right_indices(i));

        std::vector<VectorXi> v = grow_subtree(left_elems, tree_level + 1, idx_left, n_tree, tree_projections);
        std::vector<VectorXi> w = grow_subtree(right_elems, tree_level + 1, idx_right, n_tree, tree_projections);
        v.insert(v.end(), w.begin(), w.end());
        return v;
    }

    /**
    * Builds a random sparse matrix for use in random projection. The components of
    * the matrix are drawn from the distribution
    *
    *       0 w.p. 1 - a
    * N(0, 1) w.p. a
    *
    * where a = density.
    */
    void build_sparse_random_matrix() {
        sparse_random_matrix = SparseMatrix<float, RowMajor>(n_pool, dim);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> uni_dist(0, 1);
        std::normal_distribution<float> norm_dist(0, 1);

        std::vector<Triplet<float>> triplets;
        for (int j = 0; j < n_pool; ++j) {
            for (int i = 0; i < dim; ++i) {
                if (uni_dist(gen) > density) continue;
                triplets.push_back(Triplet<float>(j, i, norm_dist(gen)));
            }
        }

        sparse_random_matrix.setFromTriplets(triplets.begin(), triplets.end());
        sparse_random_matrix.makeCompressed();
    }

    /*
    * Builds a random dense matrix for use in random projection. The components of
    * the matrix are drawn from the standard normal distribution.
    */
    void build_dense_random_matrix() {
        dense_random_matrix = Matrix<float, Dynamic, Dynamic, RowMajor>(n_pool, dim);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> normal_dist(0, 1);

        std::generate(dense_random_matrix.data(), dense_random_matrix.data() + n_pool * dim,
                      [&normal_dist, &gen] { return normal_dist(gen); });
    }

    const Map<const MatrixXf> *X; // the data matrix
    const SparseMatrix<float> *Y; // the data matrix
    VectorXf X_norms; // cache norms of the observations in X for distance calculations
    MatrixXf split_points; // all split points in all trees
    std::vector<std::vector<VectorXi>> tree_leaves; // contains all leaves of all trees,
                                                    // indexed as tree_leaves[tree number][leaf number][index in leaf]
    Matrix<float, Dynamic, Dynamic, RowMajor> dense_random_matrix; // random vectors needed for all the RP-trees
    SparseMatrix<float, RowMajor> sparse_random_matrix; // random vectors needed for all the RP-trees

    const int n_samples; // sample size of data
    const int dim; // dimension of data
    const int n_trees; // number of RP-trees
    const int depth; // depth of an RP-tree with median split
    const float density; // expected ratio of non-zero components in a projection matrix
    const int n_pool; // amount of random vectors needed for all the RP-trees
    const int n_array; // length of the one RP-tree as array
    const bool sparse; // whether the data is sparse or not
    const bool test; // false: if the voting returns less than k nn-canditates, drop number of votes and try again
                     // true: do nothing, this leads to error, unless you handle the input properly
};

#endif // CPP_MRPT_H_

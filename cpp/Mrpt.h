#ifndef CPP_MRPT_H_
#define CPP_MRPT_H_

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cmath>

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
    Mrpt(const Map<const MatrixXf> *X_, int n_trees_, int depth_, float density_) :
        X(X_),
        n_samples(X_->cols()),
        dim(X_->rows()),
        n_trees(n_trees_),
        depth(depth_),
        density(density_),
        n_pool(n_trees_ * depth_),
        n_array((1 << depth_) - 1)
    { }

    ~Mrpt() {}

    /**
    * The function whose call starts the actual index construction. Initializes
    * arrays to store the tree structures and computes all the projections needed
    * later. Then repeatedly calls method grow_subtree that builds a single RP-tree.
    * @param seed - A seed given to a rng when generating random vectors;
    * a default value 0 initializes the rng randomly with rd()
    */
    void grow(int seed = 0) {
        // generate the random matrix
        density < 1 ? build_sparse_random_matrix(seed) : build_dense_random_matrix(seed);

        split_points = MatrixXf(n_array, n_trees);
        tree_leaves = std::vector<std::vector<VectorXi>>(n_trees);

        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            MatrixXf tree_projections;

            if (density < 1)
                tree_projections.noalias() = sparse_random_matrix.middleRows(n_tree * depth, depth) * *X;
            else
                tree_projections.noalias() = dense_random_matrix.middleRows(n_tree * depth, depth) * *X;

            VectorXi indices(n_samples);
            std::iota(indices.data(), indices.data() + n_samples, 0);

            std::vector<VectorXi> t = grow_subtree(indices, 0, 0, n_tree, tree_projections);
            tree_leaves[n_tree] = t;
        }
    }

    /**
    * This function finds the k approximate nearest neighbors of the query object
    * q. The accuracy of the query depends on both the parameters used for index
    * construction and additional parameters given to this function. This
    * function implements two tricks to improve performance. The voting trick
    * interprets each index object in leaves returned by tree traversals as votes,
    * and only performs the final linear search with the 'elect' most voted
    * objects.
    * @param q - The query object whose neighbors the function finds
    * @param k - The number of neighbors the user wants the function to return
    * @param votes_required - The number of votes required for an object to be included in the linear search step
    * @param out - The output buffer for the indices of the k approximate nearest neighbors
    * @param out_distances - Output buffer for distances of the k approximate nearest neighbors (optional parameter)
    * @return
    */
    void query(const Map<VectorXf> &q, int k, int votes_required, int *out, float *out_distances = nullptr) const {
        VectorXf projected_query(n_pool);
        if (density < 1)
            projected_query.noalias() = sparse_random_matrix * q;
        else
            projected_query.noalias() = dense_random_matrix * q;

        VectorXi found_leaves(n_trees);

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
            found_leaves(n_tree) = idx_tree - (1 << depth) + 1;
        }

        int n_elected = 0, max_leaf_size = n_samples / (1 << depth) + 1;
        VectorXi elected(n_trees * max_leaf_size);
        VectorXi votes = VectorXi::Zero(n_samples);

        // count votes
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            const VectorXi &idx_one_tree = tree_leaves[n_tree][found_leaves(n_tree)];
            const int nn = idx_one_tree.size(), *data = idx_one_tree.data();
            for (int i = 0; i < nn; ++i, ++data) {
                if (++votes(*data) == votes_required) {
                    elected(n_elected++) = *data;
                }
            }
        }

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
                    elected(n_elected++) = i;
            }
        }

        exact_knn(q, k, elected, n_elected, out, out_distances);
    }

    /**
    * find k nearest neighbors from data for the query point
    * @param q - query point as a vector
    * @param k - number of neighbors searched for
    * @param indices - indices of the points in the original matrix where the search is made
    * @param out - output buffer for the indices of the k approximate nearest neighbors
    * @param out_distances - output buffer for distances of the k approximate nearest neighbors (optional parameter)
    * @return
    */
    void exact_knn(const Map<VectorXf> &q, int k, const VectorXi &indices, int n_elected, int *out, float *out_distances = nullptr) const {
        VectorXf distances(n_elected);

        #pragma omp parallel for
        for (int i = 0; i < n_elected; ++i)
            distances(i) = (X->col(indices(i)) - q).squaredNorm();

        if (k == 1) {
            MatrixXf::Index index;
            distances.minCoeff(&index);
            out[0] = indices(index);

            if(out_distances) {
              out_distances[0] = std::sqrt(distances(index));
            }
            return;
        }

        VectorXi idx(n_elected);
        std::iota(idx.data(), idx.data() + n_elected, 0);
        std::partial_sort(idx.data(), idx.data() + k, idx.data() + n_elected,
                         [&distances](int i1, int i2) {return distances(i1) < distances(i2);});

        for (int i = 0; i < k; ++i) out[i] = indices(idx(i));

        if(out_distances) {
          for(int i = 0; i < k; ++i) out_distances[i] = std::sqrt(distances(idx(i)));
        }
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

    /**
    * Accessor for split points of trees (for testing purposes)
    * @param tree - index of tree in (0, ... , T-1)
    * @param index - the index of branch in (0, ... , (2^depth) - 1):
    * 0 = root
    * 1 = first branch of first level
    * 2 = second branch of first level
    * 3 = first branch of second level etc.
    * @return split point of index:th branch of tree:th tree
    */
    float get_split_point(int tree, int index) const {
      return split_points(index, tree);
    }

    /**
    * Accessor for point stored in leaves of trees (for testing purposes)
    * @param tree - index of tree in (0, ... T-1)
    * @param leaf - index of leaf in (0, ... , 2^depth)
    * @param index - index of a data point in a leaf
    * @return index of index:th data point in leaf:th leaf of tree:th tree
    */
    int get_leaf_point(int tree, int leaf, int index) const {
      return tree_leaves[tree][leaf](index);
    }

    /**
    * Accessor for the number of points in a leaf of a tree (for test purposes)
    * @param tree - index of tree in (0, ... T-1)
    * @param leaf - index of leaf in (0, ... , 2^depth)
    * @return - number of data points in leaf:th leaf of tree:th tree
    */
    int get_leaf_size(int tree, int leaf) const {
      return tree_leaves[tree][leaf].size();
    }

    /**
    * @return - number of trees in the index
    */
    int get_n_trees() const {
      return split_points.cols();
    }

    /**
    * @return - depth of trees of index
    */
    int get_depth() const {
      if(sparse_random_matrix.rows() > 0) {
        return sparse_random_matrix.rows() / get_n_trees();
      } else {
        return dense_random_matrix.rows() / get_n_trees();
      }
    }

    /**
    * @return - number of points of the data set from which the index is built
    */
    int get_n_points() const {
      return n_samples;
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
    std::vector<VectorXi> grow_subtree(VectorXi &indices, int tree_level, int i, int n_tree, const MatrixXf &tree_projections) {
        int n = indices.size();
        int idx_left = 2 * i + 1;
        int idx_right = idx_left + 1;

        if (tree_level == depth) {
            std::vector<VectorXi> v;
            v.push_back(indices);
            return v;
        }

        std::nth_element(indices.data(), indices.data() + n/2, indices.data() + n,
            [&tree_projections, tree_level] (int i1, int i2) {
              return tree_projections(tree_level, i1) < tree_projections(tree_level, i2);
            });
        int idx_split = (n % 2) ? n / 2 + 1 : n / 2;
        VectorXi left_elems = indices.head(idx_split);
        VectorXi right_elems = indices.tail(n - idx_split);

        if(n % 2) {
          split_points(i, n_tree) = tree_projections(tree_level, indices(n/2));
        } else {
          auto left_it = std::max_element(left_elems.data(),
              left_elems.data() + left_elems.size(),
              [&tree_projections, tree_level] (int i1, int i2) {
                return tree_projections(tree_level, i1) < tree_projections(tree_level, i2);
              });
          split_points(i, n_tree) = (tree_projections(tree_level, indices(n/2)) +
            tree_projections(tree_level, *left_it)) / 2.0;
        }

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
    *
    * @param seed - A seed given to a rng when generating random vectors;
    * a default value 0 initializes the rng randomly with rd()
    */
    void build_sparse_random_matrix(int seed = 0) {
        sparse_random_matrix = SparseMatrix<float, RowMajor>(n_pool, dim);

        std::random_device rd;
        int s = seed ? seed : rd();
        std::mt19937 gen(s);
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
    * @param seed - A seed given to a rng when generating random vectors;
    * a default value 0 initializes the rng randomly with rd()
    */
    void build_dense_random_matrix(int seed = 0) {
        dense_random_matrix = Matrix<float, Dynamic, Dynamic, RowMajor>(n_pool, dim);

        std::random_device rd;
        int s = seed ? seed : rd();
        std::mt19937 gen(s);
        std::normal_distribution<float> normal_dist(0, 1);

        std::generate(dense_random_matrix.data(), dense_random_matrix.data() + n_pool * dim,
                      [&normal_dist, &gen] { return normal_dist(gen); });
    }

    const Map<const MatrixXf> *X; // the data matrix
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
};

#endif // CPP_MRPT_H_

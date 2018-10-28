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
    * The constructor of the index.The constructor does not actually build
    * the index but that is done by the function 'grow' which has to be called
    * before queries can be made.
    * @param X_ - Pointer to the Eigen::Map which refers to the data matrix.
    */
    Mrpt(const Map<const MatrixXf> *X_) :
        X(X_),
        n_samples(X_->cols()),
        dim(X_->rows()) {}

    ~Mrpt() {}

    /**
    * The function whose call starts the actual index construction. Initializes
    * arrays to store the tree structures and computes all the projections needed
    * later. Then repeatedly calls method grow_subtree that builds a single RP-tree.
    * @param n_trees_ - The number of trees to be used in the index.
    * @param depth_ - The depth of the trees.
    * @param density_ - Expected ratio of non-zero components in a projection matrix.
    * @param seed - A seed given to a rng when generating random vectors;
    * a default value 0 initializes the rng randomly with rd()
    */
    void grow(int n_trees_, int depth_, float density_, int seed = 0) {
        n_trees = n_trees_;
        depth = depth_;
        density = density_;
        n_pool = n_trees_ * depth_;
        n_array = 1 << (depth_ + 1);

        density < 1 ? build_sparse_random_matrix(seed) : build_dense_random_matrix(seed);

        split_points = MatrixXf(n_array, n_trees);
        tree_leaves = std::vector<std::vector<int>>(n_trees);

        count_first_leaf_indices();

        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            MatrixXf tree_projections;

            if (density < 1)
                tree_projections.noalias() = sparse_random_matrix.middleRows(n_tree * depth, depth) * *X;
            else
                tree_projections.noalias() = dense_random_matrix.middleRows(n_tree * depth, depth) * *X;

            tree_leaves[n_tree] = std::vector<int>(n_samples);
            std::vector<int> &indices = tree_leaves[n_tree];
            std::iota(indices.begin(), indices.end(), 0);

            grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, tree_projections);
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

        std::vector<int> found_leaves(n_trees);

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

        int n_elected = 0, max_leaf_size = n_samples / (1 << depth) + 1;
        VectorXi elected(n_trees * max_leaf_size);
        VectorXi votes = VectorXi::Zero(n_samples);

        // count votes
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            int leaf_begin = leaf_first_indices[found_leaves[n_tree]];
            int leaf_end = leaf_first_indices[found_leaves[n_tree] + 1];
            const std::vector<int> &indices = tree_leaves[n_tree];
            for (int i = leaf_begin; i < leaf_end; ++i) {
                int idx = indices[i];
                if (++votes(idx) == votes_required)
                    elected(n_elected++) = idx;
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

        fwrite(&n_trees, sizeof(int), 1, fd);
        fwrite(&depth, sizeof(int), 1, fd);
        fwrite(&density, sizeof(float), 1, fd);

        fwrite(split_points.data(), sizeof(float), n_array * n_trees, fd);

        // save tree leaves
        for (int i = 0; i < n_trees; ++i) {
            int sz = tree_leaves[i].size();
            fwrite(&sz, sizeof(int), 1, fd);
            fwrite(&tree_leaves[i][0], sizeof(int), sz, fd);
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

        fread(&n_trees, sizeof(int), 1, fd);
        fread(&depth, sizeof(int), 1, fd);
        fread(&density, sizeof(float), 1, fd);

        n_pool = n_trees * depth;
        n_array = 1 << (depth + 1);

        count_first_leaf_indices();

        split_points = MatrixXf(n_array, n_trees);
        fread(split_points.data(), sizeof(float), n_array * n_trees, fd);

        // load tree leaves
        tree_leaves = std::vector<std::vector<int>>(n_trees);
        for (int i = 0; i < n_trees; ++i) {
            int sz;
            fread(&sz, sizeof(int), 1, fd);
            std::vector<int> leaves(sz);
            fread(&leaves[0], sizeof(int), sz, fd);
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
      int leaf_begin = leaf_first_indices[leaf];
      return tree_leaves[tree][leaf_begin + index];
    }

    /**
    * Accessor for the number of points in a leaf of a tree (for test purposes)
    * @param tree - index of tree in (0, ... T-1)
    * @param leaf - index of leaf in (0, ... , 2^depth)
    * @return - number of data points in leaf:th leaf of tree:th tree
    */
    int get_leaf_size(int tree, int leaf) const {
      return leaf_first_indices[leaf + 1] - leaf_first_indices[leaf];
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
    * @param begin - iterator to the index of the first data point of this branch
    * @param end - iterator to the index of the last data point of this branch
    * @param tree_level - The level in tree where the recursion is at
    * @param i - The index within the tree where we are at
    * @param n_tree - The index of the tree within the index
    * @param tree_projections - Precalculated projection values for the current tree
    */
    void grow_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
          int tree_level, int i, int n_tree, const MatrixXf &tree_projections) {
        int n = end - begin;
        int idx_left = 2 * i + 1;
        int idx_right = idx_left + 1;

        if (tree_level == depth) return;

        std::nth_element(begin, begin + n/2, end,
            [&tree_projections, tree_level] (int i1, int i2) {
              return tree_projections(tree_level, i1) < tree_projections(tree_level, i2);
            });
        auto mid = end - n/2;

        if(n % 2) {
          split_points(i, n_tree) = tree_projections(tree_level, *(mid - 1));
        } else {
          auto left_it = std::max_element(begin, mid,
              [&tree_projections, tree_level] (int i1, int i2) {
                return tree_projections(tree_level, i1) < tree_projections(tree_level, i2);
              });
          split_points(i, n_tree) = (tree_projections(tree_level, *mid) +
            tree_projections(tree_level, *left_it)) / 2.0;
        }

        grow_subtree(begin, mid, tree_level + 1, idx_left, n_tree, tree_projections);
        grow_subtree(mid, end, tree_level + 1, idx_right, n_tree, tree_projections);
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

    /**
    * Computes the leaf sizes of a tree assuming a median split and that
    * when the number points is odd, the extra point is always assigned to
    * to the left branch.
    * @param n - number data points
    * @param level - current level of the tree
    * @param tree_depth - depth of the whole tree
    * @param out_leaf_sizes - vector for the output; after completing
    * the function is a vector of length n containing the leaf sizes
    */
    void count_leaf_sizes(int n, int level, int tree_depth, std::vector<int> &out_leaf_sizes) {
      if(level == tree_depth) {
        out_leaf_sizes.push_back(n);
        return;
      }
      count_leaf_sizes(n - n/2, level + 1, tree_depth, out_leaf_sizes);
      count_leaf_sizes(n/2, level + 1, tree_depth, out_leaf_sizes);
    }

    /**
    * Computes indices of the first elements of leaves in a vector containing
    * all the leaves of a tree concatenated. Assumes that median split is used
    * and when the number points is odd, the extra point is always assigned to
    * to the left branch.
    */
    void count_first_leaf_indices() {
      std::vector<int> leaf_sizes;
      count_leaf_sizes(n_samples, 0, depth, leaf_sizes);

      leaf_first_indices = std::vector<int>(leaf_sizes.size() + 1);
      leaf_first_indices[0] = 0;
      for(int i = 0; i < leaf_sizes.size(); ++i)
        leaf_first_indices[i+1] = leaf_first_indices[i] + leaf_sizes[i];
    }


    const Map<const MatrixXf> *X; // the data matrix
    MatrixXf split_points; // all split points in all trees
    std::vector<std::vector<int>> tree_leaves; // contains all leaves of all trees
    Matrix<float, Dynamic, Dynamic, RowMajor> dense_random_matrix; // random vectors needed for all the RP-trees
    SparseMatrix<float, RowMajor> sparse_random_matrix; // random vectors needed for all the RP-trees
    std::vector<int> leaf_first_indices; // first indices of each leaf of tree in tree_leaves

    const int n_samples; // sample size of data
    const int dim; // dimension of data
    int n_trees; // number of RP-trees
    int depth; // depth of an RP-tree with median split
    float density; // expected ratio of non-zero components in a projection matrix
    int n_pool; // amount of random vectors needed for all the RP-trees
    int n_array; // length of the one RP-tree as array
};

#endif // CPP_MRPT_H_

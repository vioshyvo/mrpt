#ifndef CPP_MRPT_H_
#define CPP_MRPT_H_

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <utility>
#include <map>
#include <set>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

using namespace Eigen;

struct Parameters {
  int n_trees = 0;
  int depth = 0;
  int votes = 0;
  double estimated_qtime = 0.0;
  double estimated_recall = 0.0;
  double validation_qtime = -1.0;
  double validation_recall = -1.0;
  double validation_qtime_sd = -1.0;
  double validation_recall_sd = -1.0;
};

class Mrpt {
 public:
    /**
    * The constructor of the index. The constructor does not actually build
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

        density < 1 ? build_sparse_random_matrix(sparse_random_matrix, n_pool, dim, density, seed) : build_dense_random_matrix(dense_random_matrix, n_pool, dim, seed);

        split_points = MatrixXf(n_array, n_trees);
        tree_leaves = std::vector<std::vector<int>>(n_trees);

        count_first_leaf_indices_all(leaf_first_indices_all, n_samples, depth);

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
    * @param out_n_elected - An optional output parameter for the candidate set size
    * @return
    */
    void query(const Map<VectorXf> &q, int k, int votes_required, int *out,
       float *out_distances = nullptr, int *out_n_elected = nullptr) const {
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

        if(out_n_elected) *out_n_elected += n_elected;
        exact_knn(q, k, elected, n_elected, out, out_distances);
    }

    /**
    * This function finds the k approximate nearest neighbors of the query object
    * q. The accuracy of the query depends on both the parameters used for index
    * construction and additional parameters given to this function. This
    * function implements two tricks to improve performance. The voting trick
    * interprets each index object in leaves returned by tree traversals as votes,
    * and only performs the final linear search with the 'elect' most voted
    * objects.
    *
    * This is a version that can utilize a larger index by performing a query only
    * on first n_trees_crnt trees that are pruned to depth depth_crnt.
    *
    * @param q - The query object whose neighbors the function finds
    * @param k - The number of neighbors the user wants the function to return
    * @param votes_required - The number of votes required for an object to be included in the linear search step
    * @param out - The output buffer for the indices of the k approximate nearest neighbors
    * @param out_distances - Output buffer for distances of the k approximate nearest neighbors (optional parameter)
    * @param out_n_elected - An optional output parameter for the candidate set size
    * @param n_trees_crnt - Number of trees to use for this query
    * @param depth_crnt - Depth of trees used for this query
    * @return
    */
    void query(const Map<VectorXf> &q, int k, int votes_required, int *out, int n_trees_crnt, int depth_crnt,
          float *out_distances = nullptr, int *out_n_elected = nullptr) const {

        std::vector<int> found_leaves(n_trees_crnt);
        const std::vector<int> &leaf_first_indices = leaf_first_indices_all[depth_crnt];

        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees_crnt; ++n_tree) {
          VectorXf projected_query(depth_crnt);

          if (density < 1)
              projected_query.noalias() = sparse_random_matrix.middleRows(n_tree * depth, depth_crnt) * q;
          else
              projected_query.noalias() = dense_random_matrix.middleRows(n_tree * depth, depth_crnt) * q;

          int idx_tree = 0;
          for (int d = 0; d < depth_crnt; ++d) {
              const int idx_left = 2 * idx_tree + 1;
              const int idx_right = idx_left + 1;
              const float split_point = split_points(idx_tree, n_tree);
              if (projected_query(d) <= split_point) {
                  idx_tree = idx_left;
              } else {
                  idx_tree = idx_right;
              }
          }
          found_leaves[n_tree] = idx_tree - (1 << depth_crnt) + 1;
        }

        int n_elected = 0, max_leaf_size = n_samples / (1 << depth_crnt) + 1;
        VectorXi elected(n_trees * max_leaf_size);
        VectorXi votes = VectorXi::Zero(n_samples);

        // count votes
        for (int n_tree = 0; n_tree < n_trees_crnt; ++n_tree) {
            int leaf_begin = leaf_first_indices[found_leaves[n_tree]];
            int leaf_end = leaf_first_indices[found_leaves[n_tree] + 1];
            const std::vector<int> &indices = tree_leaves[n_tree];
            for (int i = leaf_begin; i < leaf_end; ++i) {
                int idx = indices[i];
                if (++votes(idx) == votes_required)
                    elected(n_elected++) = idx;
            }
        }

        if(out_n_elected) *out_n_elected += n_elected;
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

        if(!n_elected) {
          for(int i = 0; i < k; ++i) out[i] = -1;
          if(out_distances) {
            for(int i = 0; i < k; ++i) out_distances[i] = -1;
          }
          return;
        }

        VectorXf distances(n_elected);

        #pragma omp parallel for
        for (int i = 0; i < n_elected; ++i)
            distances(i) = (X->col(indices(i)) - q).squaredNorm();

        if (k == 1) {
            MatrixXf::Index index;
            distances.minCoeff(&index);
            out[0] = n_elected ? indices(index) : -1;

            if(out_distances)
              out_distances[0] = n_elected ? std::sqrt(distances(index)) : -1;

            return;
        }

        int n_to_sort = n_elected > k ? k : n_elected;
        VectorXi idx(n_elected);
        std::iota(idx.data(), idx.data() + n_elected, 0);
        std::partial_sort(idx.data(), idx.data() + n_to_sort, idx.data() + n_elected,
                         [&distances](int i1, int i2) {return distances(i1) < distances(i2);});

        for (int i = 0; i < k; ++i)
          out[i] = i < n_elected ? indices(idx(i)) : -1;

        if(out_distances) {
          for(int i = 0; i < k; ++i)
            out_distances[i] = i < n_elected ? std::sqrt(distances(idx(i))) : -1;
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

        count_first_leaf_indices_all(leaf_first_indices_all, n_samples, depth);
        leaf_first_indices = leaf_first_indices_all[depth];

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

    void project_query(const Map<VectorXf> &q, VectorXf &projected_query) {
      if (density < 1)
          projected_query.noalias() = sparse_random_matrix * q;
      else
          projected_query.noalias() = dense_random_matrix * q;
    }

    void vote(const VectorXf &projected_query, int votes_required, VectorXi &elected, int &n_elected, int n_trees) {
      std::vector<int> found_leaves(n_trees);
      const std::vector<int> &leaf_first_indices = leaf_first_indices_all[depth];


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

      int max_leaf_size = n_samples / (1 << depth) + 1;
      elected = VectorXi(n_trees * max_leaf_size);
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
      const std::vector<int> &leaf_first_indices = leaf_first_indices_all[depth];
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
      const std::vector<int> &leaf_first_indices = leaf_first_indices_all[depth];
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

    /**
    * @return - dimension of the data set from which the index is built
    */
    int get_dim() const {
      return dim;
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
    static void count_leaf_sizes(int n, int level, int tree_depth, std::vector<int> &out_leaf_sizes) {
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
    static void count_first_leaf_indices(std::vector<int> &indices, int n, int depth) {
      std::vector<int> leaf_sizes;
      count_leaf_sizes(n, 0, depth, leaf_sizes);

      indices = std::vector<int>(leaf_sizes.size() + 1);
      indices[0] = 0;
      for(int i = 0; i < leaf_sizes.size(); ++i)
        indices[i+1] = indices[i] + leaf_sizes[i];
    }

    static void count_first_leaf_indices_all(std::vector<std::vector<int>> &indices, int n, int depth_max) {
      for(int d = 0; d <= depth_max; ++d) {
        std::vector<int> idx;
        count_first_leaf_indices(idx, n, d);
        indices.push_back(idx);
      }
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
    static void build_sparse_random_matrix(SparseMatrix<float, RowMajor> &sparse_random_matrix,
          int n_row, int n_col, float density, int seed = 0) {
        sparse_random_matrix = SparseMatrix<float, RowMajor>(n_row, n_col);

        std::random_device rd;
        int s = seed ? seed : rd();
        std::mt19937 gen(s);
        std::uniform_real_distribution<float> uni_dist(0, 1);
        std::normal_distribution<float> norm_dist(0, 1);

        std::vector<Triplet<float>> triplets;
        for (int j = 0; j < n_row; ++j) {
            for (int i = 0; i < n_col; ++i) {
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
    static void build_dense_random_matrix(Matrix<float, Dynamic, Dynamic, RowMajor> &dense_random_matrix,
          int n_row, int n_col, int seed = 0) {
        dense_random_matrix = Matrix<float, Dynamic, Dynamic, RowMajor>(n_row, n_col);

        std::random_device rd;
        int s = seed ? seed : rd();
        std::mt19937 gen(s);
        std::normal_distribution<float> normal_dist(0, 1);

        std::generate(dense_random_matrix.data(), dense_random_matrix.data() + n_row * n_col,
                      [&normal_dist, &gen] { return normal_dist(gen); });
    }



    friend class Autotuning;

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


    void count_elected(const Map<VectorXf> &q, const Map<VectorXi> &exact, int votes_max,
      std::vector<MatrixXd> &recalls, std::vector<MatrixXd> &cs_sizes) const {
        VectorXf projected_query(n_pool);
        if (density < 1)
            projected_query.noalias() = sparse_random_matrix * q;
        else
            projected_query.noalias() = dense_random_matrix * q;

        int depth_min = depth - recalls.size() + 1;
        std::vector<std::vector<int>> start_indices(n_trees);

        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            start_indices[n_tree] = std::vector<int>(depth - depth_min + 1);
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
                if(d >= depth_min - 1)
                  start_indices[n_tree][d - depth_min + 1] = idx_tree - (1 << (d + 1)) + 1;
            }
        }

        const int *exact_begin = exact.data();
        const int *exact_end = exact.data() + exact.size();

        for(int depth_crnt = depth_min; depth_crnt <= depth; ++depth_crnt) {
          VectorXi votes = VectorXi::Zero(n_samples);
          const std::vector<int> &leaf_first_indices = leaf_first_indices_all[depth_crnt];

          MatrixXd recall(votes_max, n_trees);
          MatrixXd candidate_set_size(votes_max, n_trees);
          recall.col(0) = VectorXd::Zero(votes_max);
          candidate_set_size.col(0) = VectorXd::Zero(votes_max);

          // count votes
          for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
              std::vector<int> &found_leaves = start_indices[n_tree];

              if(n_tree) {
                recall.col(n_tree) = recall.col(n_tree - 1);
                candidate_set_size.col(n_tree) = candidate_set_size.col(n_tree - 1);
              }

              int leaf_begin = leaf_first_indices[found_leaves[depth_crnt - depth_min]];
              int leaf_end = leaf_first_indices[found_leaves[depth_crnt - depth_min] + 1];

              const std::vector<int> &indices = tree_leaves[n_tree];
              for (int i = leaf_begin; i < leaf_end; ++i) {
                  int idx = indices[i];
                  int v = ++votes(idx);
                  if (v <= votes_max) {
                    candidate_set_size(v-1, n_tree)++;
                    if(std::find(exact_begin, exact_end, idx) != exact_end) // is there a faster way to find from the sorted range?
                      recall(v-1, n_tree)++;
                  }
              }
           }
           recalls[depth_crnt - depth_min] = recall;
           cs_sizes[depth_crnt - depth_min] = candidate_set_size;
        }

    }



    const Map<const MatrixXf> *X; // the data matrix
    MatrixXf split_points; // all split points in all trees
    std::vector<std::vector<int>> tree_leaves; // contains all leaves of all trees
    Matrix<float, Dynamic, Dynamic, RowMajor> dense_random_matrix; // random vectors needed for all the RP-trees
    SparseMatrix<float, RowMajor> sparse_random_matrix; // random vectors needed for all the RP-trees
    // std::vector<int> leaf_first_indices; // first indices of each leaf of tree in tree_leaves
    std::vector<std::vector<int>> leaf_first_indices_all; // first indices for each level
    std::vector<int> leaf_first_indices;

    const int n_samples; // sample size of data
    const int dim; // dimension of data
    int n_trees = 0; // number of RP-trees
    int depth = 0; // depth of an RP-tree with median split
    float density = 1; // expected ratio of non-zero components in a projection matrix
    int n_pool = 0; // amount of random vectors needed for all the RP-trees
    int n_array = 0; // length of the one RP-tree as array
    int votes = 0;
};

class Autotuning {
  public:
    Autotuning(const Map<const MatrixXf> *X_, Map<MatrixXf> *Q_) :
      X(X_),
      Q(Q_),
      recall_level(-1) {}

    ~Autotuning() {}

    Mrpt tune(int trees_max_, int depth_min_, int depth_max_, int votes_max_,
       float density_, int k_, int seed_mrpt_ = 0) {
      trees_max = trees_max_;
      depth_min = depth_min_;
      depth_max = depth_max_;
      votes_max = votes_max_;
      k = k_;
      seed_mrpt = seed_mrpt_;
      density = density_;
      int n_test = Q->cols();
      int d = X->rows();

      double at_start = omp_get_wtime();
      recalls = std::vector<MatrixXd>(depth_max - depth_min + 1);
      cs_sizes = std::vector<MatrixXd>(depth_max - depth_min + 1);

      for(int depth = depth_min; depth <= depth_max; ++depth) {
        recalls[depth - depth_min] = MatrixXd::Zero(votes_max, trees_max);
        cs_sizes[depth - depth_min] = MatrixXd::Zero(votes_max, trees_max);
      }

      Mrpt index(X);
      index.grow(trees_max, depth_max, density, seed_mrpt);

      MatrixXi exact(k, n_test);
      compute_exact(index, exact);

      for(int i = 0; i < n_test; ++i) {
        std::vector<MatrixXd> recall_tmp(depth_max - depth_min + 1);
        std::vector<MatrixXd> cs_size_tmp(depth_max - depth_min + 1);

        index.count_elected(Map<VectorXf>(Q->data() + i * d, d), Map<VectorXi>(exact.data() + i * k, k),
         votes_max, recall_tmp, cs_size_tmp);

        for(int depth = depth_min; depth <= depth_max; ++depth) {
          recalls[depth - depth_min] += recall_tmp[depth - depth_min];
          cs_sizes[depth - depth_min] += cs_size_tmp[depth - depth_min];
        }
      }

      for(int depth = depth_min; depth <= depth_max; ++depth) {
        recalls[depth - depth_min] /= (k * n_test);
        cs_sizes[depth - depth_min] /= n_test;
      }

      fit_times(index);

      std::vector<int> target_recalls(99);
      std::iota(target_recalls.begin(), target_recalls.end(), 1);
      find_optimal_parameters(target_recalls);

      double at_end = omp_get_wtime();
      autotuning_time = at_end - at_start;

      measure_query_times(index, exact);
      return index;
    }

    float get_recall(int tree, int depth, int v) {
      return recalls[depth - depth_min](v - 1, tree - 1);
    }

    float get_candidate_set_size(int tree, int depth, int v) {
      return cs_sizes[depth - depth_min](v - 1, tree - 1);
    }

    double get_projection_time(int n_trees, int depth, int v) {
      return predict_theil_sen(n_trees * depth, beta_projection);
    }

    double get_voting_time(int n_trees, int depth, int v) {
      return predict_theil_sen(n_trees, beta_voting);
    }

    double get_exact_time(int n_trees, int depth, int v) {
      return predict_theil_sen(get_candidate_set_size(n_trees, depth, v), beta_exact);
    }

    double get_query_time(int tree, int depth, int v) {
      return get_projection_time(tree, depth, v)
           + get_voting_time(tree, depth, v)
           + get_exact_time(tree, depth, v);
    }

    Parameters get_optimal_parameters(int target_recall) {
      if(optimal_parameter_table.count(target_recall) == 0) {
        Parameters par;
        return par;
      }

      return optimal_parameter_table[target_recall];
    }

    static std::pair<double,double> fit_theil_sen(const std::vector<double> &x,
        const std::vector<double> &y) {
      int n = x.size();
      std::vector<double> slopes;
      for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
          if(i != j)
            slopes.push_back((y[j] - y[i]) / (x[j] - x[i]));

      int n_slopes = slopes.size();
      std::nth_element(slopes.begin(), slopes.begin() + n_slopes / 2, slopes.end());
      double slope = *(slopes.begin() + n_slopes / 2);

      std::vector<double> residuals(n);
      for(int i = 0; i < n; ++i)
        residuals[i] = y[i] - slope * x[i];

      std::nth_element(residuals.begin(), residuals.begin() + n / 2, residuals.end());
      double intercept = *(residuals.begin() + n / 2);

      return std::make_pair(intercept, slope);
    }

    static double predict_theil_sen(double x, std::pair<double,double> beta) {
      return beta.first + beta.second * x;
    }

    void query(const Map<VectorXf> &q, int target_recall, int *out, Mrpt &index,
        float *out_distances = nullptr, int *out_n_elected = nullptr) {

      if(recall_level != target_recall) {
        recall_level = target_recall;

        if(optimal_parameter_table.count(target_recall) == 0) {
          // std::cerr << "Target recall level " << target_recall << " too high." << std::endl;
          recall_level = -1;
          return;
        }

        optimal_parameters = optimal_parameter_table[target_recall];

      }

      index.query(q, k, optimal_parameters.votes, out, optimal_parameters.n_trees,
         optimal_parameters.depth, out_distances, out_n_elected);
    }

    void query(const Map<VectorXf> &q, int *out, Mrpt &index,
        float *out_distances = nullptr, int *out_n_elected = nullptr) {
        if(recall_level < 0) {
          std::cerr << "Recall level not set. Returning..." << std::endl;
          return;
        }
        index.query(q, k, optimal_parameters.votes, out, out_distances, out_n_elected);
    }

    void find_optimal_parameters(const std::vector<int> &target_recalls) {
      optimal_parameter_table.clear();

      for(int depth = depth_min; depth <= depth_max; ++depth) {
        for(int t = 1; t <= trees_max; ++t) {
          int votes_index = votes_max < t ? votes_max : t;
          for(int v = 1; v <= votes_index; ++v) {
            double rec = get_recall(t, depth, v) * 100.0;
            double qt = get_query_time(t, depth, v);
            for(int i = 0; i < target_recalls.size(); ++i) {
              int tr = target_recalls[i];
              if(rec >= tr) {
                if(optimal_parameter_table.count(tr) == 0 || qt < optimal_parameter_table[tr].estimated_qtime) {
                  Parameters par {t, depth, v, qt, rec / 100.0};
                  optimal_parameter_table[tr] = par;
                }
              }
            }
          }
        }
      }
    }

  static double mean(const std::vector<double> &x) {
    int n = x.size();
    double xsum = 0;
    for(int i = 0; i < n; ++i)
      xsum += x[i];
    return xsum / n;
  }

  static double var(const std::vector<double> &x) {
    int n = x.size();
    double xmean = mean(x);
    double ssr = 0;
    for(int i = 0; i < n; ++i)
      ssr += (x[i] - xmean) * (x[i] - xmean);
    return ssr / (n - 1);
  }

  void write_results(std::string filename, bool add) {
    std::ofstream outf;
    if(add) {
      outf.open(filename, std::ios::app);
    } else {
      outf.open(filename);
    }

    if(!outf) {
      std::cerr << "File " << filename << " could not be opened for writing." << std::endl;
      return;
    }

    int n_test = Q->cols();
    for(auto it = optimal_parameter_table.begin(); it != optimal_parameter_table.end(); ++it) {
      Parameters &par = it->second;
      outf << k << " "
           << par.n_trees << " "
           << par.depth << " "
           << density << " "
           << par.votes << " "
           << par.validation_recall << " "
           << par.validation_recall_sd << " "
           << par.validation_qtime * static_cast<double>(n_test) << " "
           << par.validation_qtime_sd << " "
           << autotuning_time << std::endl;
    }
  }

  void delete_extra_trees(int target_recall, Mrpt &index) {
    recall_level = target_recall;
    optimal_parameters = get_optimal_parameters(target_recall);
    if(!optimal_parameters.n_trees) {
      std::cerr << "Recall level " << target_recall << " too high. Returning..." << std::endl;
      return;
    }

    index.n_trees = optimal_parameters.n_trees;
    index.depth = optimal_parameters.depth;
    index.votes = optimal_parameters.votes;
    index.n_pool = index.depth * index.n_trees;
    index.n_array = 1 << (index.depth + 1);

    index.tree_leaves.resize(index.n_trees);
    index.split_points.conservativeResize(index.n_array, index.n_trees);
    index.leaf_first_indices = index.leaf_first_indices_all[index.depth];
    if(index.density < 1) {
      SparseMatrix<float, RowMajor> srm_new(index.n_pool, index.dim);
      for(int n_tree = 0; n_tree < index.n_trees; ++n_tree)
        srm_new.middleRows(n_tree * index.depth, index.depth) = index.sparse_random_matrix.middleRows(n_tree * depth_max, index.depth);
      index.sparse_random_matrix = srm_new;
    } else {
      Matrix<float, Dynamic, Dynamic, RowMajor> drm_new(index.n_pool, index.dim);
      for(int n_tree = 0; n_tree < index.n_trees; ++n_tree)
        drm_new.middleRows(n_tree * index.depth, index.depth) = index.dense_random_matrix.middleRows(n_tree * depth_max, index.depth);
      index.dense_random_matrix = drm_new;
    }
  }

  void subset_trees(int target_recall, const Mrpt &index, Mrpt &index2) {
    recall_level = target_recall;
    optimal_parameters = get_optimal_parameters(target_recall);
    if(!optimal_parameters.n_trees) {
      std::cerr << "Recall level " << target_recall << " too high. Returning..." << std::endl;
      return;
    }

    index2.n_trees = optimal_parameters.n_trees;
    index2.depth = optimal_parameters.depth;
    index2.votes = optimal_parameters.votes;
    index2.n_pool = index2.depth * index2.n_trees;
    index2.n_array = 1 << (index2.depth + 1);
    index2.tree_leaves.assign(index.tree_leaves.begin(), index.tree_leaves.begin() + index2.n_trees);
    index2.leaf_first_indices_all = index.leaf_first_indices_all;
    index2.density = index.density;

    index2.split_points = index.split_points.topLeftCorner(index2.n_array, index2.n_trees);
    index2.leaf_first_indices = index.leaf_first_indices_all[index2.depth];
    if(index2.density < 1) {
      index2.sparse_random_matrix = SparseMatrix<float, RowMajor>(index2.n_pool, index2.dim);
      for(int n_tree = 0; n_tree < index2.n_trees; ++n_tree)
        index2.sparse_random_matrix.middleRows(n_tree * index2.depth, index2.depth) = index.sparse_random_matrix.middleRows(n_tree * depth_max, index2.depth);
    } else {
      index2.dense_random_matrix = Matrix<float, Dynamic, Dynamic, RowMajor>(index2.n_pool, index2.dim);
      for(int n_tree = 0; n_tree < index2.n_trees; ++n_tree)
        index2.dense_random_matrix.middleRows(n_tree * index2.depth, index2.depth) = index.dense_random_matrix.middleRows(n_tree * depth_max, index2.depth);
      }
  }


  private:

    void measure_query_times(Mrpt &index, MatrixXi &exact) {
      for(auto it = optimal_parameter_table.begin(); it != optimal_parameter_table.end(); ++it) {
        Parameters &par = it->second;
        int n_test = Q->cols();
        int d = Q->rows();

        std::vector<double> qtimes, recalls;
        for(int i = 0; i < n_test; ++i) {
          std::vector<int> result(k);
          const Map<VectorXf> q(Q->data() + i * d, d);
          double start = omp_get_wtime();
          index.query(q, k, par.votes, &result[0], par.n_trees, par.depth);
          double end = omp_get_wtime();
          qtimes.push_back(end - start);

          std::sort(result.begin(), result.end());
          std::set<int> intersect;
          std::set_intersection(exact.data() + i * k, exact.data() + i * k + k, result.begin(), result.end(),
                           std::inserter(intersect, intersect.begin()));

          double rec = intersect.size() / static_cast<double>(k);
          recalls.push_back(rec);
        }

        par.validation_qtime = mean(qtimes);
        par.validation_recall = mean(recalls);
        par.validation_qtime_sd = std::sqrt(var(qtimes));
        par.validation_recall_sd = std::sqrt(var(recalls));
      }
    }

    void compute_exact(Mrpt &index, MatrixXi &out_exact) {
      int k = out_exact.rows();
      int nt = out_exact.cols();
      int n = index.get_n_points();
      int d = index.get_dim();
      for(int i = 0; i < nt; ++i) {
        VectorXi idx(n);
        std::iota(idx.data(), idx.data() + n, 0);

        index.exact_knn(Map<VectorXf>(Q->data() + i * d, d), k, idx, n, out_exact.data() + i * k);
        std::sort(out_exact.data() + i * k, out_exact.data() + i * k + k);
      }
    }

    void fit_times(Mrpt &index) {
      int n_test = Q->cols();
      int d = Q->rows();
      int n = X->cols();
      std::vector<double> projection_times, projection_x;
      std::vector<int> tested_trees;
      std::vector<double> exact_times;
      std::vector<int> exact_x;

      float idx_sum = 0;

      std::random_device rd;
      std::mt19937 rng(rd());
      std::uniform_int_distribution<int> uni(0, n_test-1);
      std::uniform_int_distribution<int> uni2(0, n-1);

      int n_tested_trees = 5;
      n_tested_trees = trees_max > n_tested_trees ? n_tested_trees : trees_max;
      int incr = trees_max / n_tested_trees;
      for(int i = 1; i <= n_tested_trees; ++i)
        tested_trees.push_back(i * incr);

      for(int depth = depth_min; depth <= depth_max; ++depth) {
        for(int i = 0; i < tested_trees.size(); ++i) {
          int t = tested_trees[i];
          int n_pool = t * depth;
          projection_x.push_back(n_pool);
          SparseMatrix<float, RowMajor> sparse_random_matrix;
          Mrpt::build_sparse_random_matrix(sparse_random_matrix, n_pool, d, density);

          double start_proj = omp_get_wtime();
          VectorXf projected_query(n_pool);
          projected_query.noalias() = sparse_random_matrix * Q->col(0);
          double end_proj = omp_get_wtime();
          projection_times.push_back(end_proj - start_proj);
          idx_sum += projected_query.norm();

          int votes_index = votes_max < t ? votes_max : t;
          for(int v = 1; v <= votes_index; ++v) {
            int cs_size = get_candidate_set_size(t, depth, v);
            if(cs_size > 0) exact_x.push_back(cs_size);
          }
        }
      }

      // int s_min = *std::min_element(exact_x.begin(), exact_x.end());
      int s_max = *std::max_element(exact_x.begin(), exact_x.end());

      int n_s_tested = 20;
      std::vector<int> s_tested;
      std::vector<double> ex;
      int increment = s_max / n_s_tested;
      for(int i = 1; i <= n_s_tested; ++i)
        s_tested.push_back(i * increment);

      int v = 3;
      std::vector<double> voting_times, voting_x;

      for(int i = 0; i < tested_trees.size(); ++i) {
        int t = tested_trees[i];
        int n_el = 0;
        VectorXi elected;
        auto ri = uni(rng);
        VectorXf projected_query = Q->col(ri);

        double start_voting = omp_get_wtime();
        index.vote(projected_query, v, elected, n_el, t);
        double end_voting = omp_get_wtime();

        voting_times.push_back(end_voting - start_voting);
        voting_x.push_back(t);
        for(int i = 0; i < n_el; ++i)
          idx_sum += elected(i);
      }

      for(int i = 0; i < n_s_tested; ++i) {
        auto ri = uni(rng);
        int s_size = s_tested[i];
        ex.push_back(s_size);
        VectorXi elected(s_size);
        for(int j = 0; j < elected.size(); ++j)
          elected(j) = uni2(rng);

        double start_exact = omp_get_wtime();
        std::vector<int> res(k);
        index.exact_knn(Map<VectorXf>(Q->data() + ri * d, d), k, elected, s_size, &res[0]);
        double end_exact = omp_get_wtime();
        exact_times.push_back(end_exact - start_exact);
        for(int l = 0; l < k; ++l)
          idx_sum += res[l];
      }

      beta_projection = fit_theil_sen(projection_x, projection_times);
      beta_voting = fit_theil_sen(voting_x, voting_times);
      beta_exact = fit_theil_sen(ex, exact_times);

      // std::cout << std::endl;
      // std::cout << "idx_sum: " << idx_sum << "\n";
      // std::cout << "projection, intercept: " << beta_projection.first << " slope: " << beta_projection.second << "\n";
      // std::cout << "voting, intercept: " << beta_voting.first << " slope: " << beta_voting.second << "\n";
      // std::cout << "exact, intercept: " << beta_exact.first << " slope: " << beta_exact.second << "\n\n";

      query_times = std::vector<MatrixXd>(depth_max - depth_min + 1);
      for(int depth = depth_min; depth <= depth_max; ++depth) {
        MatrixXd query_time = MatrixXd::Zero(votes_max, trees_max);

        for(int t = 1; t <= trees_max; ++t) {
          int votes_index = votes_max < t ? votes_max : t;
          for(int v = 1; v <= votes_index; ++v) {
            query_time(v - 1, t - 1) = get_query_time(t, depth, v);
          }
        }
        query_times[depth - depth_min] = query_time;
      }
    }

    const Map<const MatrixXf> *X;
    Map<MatrixXf> *Q;
    std::vector<MatrixXd> recalls, cs_sizes, query_times;
    int trees_max, depth_min, depth_max, votes_max, k, seed_mrpt;
    float density;
    std::pair<double,double> beta_projection, beta_voting, beta_exact;
    int recall_level;
    Parameters optimal_parameters;
    std::map<int,Parameters> optimal_parameter_table;
    double autotuning_time = -1.0;
};

#endif // CPP_MRPT_H_

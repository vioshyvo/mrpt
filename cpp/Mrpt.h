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

#include <Eigen/Dense>
#include <Eigen/SparseCore>

using namespace Eigen;

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

    void project_query(const Map<VectorXf> &q, VectorXf &projected_query) {
      if (density < 1)
          projected_query.noalias() = sparse_random_matrix * q;
      else
          projected_query.noalias() = dense_random_matrix * q;
    }

    void vote(const VectorXf &projected_query, int votes_required, VectorXi &elected, int &n_elected, int n_trees) {
      std::vector<int> found_leaves(n_trees);

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

    /**
    * @return - dimension of the data set from which the index is built
    */
    int get_dim() const {
      return dim;
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

    void count_elected(const Map<VectorXf> &q, const Map<VectorXi> &exact, int votes_max,
      std::vector<MatrixXd> &recalls, std::vector<MatrixXd> &cs_sizes) const {
        VectorXf projected_query(n_pool);
        if (density < 1)
            projected_query.noalias() = sparse_random_matrix * q;
        else
            projected_query.noalias() = dense_random_matrix * q;

        std::vector<int> found_leaves(n_trees);

        int depth_min = depth - recalls.size() + 1;
        std::vector<std::vector<int>> start_indices(n_trees);

        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            start_indices[n_tree] = std::vector<int>(depth - depth_min);
            int idx_tree = 0;
            for (int d = 0; d < depth; ++d) {
                if(d >= depth_min) {
                  int diff = depth - d;
                  start_indices[n_tree][d - depth_min] = (1 << diff) * (idx_tree + 1) - 1 - (1 << depth) + 1;
                }

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

        const int *exact_begin = exact.data();
        const int *exact_end = exact.data() + exact.size();

        for(int depth_crnt = depth_min; depth_crnt <= depth; ++depth_crnt) {
          // std::cout << "depth: " << depth_crnt << std::endl;
          VectorXi votes = VectorXi::Zero(n_samples);

          MatrixXd recall(votes_max, n_trees);
          MatrixXd candidate_set_size(votes_max, n_trees);
          recall.col(0) = VectorXd::Zero(votes_max);
          candidate_set_size.col(0) = VectorXd::Zero(votes_max);

          // count votes
          for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            // std::cout << "tree: " << n_tree << std::endl;
              if(n_tree) {
                recall.col(n_tree) = recall.col(n_tree - 1);
                candidate_set_size.col(n_tree) = candidate_set_size.col(n_tree - 1);
              }

              int diff = depth - depth_crnt;
              int leaf_begin = diff ? leaf_first_indices[start_indices[n_tree][depth_crnt - depth_min]] : leaf_first_indices[found_leaves[n_tree]];
              int leaf_end = diff ? leaf_first_indices[start_indices[n_tree][depth_crnt - depth_min] + (1 << diff)] : leaf_first_indices[found_leaves[n_tree] + 1];
              // std::cout << "leaf_begin: " << leaf_begin << std::endl;
              // std::cout << "leaf_end: " << leaf_end << std::endl;
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
    std::vector<int> leaf_first_indices; // first indices of each leaf of tree in tree_leaves

    const int n_samples; // sample size of data
    const int dim; // dimension of data
    int n_trees; // number of RP-trees
    int depth; // depth of an RP-tree with median split
    float density; // expected ratio of non-zero components in a projection matrix
    int n_pool; // amount of random vectors needed for all the RP-trees
    int n_array; // length of the one RP-tree as array
};

class Autotuning {
  public:
    Autotuning(const Map<const MatrixXf> *X_, Map<MatrixXf> *Q_) :
      X(X_),
      Q(Q_) {}

    ~Autotuning() {}

    void tune(int trees_max_, int depth_min_, int depth_max_, int votes_max_,
       float density_, int k_, int seed_mrpt_ = 0) {
      trees_max = trees_max_;
      depth_min = depth_min_;
      depth_max = depth_max_;
      votes_max = votes_max_;
      k = k_;
      seed_mrpt = seed_mrpt;
      density = density_;
      int n_test = Q->cols();
      int d = X->rows();

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
        // std::cout << "point: " << i << std::endl;
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
        // std::cout << "Autotuning recall, depth = " << depth << "\n" << recalls[depth - depth_min] << "\n\n";
      }

      fit_times(index);
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

    double get_exact_time(int tree, int depth, int v) {
      return exact_times[depth - depth_min](v - 1, tree - 1);
    }

    double get_query_time(int tree, int depth, int v) {
      return get_projection_time(tree, depth, v)
           + get_voting_time(tree, depth, v)
           + get_exact_time(tree, depth, v);
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

  private:

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
      exact_times = std::vector<MatrixXd>(depth_max - depth_min + 1);

      int n_test = Q->cols();
      int d = Q->rows();
      std::vector<double> projection_times, projection_x;
      float idx_sum = 0;

      for(int depth = depth_min; depth <= depth_max; ++depth) {
        MatrixXd exact_time = MatrixXd::Zero(votes_max, trees_max);

        for(int t = 1; t <= trees_max; ++t) {
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

          Mrpt index2(X);
          index.grow(t, depth, density, seed_mrpt);

          int votes_index = votes_max < t ? votes_max : t;
          for(int v = 1; v <= votes_index; ++v) {
            for(int i = 0; i < n_test; ++i) {

              // double start_exact = omp_get_wtime();
              // std::vector<int> res(k);
              // index2.exact_knn(Map<VectorXf>(Q->data() + i * d, d), k, elected, n_el, &res[0]);
              // double end_exact = omp_get_wtime();
              exact_time(v - 1, t - 1) += 0.0;

              // exact_time(v - 1, t - 1) += end_exact - start_exact;
            }
          }
        }

        exact_time /= n_test;
        exact_times[depth - depth_min] = exact_time;
      }

      int v = 3;

      std::vector<double> voting_times, voting_x(votes_max);
      std::iota(voting_x.begin(), voting_x.end(), 1);

      for(int t = 1; t <= trees_max; ++t) {
        int n_el = 0;
        VectorXi elected;
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni(0, n_test-1);
        auto ri = uni(rng);
        VectorXf projected_query = Q->col(ri);

        double start_voting = omp_get_wtime();
        index.vote(projected_query, v, elected, n_el, t);
        double end_voting = omp_get_wtime();

        voting_times.push_back(end_voting - start_voting);
        for(int i = 0; i < n_el; ++i)
          idx_sum += elected(i);
      }

      std::cout << "idx_sum: " << idx_sum << "\n";
      beta_projection = fit_theil_sen(projection_x, projection_times);
      beta_voting = fit_theil_sen(voting_x, voting_times);

    }

    const Map<const MatrixXf> *X;
    Map<MatrixXf> *Q;
    std::vector<MatrixXd> recalls, cs_sizes;
    std::vector<MatrixXd> voting_times, exact_times;
    int trees_max, depth_min, depth_max, votes_max, k, seed_mrpt;
    float density;
    std::pair<double,double> beta_projection, beta_voting, beta_exact;
};

#endif // CPP_MRPT_H_

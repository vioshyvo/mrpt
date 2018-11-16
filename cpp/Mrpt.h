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
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/SparseCore>


struct Mrpt_Parameters {
  int n_trees = 0;
  int depth = 0;
  int votes = 0;
  int k = 0;
  double estimated_qtime = 0.0;
  mutable double estimated_recall = 0.0;
};

class Mrpt {
 public:
    /**
    * The constructor of the index. The constructor does not actually build
    * the index but that is done by the function 'grow' which has to be called
    * before queries can be made.
    * @param X_ - Pointer to the Eigen::Map which refers to the data matrix.
    */
    Mrpt(const Eigen::Map<const Eigen::MatrixXf> &X_) :
        X(X_),
        n_samples(X_.cols()),
        dim(X_.rows()) {}

    Mrpt(const Eigen::MatrixXf &X_) :
        X(Eigen::Map<const Eigen::MatrixXf>(X_.data(), X_.rows(), X_.cols())),
        n_samples(X_.cols()),
        dim(X_.rows()) {}

    Mrpt(const float *data, int dim_, int n_samples_) :
        X(Eigen::Map<const Eigen::MatrixXf>(data, dim_, n_samples_)),
        n_samples(n_samples_),
        dim(dim_) {}

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
    void grow(int n_trees_, int depth_, float density_ = -1.0, int seed = 0) {

        if(n_trees_ <= 0) {
          throw std::out_of_range("The number of trees must be positive.");
        }

        if(depth_ <= 0 || depth_ > std::log2(n_samples)) {
          throw std::out_of_range("The depth must belong to the set {1, ... , log2(n)}.");
        }

        if(density_ < -1.0001 || density_ > 1.0001 || (density_ > -0.9999 && density_ < -0.0001)) {
          throw std::out_of_range("The density must be on the interval (0,1].");
        }

        n_trees = n_trees_;
        depth = depth_;
        n_pool = n_trees_ * depth_;
        n_array = 1 << (depth_ + 1);

        if(density_ < 0) {
          density = 1.0 / std::sqrt(dim);
        } else {
          density = density_;
        }

        density < 1 ? build_sparse_random_matrix(sparse_random_matrix, n_pool, dim, density, seed) : build_dense_random_matrix(dense_random_matrix, n_pool, dim, seed);

        split_points = Eigen::MatrixXf(n_array, n_trees);
        tree_leaves = std::vector<std::vector<int>>(n_trees);

        count_first_leaf_indices_all(leaf_first_indices_all, n_samples, depth);
        leaf_first_indices = leaf_first_indices_all[depth];

        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            Eigen::MatrixXf tree_projections;

            if (density < 1)
                tree_projections.noalias() = sparse_random_matrix.middleRows(n_tree * depth, depth) * X;
            else
                tree_projections.noalias() = dense_random_matrix.middleRows(n_tree * depth, depth) * X;

            tree_leaves[n_tree] = std::vector<int>(n_samples);
            std::vector<int> &indices = tree_leaves[n_tree];
            std::iota(indices.begin(), indices.end(), 0);

            grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, tree_projections);
        }
    }

    void grow(const Eigen::Map<const Eigen::MatrixXf> &Q, int k_, int trees_max = -1, int depth_max = -1,
       int depth_min_ = -1, int votes_max_ = -1, float density_ = -1.0, int seed_mrpt = 0) {

      if(k_ <= 0 || k_ > n_samples) {
        throw std::out_of_range("k_ must belong to the set {1, ..., n}.");
      }

      if(trees_max < -1 || trees_max == 0) {
        throw std::out_of_range("trees_max must be positive.");
      }

      if(depth_max < -1 || depth_max == 0 || depth_max > std::log2(n_samples)) {
        throw std::out_of_range("depth_max must belong to the set {1, ... , log2(n)}.");
      }

      if(depth_min_ < -1 || depth_min_ == 0 || depth_min_ > depth_max) {
        throw std::out_of_range("depth_min_ must belong to the set {1, ... , depth_max}");
      }

      if(votes_max_ < -1 || votes_max_ == 0 || votes_max_ > trees_max) {
        throw std::out_of_range("votes_max_ must belong to the set {1, ... , trees_max}.");
      }

      if(density_ < -1.0001 || density_ > 1.0001 || (density_ > -0.9999 && density_ < -0.0001)) {
        throw std::out_of_range("The density must be on the interval (0,1].");
      }

      if(Q.rows() != dim) {
        throw std::invalid_argument("Dimensions of the data and the validation set do not match.");
      }

      if(trees_max == - 1) {
        trees_max = std::min(std::sqrt(n_samples), 1000.0);
      }
      if(depth_min_ == -1) {
        depth_min = std::min(static_cast<int>(std::log2(n_samples)), 5);
      } else {
        depth_min = depth_min_;
      }
      if(depth_max == -1) {
        depth_max = std::max(static_cast<int>(std::log2(n_samples) - 4), depth_min);
      }
      if(votes_max_ == -1) {
        votes_max = std::max(trees_max / 10, std::min(trees_max, 10));
      } else {
        votes_max = votes_max_;
      }
      if(density_ < 0) {
        density = 1.0 / std::sqrt(dim);
      } else {
        density = density_;
      }

      k = k_;
      int n_test = Q.cols();

      grow(trees_max, depth_max, density, seed_mrpt);
      Eigen::MatrixXi exact(k, n_test);
      compute_exact(Q, exact);

      recalls = std::vector<Eigen::MatrixXd>(depth_max - depth_min + 1);
      cs_sizes = std::vector<Eigen::MatrixXd>(depth_max - depth_min + 1);

      for(int d = depth_min; d <= depth_max; ++d) {
        recalls[d - depth_min] = Eigen::MatrixXd::Zero(votes_max, trees_max);
        cs_sizes[d - depth_min] = Eigen::MatrixXd::Zero(votes_max, trees_max);
      }

      for(int i = 0; i < n_test; ++i) {
        std::vector<Eigen::MatrixXd> recall_tmp(depth_max - depth_min + 1);
        std::vector<Eigen::MatrixXd> cs_size_tmp(depth_max - depth_min + 1);

        count_elected(Q.col(i), Eigen::Map<Eigen::VectorXi>(exact.data() + i * k, k),
         votes_max, recall_tmp, cs_size_tmp);

        for(int d = depth_min; d <= depth_max; ++d) {
          recalls[d - depth_min] += recall_tmp[d - depth_min];
          cs_sizes[d - depth_min] += cs_size_tmp[d - depth_min];
        }
      }

      for(int d = depth_min; d <= depth_max; ++d) {
        recalls[d - depth_min] /= (k * n_test);
        cs_sizes[d - depth_min] /= n_test;
      }

      fit_times(Q);
      index_type = autotuned_unpruned;
      par.k = k_;
    }

    void grow(const Eigen::MatrixXf &Q, int k_, int trees_max = -1, int depth_max = -1,
       int depth_min_ = -1, int votes_max_ = -1, float density_ = -1.0, int seed_mrpt = 0) {
       grow(Eigen::Map<const Eigen::MatrixXf>(Q.data(), Q.rows(), Q.cols()), k_, trees_max,
          depth_max, depth_min_, votes_max_, density_, seed_mrpt);
    }

    void grow(const float *data, int n_test, int k_, int trees_max = -1, int depth_max = -1,
       int depth_min_ = -1, int votes_max_ = -1, float density_ = -1.0, int seed_mrpt = 0) {
       grow(Eigen::Map<const Eigen::MatrixXf>(data, dim, n_test), k_, trees_max,
          depth_max, depth_min_, votes_max_, density_, seed_mrpt);
    }

    void grow(double target_recall, const Eigen::Map<const Eigen::MatrixXf> &Q, int k_, int trees_max = -1,
              int depth_min_ = -1, int depth_max = -1, int votes_max_ = -1,
              float density = -1.0, int seed_mrpt = 0) {
      if(target_recall < 0.0 - epsilon || target_recall > 1.0 + epsilon) {
        throw std::out_of_range("Target recall must be on the interval [0,1].");
      }
      grow(Q, k_, trees_max, depth_min_, depth_max, votes_max_, density, seed_mrpt);
      prune(target_recall);
    }

    void grow(double target_recall, const Eigen::MatrixXf &Q, int k_, int trees_max = -1,
              int depth_min_ = -1, int depth_max = -1, int votes_max_ = -1,
              float density = -1.0, int seed_mrpt = 0) {
      if(target_recall < 0.0 - epsilon || target_recall > 1.0 + epsilon) {
        throw std::out_of_range("Target recall must be on the interval [0,1].");
      }
      grow(Q, k_, trees_max, depth_min_, depth_max, votes_max_, density, seed_mrpt);
      prune(target_recall);
    }

    void grow(double target_recall, const float *data, int n_test, int k_, int trees_max = -1,
              int depth_min_ = -1, int depth_max = -1, int votes_max_ = -1,
              float density = -1.0, int seed_mrpt = 0) {
      if(target_recall < 0.0 - epsilon || target_recall > 1.0 + epsilon) {
        throw std::out_of_range("Target recall must be on the interval [0,1].");
      }
      grow(data, n_test, k_, trees_max, depth_min_, depth_max, votes_max_, density, seed_mrpt);
      prune(target_recall);
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
    void query(const Eigen::VectorXf &q, int k, int votes_required, int *out,
               float *out_distances = nullptr, int *out_n_elected = nullptr) const {

        if(k <= 0 || k > n_samples) {
          throw std::out_of_range("k must belong to the set {1, ..., n}.");
        }

        if(votes_required <= 0 || votes_required > n_trees) {
          throw std::out_of_range("votes_required must belong to the set {1, ... , n_trees}.");
        }

        if(empty()) {
          throw std::logic_error("The index must be built before making queries.");
        }

        Eigen::VectorXf projected_query(n_pool);
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
        Eigen::VectorXi elected(n_trees * max_leaf_size);
        Eigen::VectorXi votes = Eigen::VectorXi::Zero(n_samples);

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

        if(out_n_elected) {
          *out_n_elected = n_elected;
        }

        exact_knn(q, k, elected, n_elected, out, out_distances);
    }


    void query(const Eigen::VectorXf &q, int *out, float *out_distances = nullptr,
               int *out_n_elected = nullptr) const {
      if(index_type == normal) {
        throw std::logic_error("The index is not autotuned: k and vote threshold has to be specified.");
      }
      if(index_type == autotuned_unpruned) {
        throw std::logic_error("The target recall level has to be set before making queries.");
      }
      query(q, k, votes, out, out_distances, out_n_elected);
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
    void exact_knn(const Eigen::VectorXf &q, int k, const Eigen::VectorXi &indices,
      int n_elected, int *out, float *out_distances = nullptr) const {

        if(!n_elected) {
          for(int i = 0; i < k; ++i) out[i] = -1;
          if(out_distances) {
            for(int i = 0; i < k; ++i) out_distances[i] = -1;
          }
          return;
        }

        Eigen::VectorXf distances(n_elected);

        #pragma omp parallel for
        for (int i = 0; i < n_elected; ++i)
            distances(i) = (X.col(indices(i)) - q).squaredNorm();

        if (k == 1) {
            Eigen::MatrixXf::Index index;
            distances.minCoeff(&index);
            out[0] = n_elected ? indices(index) : -1;

            if(out_distances)
              out_distances[0] = n_elected ? std::sqrt(distances(index)) : -1;

            return;
        }

        int n_to_sort = n_elected > k ? k : n_elected;
        Eigen::VectorXi idx(n_elected);
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

        int i = index_type;
        fwrite(&i, sizeof(int), 1, fd);
        if(index_type == 2) {
          write_parameter_list(opt_pars, fd);
        }
        write_parameters(&par, fd);
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
                for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(sparse_random_matrix, k); it; ++it) {
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

        int i;
        fread(&i, sizeof(int), 1, fd);
        index_type = static_cast<itype>(i);
        if(index_type == autotuned_unpruned) {
          read_parameter_list(fd);
        }
        read_parameters(&par, fd);
        fread(&n_trees, sizeof(int), 1, fd);
        fread(&depth, sizeof(int), 1, fd);
        fread(&density, sizeof(float), 1, fd);

        n_pool = n_trees * depth;
        n_array = 1 << (depth + 1);

        count_first_leaf_indices_all(leaf_first_indices_all, n_samples, depth);
        leaf_first_indices = leaf_first_indices_all[depth];

        split_points = Eigen::MatrixXf(n_array, n_trees);
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

            sparse_random_matrix = Eigen::SparseMatrix<float>(n_pool, dim);
            std::vector<Eigen::Triplet<float>> triplets;
            for (int k = 0; k < non_zeros; ++k) {
                int row, col;
                float val;
                fread(&row, sizeof(int), 1, fd);
                fread(&col, sizeof(int), 1, fd);
                fread(&val, sizeof(float), 1, fd);
                triplets.push_back(Eigen::Triplet<float>(row, col, val));
            }

            sparse_random_matrix.setFromTriplets(triplets.begin(), triplets.end());
            sparse_random_matrix.makeCompressed();
        } else {
            dense_random_matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(n_pool, dim);
            fread(dense_random_matrix.data(), sizeof(float), n_pool * dim, fd);
        }

        fclose(fd);
        return true;
    }


    /**
    * @return - is the index empty: can it be used for queries?
    */
    bool empty() const {
      return n_trees == 0;
    }

    Mrpt_Parameters parameters() const {
      if(index_type == normal || index_type == autotuned_unpruned) {
        Mrpt_Parameters p;
        p.n_trees = n_trees;
        p.depth = depth;
        p.k = par.k;
        return p;
      }
      return par;
    }


  void prune(double target_recall) {
    if(target_recall < 0.0 - epsilon || target_recall > 1.0 + epsilon) {
      throw std::out_of_range("Target recall must be on the interval [0,1].");
    }
    par = parameters(target_recall);
    if(!par.n_trees) {
      return;
    }

    int depth_max = depth;

    n_trees = par.n_trees;
    depth = par.depth;
    votes = par.votes;
    n_pool = depth * n_trees;
    n_array = 1 << (depth + 1);

    tree_leaves.resize(n_trees);
    split_points.conservativeResize(n_array, n_trees);
    leaf_first_indices = leaf_first_indices_all[depth];
    if(density < 1) {
      Eigen::SparseMatrix<float, Eigen::RowMajor> srm_new(n_pool, dim);
      for(int n_tree = 0; n_tree < n_trees; ++n_tree)
        srm_new.middleRows(n_tree * depth, depth) = sparse_random_matrix.middleRows(n_tree * depth_max, depth);
      sparse_random_matrix = srm_new;
    } else {
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> drm_new(n_pool, dim);
      for(int n_tree = 0; n_tree < n_trees; ++n_tree)
        drm_new.middleRows(n_tree * depth, depth) = dense_random_matrix.middleRows(n_tree * depth_max, depth);
      dense_random_matrix = drm_new;
    }
    index_type = autotuned;
  }

  Mrpt subset(double target_recall) const {
    if(target_recall < 0.0 - epsilon || target_recall > 1.0 + epsilon) {
      throw std::out_of_range("Target recall must be on the interval [0,1].");
    }

    Mrpt index2(X);
    index2.par = parameters(target_recall);

    int depth_max = depth;

    index2.n_trees = index2.par.n_trees;
    index2.depth = index2.par.depth;
    index2.votes = index2.par.votes;
    index2.n_pool = index2.depth * index2.n_trees;
    index2.n_array = 1 << (index2.depth + 1);
    index2.tree_leaves.assign(tree_leaves.begin(), tree_leaves.begin() + index2.n_trees);
    index2.leaf_first_indices_all = leaf_first_indices_all;
    index2.density = density;
    index2.k = k;

    index2.split_points = split_points.topLeftCorner(index2.n_array, index2.n_trees);
    index2.leaf_first_indices = leaf_first_indices_all[index2.depth];
    if(index2.density < 1) {
      index2.sparse_random_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(index2.n_pool, index2.dim);
      for(int n_tree = 0; n_tree < index2.n_trees; ++n_tree)
        index2.sparse_random_matrix.middleRows(n_tree * index2.depth, index2.depth) = sparse_random_matrix.middleRows(n_tree * depth_max, index2.depth);
    } else {
      index2.dense_random_matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(index2.n_pool, index2.dim);
      for(int n_tree = 0; n_tree < index2.n_trees; ++n_tree)
        index2.dense_random_matrix.middleRows(n_tree * index2.depth, index2.depth) = dense_random_matrix.middleRows(n_tree * depth_max, index2.depth);
    }
    index2.index_type = autotuned;
    return index2;
  }

  std::vector<Mrpt_Parameters> optimal_pars() const {
    if(index_type == normal) {
      throw std::logic_error("The list of optimal parameters cannot be retrieved for the non-autotuned index.");
    }
    if(index_type == autotuned) {
      throw std::logic_error("The list of optimal parameters cannot be retrieved for the index which has already been subsetted or deleted to the target recall level.");
    }
    std::vector<Mrpt_Parameters> new_pars;
    std::copy(opt_pars.begin(), opt_pars.end(), std::back_inserter(new_pars));
    return new_pars;
  }

  // Friend declarations for test classes.
  // Tests are located at https://github.com/vioshyvo/RP-test
  friend class MrptTest;
  friend class UtilityTest;

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
          int tree_level, int i, int n_tree, const Eigen::MatrixXf &tree_projections) {
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


    void count_elected(const Eigen::VectorXf &q, const Eigen::Map<Eigen::VectorXi> &exact, int votes_max,
      std::vector<Eigen::MatrixXd> &recalls, std::vector<Eigen::MatrixXd> &cs_sizes) const {
        Eigen::VectorXf projected_query(n_pool);
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
          Eigen::VectorXi votes = Eigen::VectorXi::Zero(n_samples);
          const std::vector<int> &leaf_first_indices = leaf_first_indices_all[depth_crnt];

          Eigen::MatrixXd recall(votes_max, n_trees);
          Eigen::MatrixXd candidate_set_size(votes_max, n_trees);
          recall.col(0) = Eigen::VectorXd::Zero(votes_max);
          candidate_set_size.col(0) = Eigen::VectorXd::Zero(votes_max);

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
    static void build_sparse_random_matrix(Eigen::SparseMatrix<float, Eigen::RowMajor> &sparse_random_matrix,
          int n_row, int n_col, float density, int seed = 0) {
        sparse_random_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(n_row, n_col);

        std::random_device rd;
        int s = seed ? seed : rd();
        std::mt19937 gen(s);
        std::uniform_real_distribution<float> uni_dist(0, 1);
        std::normal_distribution<float> norm_dist(0, 1);

        std::vector<Eigen::Triplet<float>> triplets;
        for (int j = 0; j < n_row; ++j) {
            for (int i = 0; i < n_col; ++i) {
                if (uni_dist(gen) > density) continue;
                triplets.push_back(Eigen::Triplet<float>(j, i, norm_dist(gen)));
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
    static void build_dense_random_matrix(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &dense_random_matrix,
          int n_row, int n_col, int seed = 0) {
        dense_random_matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(n_row, n_col);

        std::random_device rd;
        int s = seed ? seed : rd();
        std::mt19937 gen(s);
        std::normal_distribution<float> normal_dist(0, 1);

        std::generate(dense_random_matrix.data(), dense_random_matrix.data() + n_row * n_col,
                      [&normal_dist, &gen] { return normal_dist(gen); });
    }


    void compute_exact(const Eigen::Map<const Eigen::MatrixXf> &Q, Eigen::MatrixXi &out_exact) const {
      int n_test = Q.cols();
      for(int i = 0; i < n_test; ++i) {
        Eigen::VectorXi idx(n_samples);
        std::iota(idx.data(), idx.data() + n_samples, 0);

        exact_knn(Q.col(i), k, idx, n_samples, out_exact.data() + i * k);
        std::sort(out_exact.data() + i * k, out_exact.data() + i * k + k);
      }
    }

    static bool is_faster(const Mrpt_Parameters &par1, const Mrpt_Parameters &par2) {
      return par1.estimated_qtime < par2.estimated_qtime;
    }

    void vote(const Eigen::VectorXf &projected_query, int votes_required, Eigen::VectorXi &elected,
      int &n_elected, int n_trees, int depth_crnt) {
      std::vector<int> found_leaves(n_trees);
      const std::vector<int> &leaf_first_indices = leaf_first_indices_all[depth_crnt];


      #pragma omp parallel for
      for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
        int idx_tree = 0;
        for (int d = 0; d < depth_crnt; ++d) {
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
        found_leaves[n_tree] = idx_tree - (1 << depth_crnt) + 1;
      }

      int max_leaf_size = n_samples / (1 << depth_crnt) + 1;
      elected = Eigen::VectorXi(n_trees * max_leaf_size);
      Eigen::VectorXi votes = Eigen::VectorXi::Zero(n_samples);

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


    void fit_times(const Eigen::Map<const Eigen::MatrixXf> &Q) {
      int n_test = Q.cols();
      std::vector<double> projection_times, projection_x;
      std::vector<double> exact_times;
      std::vector<int> exact_x;

      long double idx_sum = 0;

      std::random_device rd;
      std::mt19937 rng(rd());
      std::uniform_int_distribution<int> uni(0, n_test-1);
      std::uniform_int_distribution<int> uni2(0, n_samples-1);

      std::vector<int> tested_trees {1,2,3,4,5,7,10,15,20,25,30,40,50};
      int n_tested_trees = 10;
      n_tested_trees = n_trees > n_tested_trees ? n_tested_trees : n_trees;
      int incr = n_trees / n_tested_trees;
      for(int i = 1; i <= n_tested_trees; ++i)
        if(std::find(tested_trees.begin(), tested_trees.end(), i * incr) == tested_trees.end()) {
          tested_trees.push_back(i * incr);
        }

      // remove tested tree numbers that are larger than the number of trees in the index
      std::sort(tested_trees.begin(), tested_trees.end());
      auto tt = tested_trees.begin();
      for(; tt != tested_trees.end() && *tt <= n_trees; ++tt);
      tested_trees.erase(tt, tested_trees.end());

      for(int d = depth_min; d <= depth; ++d) {
        for(int i = 0; i < tested_trees.size(); ++i) {
          int t = tested_trees[i];
          int n_random_vectors = t * d;
          projection_x.push_back(n_random_vectors);
          Eigen::SparseMatrix<float, Eigen::RowMajor> sparse_mat;
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dense_mat;
          if(density < 1) {
            build_sparse_random_matrix(sparse_mat, n_random_vectors, dim, density);
          } else {
            build_dense_random_matrix(dense_mat, n_random_vectors, dim);
          }

          double start_proj = omp_get_wtime();
          Eigen::VectorXf projected_query(n_random_vectors);
          if(density < 1) {
            projected_query.noalias() = sparse_mat * Q.col(0);
          } else {
            projected_query.noalias() = dense_mat * Q.col(0);
          }
          double end_proj = omp_get_wtime();
          projection_times.push_back(end_proj - start_proj);
          idx_sum += projected_query.norm();

          int votes_index = votes_max < t ? votes_max : t;
          for(int v = 1; v <= votes_index; ++v) {
            int cs_size = get_candidate_set_size(t, d, v);
            if(cs_size > 0) exact_x.push_back(cs_size);
          }
        }
      }

      std::vector<int> s_tested {1,2,5,10,20,35,50,75,100,150,200,300,400,500};
      int s_max = n_samples / 20;
      int n_s_tested = 20;
      std::vector<double> ex;
      int increment = s_max / n_s_tested;
      for(int i = 1; i <= n_s_tested; ++i)
        if(std::find(s_tested.begin(), s_tested.end(), i * increment) == s_tested.end()) {
          s_tested.push_back(i * increment);
        }

      // remove candidate set sizes that are larger than the size of the data set
      std::sort(s_tested.begin(), s_tested.end());
      auto s = s_tested.begin();
      for(; s != s_tested.end() && *s <= n_samples; ++s);
      s_tested.erase(s, s_tested.end());

      std::vector<double> vote_thresholds_x {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
      int n_votes = 10; // for how many different vote thresholds voting is tested
      n_votes = votes_max > n_votes ? n_votes : votes_max;
      int inc = votes_max / n_votes;
      for(int i = 1; i <= n_votes; ++i)
        if(std::find(vote_thresholds_x.begin(), vote_thresholds_x.end(), i * inc) == vote_thresholds_x.end()) {
          vote_thresholds_x.push_back(i * inc);
        }

      // remove tested vote thresholds that are larger than the preset maximum vote threshold
      std::sort(vote_thresholds_x.begin(), vote_thresholds_x.end());
      auto vt = vote_thresholds_x.begin();
      for(; vt != vote_thresholds_x.end() && *vt <= votes_max; ++vt);
      vote_thresholds_x.erase(vt, vote_thresholds_x.end());

      beta_voting = std::vector<std::map<int,std::pair<double,double>>>();

      for(int d = depth_min; d <= depth; ++d) {
        std::map<int,std::pair<double,double>> beta;
        for(const auto &v : vote_thresholds_x) {
          std::vector<double> voting_times, voting_x;

          for(int i = 0; i < tested_trees.size(); ++i) {
            int t = tested_trees[i];
            int n_el = 0;
            Eigen::VectorXi elected;
            auto ri = uni(rng);

            Eigen::VectorXf projected_query(n_trees * depth);
            if(density < 1) {
              projected_query.noalias() = sparse_random_matrix * Q.col(ri);
            } else {
              projected_query.noalias() = dense_random_matrix * Q.col(ri);
            }

            double start_voting = omp_get_wtime();
            vote(projected_query, v, elected, n_el, t, d);
            double end_voting = omp_get_wtime();

            voting_times.push_back(end_voting - start_voting);
            voting_x.push_back(t);
            for(int i = 0; i < n_el; ++i)
              idx_sum += elected(i);
          }
          beta[v] = fit_theil_sen(voting_x, voting_times);
        }
        beta_voting.push_back(beta);
      }

      int n_sim = 100;
      for(int i = 0; i < s_tested.size(); ++i) {
        double mean_exact_time = 0;
        int s_size = s_tested[i];
        ex.push_back(s_size);

        for(int m = 0; m < n_sim; ++m) {
          auto ri = uni(rng);
          Eigen::VectorXi elected(s_size);
          for(int j = 0; j < elected.size(); ++j)
            elected(j) = uni2(rng);

          double start_exact = omp_get_wtime();
          std::vector<int> res(k);
          exact_knn(Q.col(ri), k, elected, s_size, &res[0]);
          double end_exact = omp_get_wtime();
          mean_exact_time += (end_exact - start_exact);

          for(int l = 0; l < k; ++l)
            idx_sum += res[l];
        }

        mean_exact_time /= n_sim;
        exact_times.push_back(mean_exact_time);
      }

      beta_projection = fit_theil_sen(projection_x, projection_times);
      beta_exact = fit_theil_sen(ex, exact_times);

      std::set<Mrpt_Parameters,decltype(is_faster)*> pars(is_faster);
      query_times = std::vector<Eigen::MatrixXd>(depth - depth_min + 1);
      for(int d = depth_min; d <= depth; ++d) {
        Eigen::MatrixXd query_time = Eigen::MatrixXd::Zero(votes_max, n_trees);

        for(int t = 1; t <= n_trees; ++t) {
          int votes_index = votes_max < t ? votes_max : t;
          for(int v = 1; v <= votes_index; ++v) {
            double qt = get_query_time(t, d, v);
            query_time(v - 1, t - 1) = qt;
            Mrpt_Parameters p;
            p.n_trees = t;
            p.depth = d;
            p.votes = v;
            p.k = k;
            p.estimated_qtime = qt;
            p.estimated_recall = get_recall(t, d, v);
            pars.insert(p);
          }
        }
        query_times[d - depth_min] = query_time;
      }

      // Just to make sure that the compiler does not optimize away timed code.
      pars.begin()->estimated_recall += idx_sum > 1.0 ? 0.0000 : 0.0001;

      opt_pars = std::set<Mrpt_Parameters,decltype(is_faster)*>(is_faster);
      double best_recall = -1.0;
      for(const auto &p : pars) // compute pareto frontier for query times and recalls
        if(p.estimated_recall > best_recall) {
          opt_pars.insert(p);
          best_recall = p.estimated_recall;
        }
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

    void write_parameters(const Mrpt_Parameters *p, FILE *fd) const {
      if(!fd) {
        return;
      }
      fwrite(&p->n_trees, sizeof(int), 1, fd);
      fwrite(&p->depth, sizeof(int), 1, fd);
      fwrite(&p->votes, sizeof(int), 1, fd);
      fwrite(&p->k, sizeof(int), 1, fd);
      fwrite(&p->estimated_qtime, sizeof(double), 1, fd);
      fwrite(&p->estimated_recall, sizeof(double), 1, fd);
    }

    void read_parameters(Mrpt_Parameters *p, FILE *fd) {
      fread(&p->n_trees, sizeof(int), 1, fd);
      fread(&p->depth, sizeof(int), 1, fd);
      fread(&p->votes, sizeof(int), 1, fd);
      fread(&p->k, sizeof(int), 1, fd);
      fread(&p->estimated_qtime, sizeof(double), 1, fd);
      fread(&p->estimated_recall, sizeof(double), 1, fd);
    }

    void write_parameter_list(const std::set<Mrpt_Parameters,decltype(is_faster)*> &pars, FILE *fd) const {
      if(!fd) {
        return;
      }
      int par_sz = pars.size();
      fwrite(&par_sz, sizeof(int), 1, fd);
      for(const auto p : pars)
        write_parameters(&p, fd);
    }

    void read_parameter_list(FILE *fd) {
      if(!fd) {
        return;
      }
      opt_pars = std::set<Mrpt_Parameters,decltype(is_faster)*>(is_faster);
      int par_sz = 0;
      fread(&par_sz, sizeof(int), 1, fd);
      for(int i = 0; i < par_sz; ++i) {
        Mrpt_Parameters p;
        read_parameters(&p, fd);
        opt_pars.insert(p);
      }
    }



    Mrpt_Parameters parameters(double target_recall) const {
      double tr = target_recall - epsilon;
      for(const auto &p : opt_pars)
        if(p.estimated_recall > tr) {
          return p;
        }

      if(!opt_pars.empty()) {
        return *(opt_pars.rbegin());
      }

      return Mrpt_Parameters();
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

    static double predict_theil_sen(double x, std::pair<double,double> beta) {
      return beta.first + beta.second * x;
    }

    double get_recall(int tree, int depth, int v) const {
      return recalls[depth - depth_min](v - 1, tree - 1);
    }

    double get_candidate_set_size(int tree, int depth, int v) const {
      return cs_sizes[depth - depth_min](v - 1, tree - 1);
    }

    double get_projection_time(int n_trees, int depth, int v) const {
      return predict_theil_sen(n_trees * depth, beta_projection);
    }

    double get_voting_time(int n_trees, int depth, int v) const {
      const std::map<int,std::pair<double,double>> &beta = beta_voting[depth - depth_min];
      if(v <= 0 || beta.empty()) {
        return 0.0;
      }
      for(const auto &b : beta)
        if(v <= b.first) {
          return predict_theil_sen(n_trees, b.second);
        }

      return predict_theil_sen(n_trees, beta.rbegin()->second);
    }

    double get_exact_time(int n_trees, int depth, int v) const {
      return predict_theil_sen(get_candidate_set_size(n_trees, depth, v), beta_exact);
    }

    double get_query_time(int tree, int depth, int v) const {
      return get_projection_time(tree, depth, v)
           + get_voting_time(tree, depth, v)
           + get_exact_time(tree, depth, v);
    }

    const Eigen::Map<const Eigen::MatrixXf> X; // the data matrix
    Eigen::Map<Eigen::MatrixXf> *Q; // validation set
    Eigen::MatrixXf split_points; // all split points in all trees
    std::vector<std::vector<int>> tree_leaves; // contains all leaves of all trees
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dense_random_matrix; // random vectors needed for all the RP-trees
    Eigen::SparseMatrix<float, Eigen::RowMajor> sparse_random_matrix; // random vectors needed for all the RP-trees
    std::vector<std::vector<int>> leaf_first_indices_all; // first indices for each level
    std::vector<int> leaf_first_indices; // first indices of each leaf of tree in tree_leaves

    const int n_samples; // sample size of data
    const int dim; // dimension of data
    Mrpt_Parameters par;
    int n_trees = 0; // number of RP-trees
    int depth = 0; // depth of an RP-tree with median split
    float density = -1.0; // expected ratio of non-zero components in a projection matrix
    int n_pool = 0; // amount of random vectors needed for all the RP-trees
    int n_array = 0; // length of the one RP-tree as array
    int votes = 0; // optimal number of votes to use
    int depth_min = 0;
    int votes_max = 0;
    int k = 0;
    enum itype {normal, autotuned, autotuned_unpruned};
    itype index_type = normal;
    const double epsilon = 0.0001; // error bound for comparisons of recall levels

    std::vector<Eigen::MatrixXd> recalls, cs_sizes, query_times;
    std::pair<double,double> beta_projection, beta_exact;
    std::vector<std::map<int,std::pair<double,double>>> beta_voting;
    std::set<Mrpt_Parameters,decltype(is_faster)*> opt_pars;
};


#endif // CPP_MRPT_H_

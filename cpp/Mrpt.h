/********************************************************
 * Ville Hyvönen & Teemu Pitkänen                       *
 * HIIT / University of Helsinki                        *
 * ville.o.hyvonen<at>helsinki.fi                       *
 * teemu.pitkanen<at>cs.helsinki.fi                     *
 * 2016                                                 *
 ********************************************************/

#include <vector>
#include <string>
#include <random>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

using namespace Eigen;

#ifndef MRPT_H
#define	MRPT_H

enum Metric { EUCLIDEAN, ANGULAR };

class Mrpt {
public:
    Mrpt(const MatrixXf& X_, int n_trees, int n_0_, float density_, std::string metric_, std::string id_);
    /* Mrpt(const std::string X_, int n_trees, int n_0_, std::string id_); */
    ~Mrpt() {}
    void grow();
    VectorXi query(const VectorXf& q, int k, int votes_required, int branches);
    VectorXi query(const VectorXf& q, int k); // the old query

private:
    std::vector<VectorXi> grow_subtree(const VectorXi &indices, int tree_level, int i, unsigned n_tree);
    const MatrixXf X; // data matrix, col = observation, row = dimension
    VectorXf X_norms; // cache norms of the observations in X for distance calculations
    int n_trees; // number of RP-trees
    int n_0; // maximum leaf size of all the RP-trees
    int n_samples; // sample size of data
    int dim; // dimension of data
    int depth; // depth of an RP-tree with median split
    int n_pool; // amount of random vectors needed for all the RP-trees
    MatrixXf dense_random_matrix; // random vectors needed for all the RP-trees
    SparseMatrix<float> sparse_random_matrix; // random vectors needed for all the RP-trees
    MatrixXf projected_data; // data matrix projected onto all the random vectors
    MatrixXf split_points; // All split points in all trees.
    std::vector<std::vector<VectorXi > > tree_leaves; // Contains all leaves of all trees. Indexed as tree_leaves[tree number][leaf number][index in leaf]
    int n_array; // length of the one RP-tree as array
    unsigned first_idx;
    float density;
    Metric metric;
    std::string id;
};


/**
 * The exact nn needed as the final step of MRPT query
 * @param D - The data set. Samples as columns, features as rows.
 * @param D_norms - Norms of the samples in D
 * @param q - The object whose neighbors are searched
 * @param k - The number of neighbors searched for
 * @param indices - A subset of indices of range(1, n_samples), the set of 
 * samples taken into account in the knn search. The other samples are ignored.
 * @return The indices (col numbers) of the neighbors in D
 */
VectorXi exact_knn(const MatrixXf& D, const VectorXf& D_norms, const VectorXf& q, unsigned k,
                   const VectorXi &indices, Metric metric);

/**
 * Builds a random sparse matrix for use in random projection. The components of
 * the matrix are drawn from the distribution
 *
 * -sqrt(s)   w.p. 1 / 2s
 *  0         w.p. 1 - 1 / s
 * +sqrt(s)   w.p. 1 / 2s
 *
 * where s = 1 / density.
 * @param rows - The number of rows in the resulting matrix.
 * @param cols - The number of columns in the resulting matrix.
 * @param density - Expected ratio of non-zero components in the resulting matrix.
 * @param gen - A random number engine.
 */
SparseMatrix<float> buildSparseRandomMatrix(int rows, int cols, float density, std::mt19937 &gen);

/*
 * Builds a random dense matrix for use in random projection. The components of
 * the matrix are drawn from the standard normal distribution.
 * @param rows - The number of rows in the resulting matrix.
 * @param cols - The number of rows in the resulting matrix.
 * @param gen - A random number engine.
 */
MatrixXf buildDenseRandomMatrix(int rows, int cols, std::mt19937 &gen);

/**
 * This class defines the elements that are stored in the priority queue for 
 * the extra branch / priority queue trick. An instance of the class describes a
 * single node in a rp-tree in a single query. The most important field 
 * gap_width tells the difference of the split value used in this node and the 
 * projection of the query vector in this node. This is used as a criterion to 
 * choose extra branches -- a small distance indicates that some neighbors may
 * easily end up on the other side of split. The rest of the fields are needed 
 * to start a tree traversal from the node "on the other side of the split", 
 * and the methods are needed for sorting in the priority queue.
 */
class Gap {
public:
    int tree; // The ordinal of the tree
    int node; // The node corresponding to the other side of the split
    int level; // The level in the tree where node lies
    double gap_width; // The gap between the query projection and split value at the parent of node.
    
    Gap(int tree_, int node_, int level_, double gap_width_)
    : tree(tree_), node(node_), level(level_), gap_width(gap_width_) {
    }
    friend bool operator<(
            const Gap& a, const Gap& b) {
        return a.gap_width < b.gap_width;
    }
    friend bool operator>(
            const Gap& a, const Gap& b) {
        return a.gap_width > b.gap_width;
    }
};

#endif	/* MRPT_H */


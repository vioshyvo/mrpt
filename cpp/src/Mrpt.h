#include <armadillo>
using namespace arma;

#ifndef MRPT_H
#define	MRPT_H

/*******************************************************
 * Multiple random projection trees class
 * Ville Hyvönen
 * HIIT
 * ville.o.hyvonen<at>helsinki.fi 
 * 07.11.2015
 ********************************************************/

class Mrpt {
public:

    Mrpt(const fmat& X_, int n_trees, int n_0_, std::string id_);

    ~Mrpt() {}

    std::vector<double> grow();
    
    void read_trees();

    uvec query(const fvec& q, int k, int elect, int branches);

    uvec query_canditates(const fvec& q, int k);

    void matrix_multiplication(const fvec& q);

private:
    void grow_subtree(const uvec &indices, int tree_level, int i, uword n_tree);

    fmat X; // data matrix, col = observation, row = dimension
    int n_trees; // number of RP-trees
    int n_0; // maximum leaf size of all the RP-trees
    int n_rows; // sample size of data
    int dim; // dimension of data
    int depth; // depth of an RP-tree with median split
    int n_pool; // amount of random vectors needed for all the RP-trees
    fmat random_matrix; // random vectors needed for all the RP-trees
    fmat projected_data; // data matrix projected onto all the random vectors
    fmat trees; // all the RP-trees, col = tree, row = node
    std::vector<std::vector<uvec> > leaf_labels; // leaf labels of all the data points, col = tree, row = data point
    int n_array; // length of the one RP-tree as array
    std::string id;
};

class Gap {
/*
 * Elements stored in the priority queue. gap_width is used as the priority and 
 * the other fields are needed to find the correct node where to continue routing.
 */
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
        if (a.gap_width < b.gap_width)
            return true;
        return false;
    }
    friend bool operator>(
            const Gap& a, const Gap& b) {
        if (a.gap_width > b.gap_width)
            return true;
        return false;
    }
};

#endif	/* MRPT_H */


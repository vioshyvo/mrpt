/********************************************************
 * Ville Hyvönen & Teemu Pitkänen                       *
 * HIIT / University of Helsinki                        *
 * ville.o.hyvonen<at>helsinki.fi                       *
 * teemu.pitkanen<at>cs.helsinki.fi                     *
 * 2016                                                 *
 ********************************************************/

#include <armadillo>
using namespace arma;

#ifndef MRPT_H
#define	MRPT_H

class Mrpt {
public:
    Mrpt(const fmat& X_, int n_trees, int n_0_, std::string id_);
    ~Mrpt() {}
    void grow();
    uvec query(const fvec& q, int k, int elect, int branches);
    uvec query(const fvec& q, int k, int elect); // Query without priority queue
    uvec query(const fvec& q, int k); // the old query

private:
    std::vector<uvec> grow_subtree(const uvec &indices, int tree_level, int i, uword n_tree);
    fmat X; // data matrix, col = observation, row = dimension
    int n_trees; // number of RP-trees
    int n_0; // maximum leaf size of all the RP-trees
    int n_samples; // sample size of data
    int dim; // dimension of data
    int depth; // depth of an RP-tree with median split
    int n_pool; // amount of random vectors needed for all the RP-trees
    fmat random_matrix; // random vectors needed for all the RP-trees
    fmat projected_data; // data matrix projected onto all the random vectors
    fmat split_points; // All split points in all trees.
    std::vector<std::vector<uvec> > tree_leaves; // Contains all leaves of all trees. Indexed as tree_leaves[tree number][leaf number][index in leaf]
    int n_array; // length of the one RP-tree as array
    uword first_idx;
    std::string id;
};


/**
 * The exact nn needed as the final step of MRPT query
 * @param D - The data set. Samples as columns, features as rows.
 * @param q - The object whose neighbors are searched
 * @param k - The number of neighbors searched for
 * @param indices - A subset of indices of range(1, n_samples), the set of 
 * samples taken into account in the knn search. The other samples are ignored.
 * @return The indices (col numbers) of the neighbors in D
 */
uvec exact_knn(const fmat& D, const fvec& q, uword k, uvec indices);
    

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


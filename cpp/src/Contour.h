#ifndef CONTOUR_H
#define	CONTOUR_H

/*******************************************************
 * Class with Multiple random projection trees objects with
 * constant n_trees * n_0
 * Ville Hyvönen
 * HIIT
 * ville.o.hyvonen<at>helsinki.fi 
 * 07.11.2015
 ********************************************************/
class Mrpt;

#include "Result.h"

class Contour {
    
public:
    Contour(const fmat& X_, uvec n_trees_, uvec n_0_, std::string id_);

    ~Contour();

    void grow();

    void query(const fmat& Q, int k_, umat true_knn);

    Result results(double time_exact);

private:
    const fmat& X; // original data matrix, col = observation, row = dimension
    int n_mrpts; // number of Mrpt:s
    uvec n_trees; // vector of number of trees
    uvec n_0; // vector of maximum leaf sizes
    Mrpt* mrpts; // all the Mrpt:s
    fvec nn_found; // average true nearest neighbors for all the test points 
    // vec n_search_space;
    int k; // number of nearest neighbors searched for
    fvec times_query; // query times in trees 
    fvec times_knn; // knn times in a final search spaxe
    fvec times_total; // total query times
    fvec growing_times; // growing times for the trees
    int n_points; // number of test points
    fvec times_matrix;
    fvec times_trees;
    fvec times_multi;
    std::string id;
};

#endif	/* CONTOUR_H */


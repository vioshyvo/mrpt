#ifndef CONTOURS_H
#define	CONTOURS_H

/*******************************************************
 * Container for Contour objects
 * Ville Hyvönen
 * HIIT
 * ville.o.hyvonen<at>helsinki.fi 
 * 07.11.2015
 ********************************************************/

class Contour;

#include "Result.h"

class Contours {

public:
    Contours(const arma::fmat& X_, int n_contours_, std::vector<arma::uvec> n_trees_, std::vector<arma::uvec> n_0_, std::string id_);

    // min_S = log_2 of minimum search space size
    // max_S = log_2 of maximum search space size
    // min_leaf = smallest minimum leaf size n_0 used
    Contours(const arma::fmat& X_, int min_S, int max_S, int min_leaf, std::string id_);

    ~Contours();
    
    void grow();

    void query(const arma::fmat& Q, int k);
    
    void write_results(std::vector<Result> results, std::string io);
      
    std::vector<Result> results();

private:
    const arma::fmat& X; // original data matrix, col = observation, row = dimension
    int n_contours; // number of Contour objects in contours
    std::vector<arma::uvec> n_trees; // vector of number of trees
    std::vector<arma::uvec> n_0; // vector of maximum leaf sizes
    Contour** contours; // all the Contours
    double time_exact; // time it takes to do exact knn search for n_points points
    std::string id;
};

#endif	/* CONTOURS_H */


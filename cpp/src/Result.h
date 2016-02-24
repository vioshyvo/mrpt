#ifndef RESULT_H
#define	RESULT_H


struct Result {
    fvec nn_found;
    int k;
    uvec n_trees;
    uvec n_0;
    fvec times_knn;
    fvec times_query;
    fvec growing_times;
    double time_exact;
    int n_points;
    fvec times_matrix;
    fvec times_trees;
    fvec times_total;
    fvec times_multi;
};

void print_result(Result r);

#endif	/* RESULT_H */


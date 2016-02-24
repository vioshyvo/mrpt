#include <armadillo>

using namespace arma;

#include "Result.h"

void print_result(Result r) {
    std::cout << "nn found: " << r.nn_found.t() << "\n";
    std::cout << "k: " << r.k << "\n\n";
    std::cout << "n_trees: " << r.n_trees.t() << "\n";
    std::cout << "n_0: " << r.n_0.t() << "\n";
    std::cout << "times_knn: " << r.times_knn.t() << "\n";
    std::cout << "times_query: " << r.times_query.t() << "\n";
    std::cout << "growing_times: " << r.growing_times.t() << "\n";
    std::cout << "time_exact: " << r.time_exact << "\n\n";
    std::cout << "n_points: " << r.n_points << "\n\n";
    std::cout << "times_matrix: " << r.times_matrix.t() << "\n";
    std::cout << "times_trees: " << r.times_trees.t() << "\n";
    std::cout << "times_total: " << r.times_total.t() << "\n";
    std::cout << "times_multi: " << r.times_multi.t() << "\n\n\n";
}



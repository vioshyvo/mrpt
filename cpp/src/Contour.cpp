#include "armadillo"
#include <ctime>

using namespace arma;

#include "Result.h"
#include "Mrpt.h"
#include "Contour.h"
#include "knn.h"

Contour::Contour(const fmat& X_, uvec n_trees_, uvec n_0_, std::string id_) : X(X_), n_trees(n_trees_), n_0(n_0_), id(id_) {
    n_mrpts = n_trees.size();
    mrpts = nullptr;
    times_matrix = fvec(n_mrpts);
    times_trees = fvec(n_mrpts);
}

Contour::~Contour() {
    if (mrpts) {
        delete mrpts;
        mrpts = nullptr;
    }
}

void Contour::grow() {
    growing_times = fvec(n_mrpts);
    std::vector<double> times;

    for (int i = 0; i < n_mrpts; i++) {
        std::cout << "n_trees: " << n_trees[i] << ", n_0: " << n_0[i] << std::endl;

        clock_t begin = clock();
        mrpts = new Mrpt(X, n_trees[i], n_0[i], id + std::to_string(i));
        times = mrpts->grow();
        clock_t end = clock();
        growing_times[i] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);
        times_matrix[i] = times[0];
        times_trees[i] = times[1];
        delete mrpts;
        mrpts = nullptr;
    }

    std::cout << std::endl;
}

void Contour::query(const fmat& Q, int k_, umat true_knn) {
    k = k_;
    n_points = Q.n_cols;
    nn_found = zeros<fvec>(n_mrpts);
    // n_search_space = zeros<vec>(n_mrpts);
    std::vector<uvec> approximate_knn(n_points);
    std::vector<uvec> idx_canditates(n_points);
    times_query = fvec(n_mrpts);
    times_knn = fvec(n_mrpts);
    times_total = fvec(n_mrpts);
    times_multi = fvec(n_mrpts);

    for (int i = 0; i < n_mrpts; i++) {
        mrpts = new Mrpt(X, n_trees[i], n_0[i], id + std::to_string(i));
        mrpts->read_trees();

        clock_t begin = clock();
        for (int j = 0; j < n_points; j++)
            idx_canditates[j] = mrpts->query_canditates(Q.unsafe_col(j), k);
        clock_t end = clock();
        times_query[i] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);

        begin = clock();
        for (int j = 0; j < n_points; j++)
            approximate_knn[j] = knnCpp_T_indices(X, Q.unsafe_col(j), k, idx_canditates[j]);
        end = clock();
        times_knn[i] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);

        begin = clock();
        for (int j = 0; j < n_points; j++)
            mrpts->query(Q.unsafe_col(j), k);
        end = clock();
        times_total[i] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);

        begin = clock();
        for (int j = 0; j < n_points; j++)
            mrpts->matrix_multiplication(Q.unsafe_col(j));
        end = clock();
        times_multi[i] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);


        for (int j = 0; j < n_points; j++) {
            int n_knn = approximate_knn[j].size();
            for (int l = 0; l < n_knn; l++)
                if (any(true_knn.col(j) == approximate_knn[j][l]))
                    nn_found[i]++;
        }

        delete mrpts;
        mrpts = nullptr;
    }

    nn_found /= n_points;

}

Result Contour::results(double time_exact) {
    return Result{nn_found, k, n_trees, n_0, times_knn, times_query, growing_times, time_exact, n_points,
        times_matrix, times_trees, times_total, times_multi};
}


//  Rcpp::List results(double time_exact) {
//    Rcpp::List ret = Rcpp::List::create(
//      Rcpp::_["nn_found"] = std::vector<double>(nn_found.begin(), nn_found.end()),
//      Rcpp::_["k"] = k,
//      Rcpp::_["n_trees"] = std::vector<int>(n_trees.begin(), n_trees.end()),
//      Rcpp::_["n_0"] = std::vector<int>(n_0.begin(), n_0.end()),
//      Rcpp::_["times_knn"] = times_knn,
//      Rcpp::_["times_query"] = times_query,
//      Rcpp::_["growing_times"] = growing_times,
//      Rcpp::_["time_exact"] = time_exact,
//      Rcpp::_["n_points"] = n_points,
//      Rcpp::_["times_matrix"] = times_matrix,
//      Rcpp::_["times_trees"] = times_trees,
//      Rcpp::_["times_total"] = times_total,
//      Rcpp::_["times_multi"] = times_multi
//    );
//
//    ret.attr("class") = "contour";
//
//    return ret;
//  }

#include <armadillo>
#include <ctime>
#include <fstream>

using namespace arma;

#include "knn.h"
#include "Contour.h"
#include "Contours.h"

Contours::Contours(const fmat& X_, int n_contours_, std::vector<uvec> n_trees_, std::vector<uvec> n_0_, std::string id_) :
X(X_), n_contours(n_contours_), n_trees(n_trees_), n_0(n_0_), id(id_){
    contours = nullptr;
}

// min_S = log_2 of minimum search space size
// max_S = log_2 of maximum search space size
// min_leaf = smallest minimum leaf size n_0 used

Contours::Contours(const fmat& X_, int min_S, int max_S, int min_leaf, std::string id_) : X(X_), id(id_) {
    contours = nullptr;
    n_contours = max_S - min_S + 1;
    n_trees = std::vector<uvec>(n_contours);
    n_0 = std::vector<uvec>(n_contours);

    for (int i = min_S; i <= max_S; i++) {
        uvec temp(i - min_leaf + 1);
        for (int j = 0; j <= i - min_leaf; j++)
            temp[j] = pow(2, j);
        n_trees[i - min_S] = temp;

        int c = 0;
        for (int j = i; j >= min_leaf; j--)
            temp[c++] = pow(2, j);
        n_0[i - min_S] = temp;
    }
}

Contours::~Contours() {
    if (contours)
        for (int i = 0; i < n_contours; i++)
            delete contours[i];
    delete[] contours;
}

void Contours::grow() {
    contours = new Contour*[n_contours];
    for (int i = 0; i < n_contours; i++) {
        contours[i] = new Contour(X, n_trees[i], n_0[i], id + "_" + std::to_string(i));
        contours[i]->grow();
    }
}

void Contours::query(const fmat& Q, int k) {
    int n_points = Q.n_cols;
    umat true_knn(k, n_points);

    std::cout << "Haetaan todelliset naapurit." << std::endl;

    clock_t begin = clock();
    for (int i = 0; i < n_points; i++)
        true_knn.col(i) = knnCppT(X, Q.col(i), k);
    clock_t end = clock();
    time_exact = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);

    std::cout << "Todelliset " << k << " naapuria haettu" << std::endl;

    for (int i = 0; i < n_contours; i++)
        contours[i]->query(Q, k, true_knn);
}

std::vector<Result> Contours::results() {
    std::vector<Result> ret(n_contours);

    for (int i = 0; i < n_contours; i++)
        ret[i] = contours[i]->results(time_exact);
    return ret;
}

void Contours::write_results(std::vector<Result> results, std::string filename) {
    auto end_results = results.end();
    std::ofstream outfile(filename);
    std::vector<std::string> names{"nn_found", "k", "n_trees", "n_0", "times_knn", "times_query", "growing_times",
        "time_exact", "n_points", "times_matrix", "times_trees", "times_total", "times_multi"};

    if (!outfile) {
        std::cerr << "File could not be opened for writing.\n";
        return;
    }

    int n_fields = 13; // # of fields in on Result struct    
    outfile << results.size() << "\n";
    outfile << n_fields << "\n";
       
    for (auto r = results.begin(); r < end_results; ++r) {
        outfile << "nn_found " << r->nn_found.t() << "\n";
        outfile << "k " << r->k << "\n\n";
        outfile << "n_trees " << r->n_trees.t() << "\n";
        outfile << "n_0 " << r->n_0.t() << "\n";
        outfile << "times_knn " << r->times_knn.t() << "\n";
        outfile << "times_query " << r->times_query.t() << "\n";
        outfile << "growing_times " << r->growing_times.t() << "\n";
        outfile << "time_exact " << r->time_exact << "\n\n";
        outfile << "n_points " << r->n_points << "\n\n";
        outfile << "times_matrix " << r->times_matrix.t() << "\n";
        outfile << "times_trees " << r->times_trees.t() << "\n";
        outfile << "times_total " << r->times_total.t() << "\n";
        outfile << "times_multi " << r->times_multi.t() << "\n";
    }
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

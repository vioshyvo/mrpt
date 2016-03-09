#include <iostream>
#include "armadillo"
#include <ctime>

using namespace arma;

#include "Contours.h"
#include "Result.h"
#include "Mrpt.h"
#include "knn.h"



/*******************************************************
 * MRPT algorithm - test code
 * Ville Hyvönen
 * HIIT
 * ville.o.hyvonen<at>helsinki.fi
 * 06.11.2015
 ********************************************************/

arma::mat combine_matrix(std::string filename, int n_chunks) {
  arma::mat X;
  X.load(filename + "1");
  arma::mat X_chunk;
  for(int i = 2; i <= n_chunks; i++) {
    X_chunk.load(filename + std::to_string(i));
    X = arma::join_rows(X, X_chunk);
    std::cout << "chunk: " << i << std::endl;
    std::cout << X_chunk.submat(0,0,4,4) << std::endl;
  }
  return X;
}

int main() {

    // set file paths and names
    std::string id = "mnist128_to_2048";
    std::string tree_path = "../trees/";
    // std::string tree_path = "/home/hyvi/HYVI/git/mrpt/results/trees/";
    std::string result_path = "../../results/cpp/Results_";
    // std::string result_path = "/home/hyvi/HYVI/git/mrpt/results/cpp/Results_";
    std::string data_path = "../../datasets/";
    // std::string data_path = "/home/hyvi/HYVI/git/MRPT/data/mnist/";


    // set parameters
    int min_S = 7;
    int max_S = 11;
    int min_leaf = 3;
    std::vector<int> k = {10, 50, 100};

    // load news data set
//     mat X_double = combine_matrix(data_path + "t_news_", 10);
//     std::cout << "X_double.n_cols: " << X_double.n_cols << ", X_double.n_rows: " << X_double.n_rows  << std::endl;

    // load mnist data set
    mat X_double;
    X_double.load(data_path + "mnist/mnist_arma_t.mat");

    // convert into float matrix
    fmat X_mnist = conv_to<fmat>::from(X_double);
    X_double.reset();

    // load audio data set
    // fmat X_mnist;
    // X_mnist.load(data_path + "audio/arma_audio");

    std::cout << "Ladattu matriisi\n" << X_mnist.submat(0, 0, 4, 4) << std::endl;
    std::cout << "sizeof(X_mnist): " << sizeof(X_mnist) << std::endl;

    // shuffle matrix
    X_mnist = shuffle(X_mnist, 1);

    // split into training and test set
    int n_train = pow(2, floor(log2(X_mnist.n_cols)));
    int n_test = 100;
    fmat X_test = X_mnist.cols(0, n_test - 1);
    fmat X_train = X_mnist.cols(n_test, n_train + n_test - 1);
    X_mnist.reset();

//    umat true_knn(k[0], n_test);
//    std::vector<uvec> approximate_knn(n_test);
//    // std::vector<uvec> index_canditates(n_test);
//    double nn_found = 0;

    std::cout << "X_train.n_cols: " << X_train.n_cols << ", X_test.n_cols: " << X_test.n_cols << ", X_mnist.n_cols: " << X_mnist.n_cols << std::endl;

//    int n_trees = 64;
//    int n_0 = 32;
//    std::string id0 = "single_mnist_64_32";
//
//    std::cout << "X_test.col(0).n_elem: " << X_test.col(0).n_elem <<  ", X_train.n_rows: " << X_train.n_rows << "\n";
//    std::cout << "nearest neighbors of X_test.col(0: ) \n" << knnCppT(X_train, X_test.col(0), k[0]) << std::endl;
//
//    clock_t begin = clock();
//    for (int i = 0; i < n_test; i++) true_knn.col(i) = knnCppT(X_train, X_test.col(i), k[0]);
//    clock_t end = clock();
//    double time_exact = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);
//
//    std::cout << "Exact k-NN time / point: " << 1000 * time_exact / n_test << " ms\n";
//
//    begin = clock();
//    Mrpt* mrpts = nullptr;
//    mrpts = new Mrpt(X_train, n_trees, n_0, id0);
//    mrpts->grow();
//    end = clock();
//    double growing_time = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);
//
//    std::cout << "Growing time: " << growing_time << " s.\n";
//
////    begin = clock();
////        for (int j = 0; j < n_points; j++)
////            approximate_knn[j] = knnCpp_T_indices(X, Q.unsafe_col(j), k, idx_canditates[j]);
////        end = clock();
////        times_knn[i] = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);
//
//    begin = clock();
//    for (int j = 0; j < n_test; j++) approximate_knn[j] = mrpts->query(X_test.unsafe_col(j), k[0]);
//    end = clock();
//    double query_time = (end - begin) / static_cast<double> (CLOCKS_PER_SEC);
//
//    std::cout << "Query time / point: " << 1000 * query_time / n_test << " ms\n";
//
//    uvec test = {1,2,3,4,5};
//    std::cout << "Any: " << any(test == 6) << "\n";
//
//    for (int j = 0; j < n_test; j++) {
//        int n_knn = approximate_knn[j].size();
//        for (int l = 0; l < n_knn; l++)
//            if (any(true_knn.col(j) == approximate_knn[j][l]))
//                nn_found++;
//    }
//
//
//    std::cout << "Approximate k-NN:\n";
//    for(int j = 0; j < 20; j++) std::cout << approximate_knn[j] << "\n";
//
//
//    std::cout << "Recall: " << nn_found / (n_test * k[0]) << "\n";
//
//    delete mrpts;
//    mrpts = nullptr;
//
//    return 0;

    // Grow and save trees
    std::cout << "Nyt mennaan!" << std::endl;
    Contours* contours = new Contours(X_train, min_S, max_S, min_leaf, tree_path + id);
    contours->grow();
    std::cout << "Puut rakennettu." << std::endl;

    // Answer k-NN queries for all values in k[]
    int k_length = k.size();
    for(int j = 0; j < k_length; j++) {
        contours->query(X_test, k[j]);
        std::cout << "Kysely tehty, k = " << k[j] << std::endl;
        std::vector<Result> ret = contours->results();

        for (int i = 0; i < ret.size(); i++) {
            std::cout << "contour: " << i << "\n\n";
            print_result(ret[i]);
        }

        std::string filename = result_path + id + "_k" + std::to_string(k[j]) + ".dat";
        contours->write_results(ret, filename);
    }


    std::cout << "Tulokset haettu" << std::endl;

    delete contours;
    contours = nullptr;
    std::cout << "Puut poistettu." << std::endl;

    return 0;
}



#include <iostream>
#include <armadillo>
#include <ctime>

using namespace arma;

#include "Contours.h"
#include "Result.h"


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
    std::string id = "release_mnist128_to_4096";
    std::string tree_path = "../results/trees/";
    std::string result_path = "../results/cpp/Results_";
    std::string data_path = "../../MRPT/data/mnist/";

    // set parameters
    int min_S = 7;
    int max_S = 12;
    int min_leaf = 3;
    std::vector<int> k = {10, 50, 100};

    // load news data set
    //mat X_double = combine_matrix(data_path + "t_news_", 10);
    //std::cout << "X_double.n_cols: " << X_double.n_cols << ", X_double.n_rows: " << X_double.n_rows  << std::endl;

    // load mnist data set
    mat X_double;
    X_double.load(data_path + "mnist_arma_t.mat");

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

    std::cout << "X_train.n_cols: " << X_train.n_cols << ", X_test.n_cols: " << X_test.n_cols << ", X_mnist.n_cols: " << X_mnist.n_cols << std::endl;

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



# MRPT - fast nearest neighbor search with random projection
![NN search in 3 RP-trees](mrpt-image.png)

NEW! Python 3 bindings available!

MRPT is a library for approximate nearest neighbor search written in C++11. According to [our experiments](https://github.com/ejaasaari/mrpt-comparison/) MRPT is currently the *fastest* alternative to reach high recall levels in common benchmark data sets.

In the offline phase of the algorithm MRPT indexes the data with a collection of *random projection trees*. In the online phase the index structure allows us to answer queries in superior time. A detailed description of the algorithm with the time and space complexities, and the aforementioned comparisons can be found on [our article](https://www.cs.helsinki.fi/u/ttonteri/pub/bigdata2016.pdf) that was published on IEEE International Conference on Big Data 2016.

## Python installation

Install the module with `pip install git+https://github.com/teemupitkanen/mrpt/`

On MacOS, LLVM is needed for compiling: `brew install llvm`

You can now run the demo (runs in less than a minute): `python demo.py`. An example output:
~~~~
Indexing time: 5.993 seconds
100 approximate queries time: 0.230 seconds
100 exact queries time: 11.776 seconds
Average recall: 0.97
~~~~

## Minimal examples

### Python

TODO

### C++

Mrpt is a header only library, so no compilation is required: just include the header `cpp/Mrpt.h`. Only dependencies are Eigen linear algebra library (which is bundled at `cpp/lib`) and OpenMP, so when using g++, the following minimal example can be compiled (add `-std=c++11` if the default for your compiler is not at least c++11) for example as:
```
g++ -Icpp -Icpp/lib ex1.cpp -o ex1 -fopenmp -O3
```

Let's first generate a 200-dimensional data set of 10000 points, and 100 test query points (row = dimension, column = data point). Then `Mrpt::exact_knn` can be used to find the indices of true 10 nearest neighbors of the first test query.

Function `grow` builds an index for approximate k-nn search; it uses automatic parameter tuning, so only the target recall level (90% in this example), the set of test queries and the number of neighbors searched for have to be specified.

```c++
#include <iostream>
#include "Eigen/Dense"
#include "Mrpt.h"

int main() {
  int n = 10000, n_test = 100, d = 200, k = 10;
  double target_recall = 0.9;
  Eigen::MatrixXf X = Eigen::MatrixXf::Random(d, n);
  Eigen::MatrixXf Q = Eigen::MatrixXf::Random(d, n_test);

  Eigen::VectorXi indices(k), indices_exact(k);

  Mrpt::exact_knn(Q.col(0), X, k, indices_exact.data());
  std::cout << indices_exact.transpose() << "\n";

  Mrpt mrpt(X);
  mrpt.grow(target_recall, Q, k);

  mrpt.query(Q.col(0), indices.data());
  std::cout << indices.transpose() << "\n";
}
```

The approximate nearest neighbors are then searched by function `query`; because the index was autotuned, no other arguments than a query point and an output buffer for indices are required.

Here is a sample output:
```
8108 1465 6963 2165   83 5900  662 8112 3592 5505
8108 1465 6963 2165   83 5900  662 8112 5505 7992
```
The approximate nearest neighbor search found 9 of 10 true nearest neighbors; so this time the observed recall happened to match the expected recall exactly (results vary between the runs because the algorithm is randomized).


## MRPT for other languages

- [Go](https://github.com/rikonor/go-ann)

## License

MRPT is available under the MIT License (see LICENSE.txt). Note that third-party libraries in the cpp/lib folder may be distributed under other open source licenses. The Eigen library is licensed under the MPL2.

## Citation
~~~~
@inproceedings{Hyvonen2016,
  title={Fast nearest neighbor search through sparse random projections and voting},
  author={Hyv{\"o}nen, Ville and Pitk{\"a}nen, Teemu and Tasoulis, Sotiris and J{\"a}{\"a}saari, Elias and Tuomainen, Risto and Wang, Liang and Corander, Jukka and Roos, Teemu},
  booktitle={Big Data (Big Data), 2016 IEEE International Conference on},
  pages={881--888},
  year={2016},
  organization={IEEE}
}
~~~~

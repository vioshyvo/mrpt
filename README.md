# MRPT - fast nearest neighbor search with random projection
![NN search in 3 RP-trees](mrpt-image.png)

NEW! Python 3 bindings available!

MRPT is a library for approximate nearest neighbor search written in C++11. According to [our experiments](https://github.com/ejaasaari/mrpt-comparison/) MRPT is currently the *fastest* alternative to reach high recall levels in common benchmark data sets.

In the offline phase of the algorithm MRPT indexes the data with a collection of *random projection trees*. In the online phase the index structure allows us to answer queries in superior time. More technical details can be found in the pre-print version of [our paper](https://arxiv.org/pdf/1509.06957.pdf).


## Python installation

1. [Download](https://github.com/teemupitkanen/mrpt/archive/master.zip) this Github repo and unzip it.
2. Navigate to the `mrpt/cpp` directory.
3. Build the module by issuing command `python setup.py build`.
4. The module file (.so -file) is created in `build/lib*`, with the exact directory names depending on architecture and Python version. 
5. Move the module file to `mrpt/cpp` AND make sure it is named `mrptlib.so`. Rename, if not.
6. You can now run the demo (runs in less than a minute): `python demo.py`. An example output:
~~~~
Indexing time: 5.993 seconds
100 approximate queries time: 0.230 seconds
100 exact queries time: 11.776 seconds
Average recall: 0.97
~~~~

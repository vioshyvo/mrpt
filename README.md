# MRPT - fast nearest neighbor search with random projection
![NN search in 3 RP-trees](mrpt-image.png)

NEW! Python 3 bindings available!

MRPT is a library for approximate nearest neighbor search written in C++11. According to [our experiments](https://github.com/ejaasaari/mrpt-comparison/) MRPT is currently the *fastest* alternative to reach high recall levels in common benchmark data sets.

In the offline phase of the algorithm MRPT indexes the data with a collection of *random projection trees*. In the online phase the index structure allows us to answer queries in superior time. More technical details can be found in the pre-print version of [our paper](https://arxiv.org/pdf/1509.06957.pdf).


## Python installation

Install the module with `pip install git+https://github.com/teemupitkanen/mrpt/`. You can now run the demo (runs in less than a minute): `python demo.py`. An example output:
~~~~
Indexing time: 5.993 seconds
100 approximate queries time: 0.230 seconds
100 exact queries time: 11.776 seconds
Average recall: 0.97
~~~~

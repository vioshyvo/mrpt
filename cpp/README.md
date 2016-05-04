# MRPT in C++11

This directory contains a highly efficient C++ implementation of the MRPT algorithm, and wrappers to use it with Python 2.7.

## Dependencies

Requires [Armadillo](http://arma.sourceforge.net/docs.html). At the time of writing, the module unfortunately only works with Python 2.7. (and C++ naturally.)

## Installation

1. [Download](https://github.com/teemupitkanen/mrpt/archive/master.zip) this Github repo zip and unzip it.
2. Make sure to have [Armadillo](http://arma.sourceforge.net/docs.html) installed.
3. Move to the `mrpt/cpp` directory.
4. Build the module by issuing command `python setup.py build`.
5. Locate the module file (usually named `mrptlib.so`) in the directory structure under `build`.
6. Move the module file to `mrpt/cpp`.

After these steps you should be able to use MRPT in Python by the following code

    import sys
    sys.path.append('<the whole path>/mrpt/cpp')
    from mrpt import MrptIndex  # mrpt.py is an extra layer of Python that actually imports mrptlib
    
    # Usage example
    import numpy as np
    data = np.array(<give a samplesize x dimensionality data set>)
    query = np.array(<give a single query object of the same dimensionality>)

    index = MrptIndex(data, n0=500, n_trees=30)
    neighbors = index.ann(query, 10, n_extra_branches=150, n_elect=200)

## Questions
Feel free to contact: teemu.pitkanen'at'cs.helsinki.fi 

# MRPT in C++11

This directory contains a C++ implementation of the MRPT algorithm, and wrappers to use it with Python 2.7.

## Installation

1. [Download](https://github.com/teemupitkanen/mrpt/archive/master.zip) this Github repo and unzip it.
2. Move to the `mrpt/cpp` directory.
3. Build the module by issuing command `python setup.py build`.
4. Locate the module file (usually named `mrptlib.so`) in the directory structure under `build`.
5. Move the module file to `mrpt/cpp`.

After these steps you should be able to use MRPT in Python by the following code

    import sys
    sys.path.append('<the whole path>/mrpt/cpp')
    from mrpt import MRPTIndex  # mrpt.py is an extra layer of Python that actually imports mrptlib
    
    # Usage example
    import numpy as np
    data = np.array(<give a samplesize x dimensionality data set>)
    query = np.array(<give a single query object of the same dimensionality>)

    index = MRPTIndex(data, depth=8, n_trees=30)
    index.build()
    neighbors = index.ann(query, 10, votes_required=5, n_extra_branches=150)

## Questions
Feel free to contact: teemu.pitkanen'at'cs.helsinki.fi 

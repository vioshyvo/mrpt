import os
import array
import struct
import numpy as np


def ndarray_to_binary(data, out, n=-1):
    """
    Convert a numpy ndarray to binary format.
    :param data: the data as an ndarray
    :param out: path to the output file
    :param n: write only the first n rows, or -1 for all rows
    """
    with open(out, 'wb') as outfile:
        for i, row in enumerate(data):
            if i == n: break
            _write_floats(row.astype(np.float32), outfile)


def csv_to_binary(fname, out, delim=',', n=-1, skip_cols=0):
    """
    Convert a csv file to binary format.
    :param fname: path to the csv file
    :param out: path to the output file
    :param delim: the delimiter used in the csv file
    :param n: write only the first n rows, or -1 for all rows
    :param skip_cols: skip this amount of columns for each row
    """
    import csv

    with open(fname, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=delim)
        with open(out, 'wb') as outfile:
            for i, row in enumerate(datareader):
                if i == n: break
                floats = [float(x) for x in row[skip_cols:]]
                _write_floats(floats, outfile)


def mat_to_binary(fname, out, dataset, n=-1):
    """
    Convert a MAT-file (or any HDF5 file) to binary format.
    Note that the resulting binary file can be much larger than the
    input as the input file could be in compressed format.
    :param fname: path to the MAT-file
    :param out: path to the output file
    :param dataset: the HDF5 dataset to use
    :param n: write only the first n rows, or -1 for all rows
    """
    from tables import open_file # PyTables

    fileh = open_file(fname, "r")
    data = getattr(fileh.root, dataset)

    with open(out, 'wb') as outfile:
        for i, row in enumerate(data.iterrows()):
            if i == n: break
            _write_floats(row, outfile)

    fileh.close()


def rdata_to_binary(fname, out, matrix, n=-1):
    """
    Convert a RData file to binary format.
    :param fname: path to the RData file
    :param out: path to the output file
    :param matrix: name of the matrix being converted in the RData
    :param n: write only the first n rows, or -1 for all rows
    """
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    robjects.r['load'](fname)
    data = robjects.numpy2ri.ri2py(robjects.r[matrix][0])

    with open(out, 'wb') as outfile:
        for i, row in enumerate(data):
            if i == n: break
            _write_floats(row.astype(np.float32), outfile)


def fvecs_to_binary(fname, out, n=-1):
    """
    Convert a fvecs file to binary format.
    The fvecs format is used in e.g. http://corpus-texmex.irisa.fr/
    :param fname: path to the fvecs file
    :param out: path to the output file
    :param matrix: name of the matrix being converted in the RData
    :param n: write only the first n rows, or -1 for all rows
    """
    sz = os.path.getsize(fname)

    with open(fname, 'rb') as inp:
        dim = struct.unpack('i', inp.read(4))[0]

    with open(fname, 'rb') as inp:
        rows = sz / (dim + 1) / 4
        with open(out, 'wb') as outfile:
            for i in xrange(rows):
                if i == n: break
                tmp = struct.unpack('<i', inp.read(4))[0]
                vec = array.array('f')
                vec.read(inp, dim)
                _write_floats(vec, outfile)


def stdin_to_binary(out, delimiter=',', n=-1):
    """
    Write input from stdin to binary format.
    :param out: path to the output file
    :param delimiter: the delimiter used to split the values
    :param n: write only the first n rows, or -1 for all rows
    """
    import sys

    with open(out, 'wb') as outfile:
        for i, row in enumerate(sys.stdin):
            if i == n: break
            sample = [float(x) for x in row.strip().split(delimiter)]
            _write_floats(sample, outfile)


def _write_floats(floats, outfile):
    float_arr = array.array('d', floats)
    s = struct.pack('f' * len(float_arr), *float_arr)
    outfile.write(s)


if __name__ == '__main__':
    fvecs_to_binary('sift_base.fvecs', 'sift_base.fvecs.bin')

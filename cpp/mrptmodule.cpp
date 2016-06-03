/********************************************************************
 * Author: Teemu Henrikki Pitkänen                                  *
 * Email: teemu.pitkanen@cs.helsinki.fi                             *
 * Helsinki Institute for Information Technology (HIIT) 2016        *
 * University of Helsinki, Finland                                  *
 ********************************************************************/

/*
 * This file wraps the C++11 Mrpt code to an extension module compatible with
 * Python 2.7.
 */

#include "Python.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <numeric>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "Mrpt.h"

#include "numpy/arrayobject.h"

#include <Eigen/Dense>

typedef struct {
    PyObject_HEAD
    MrptInterface* ptr;
} mrptIndex;

struct { PyObject *err; std::string str; } errors[] =
{
    {PyExc_IOError, "Could not open input file"},
    {PyExc_ValueError, "Size of the input file is not N * dim"},
    {PyExc_IOError, "Failed to mmap data"},
    {PyExc_MemoryError, "Failed to allocate enough memory"},
    {PyExc_IOError, "Unable to save index"},
    {PyExc_IOError, "Unable to load saved index"},

};

static void handle_exception(int e) {
    PyErr_SetString(errors[e - 1].err, errors[e - 1].str.c_str());
}

static PyObject *Mrpt_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    mrptIndex *self;
    self = (mrptIndex *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ptr = NULL;
    }
    return (PyObject *) self;
}

float *read_memory(char *file, int n, int dim) {
    float *data;
    try {
        data = new float[n * dim];
    } catch (std::bad_alloc &ba) {
        throw 4;
    }

    try {
        std::ifstream in(file, std::ios::in | std::ios::binary);
        in.read(reinterpret_cast<char *>(data), n * dim * sizeof(float));
        in.close();
    } catch (std::exception &e) {
        throw 1;
    }

    return data;
}

float *read_mmap(char *file, int n, int dim) {
    FILE *fd;
    if ((fd = fopen(file, "rb")) == NULL)
        throw 1;

    float *data;
    if ((data = (float *) mmap(0, n * dim * sizeof(float), PROT_READ,
                               MAP_SHARED | MAP_POPULATE, fileno(fd), 0)) == MAP_FAILED) {
        throw 3;
    }

    return data;
}

void check_size(char *file, int n, int dim) {
    struct stat sb;
    if (stat(file, &sb) != 0)
        throw 1;

    if (sb.st_size != sizeof(float) * dim * n) {
        throw 2;
    }
}

static int Mrpt_init(mrptIndex *self, PyObject *args) {
    PyObject *py_data;
    int depth, n_trees, n, dim, mmap, sparse, metric_val;
    float density;

    if (!PyArg_ParseTuple(args, "Oiiiifiii", &py_data, &n, &dim, &depth, &n_trees, &density, &metric_val, &sparse, &mmap))
        return -1;

    float *data;
    if (PyString_Check(py_data)) {
        try {
            char *file = PyString_AsString(py_data);
            check_size(file, n, dim);
            data = mmap ? read_mmap(file, n, dim) : read_memory(file, n, dim);
        } catch (int e) {
            handle_exception(e);
            return -1;
        }
    } else {
        data = (float *) PyArray_DATA(py_data);
    }

    Metric metric = static_cast<Metric>(metric_val);
    if (sparse) {
        SparseMatrix<float> *X = new SparseMatrix<float>(dim, n);
        std::vector<Triplet<float> > triplets;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < dim; ++j) {
                float val = data[i * dim + j];
                if (val != 0) {
                    triplets.push_back(Triplet<float>(j, i, val));
                }
            }
        }

        X->setFromTriplets(triplets.begin(), triplets.end());
        X->makeCompressed();

        self->ptr = new Mrpt<SparseMatrix<float>>(X, n, dim, n_trees, depth, density, metric);
    } else {
        Map<MatrixXf> *X = new Map<MatrixXf>(data, dim, n);
        self->ptr = new Mrpt<Map<MatrixXf>>(X, n, dim, n_trees, depth, density, metric);
    }

    return 0;
}

static PyObject *build(mrptIndex *self) {
    self->ptr->grow();
    Py_RETURN_NONE;
}

static void mrpt_dealloc(mrptIndex *self) {
    if (self->ptr)
        delete self->ptr;
    self->ob_type->tp_free((PyObject*) self);
}

static PyObject *ann(mrptIndex *self, PyObject *args) { 
    PyObject *v;
    int k, elect, branches, dim;

    if (!PyArg_ParseTuple(args, "Oiii", &v, &k, &elect, &branches))
        return NULL;

    dim = PyArray_DIM(v, 0);
    float *indata = (float *) PyArray_DATA(v);

    // create a numpy array to hold the output
    npy_intp dims[1] = {k};
    PyObject *ret = PyArray_SimpleNew(1, dims, NPY_INT);
    int *outdata = (int *) PyArray_DATA(ret);

    self->ptr->query(Eigen::Map<VectorXf>(indata, dim), k, elect, branches, outdata);
    return ret;
}

static PyObject *parallel_ann(mrptIndex *self, PyObject *args) {
    PyObject *v;
    int k, n, elect, branches, dim;

    if (!PyArg_ParseTuple(args, "Oiii", &v, &k, &elect, &branches))
        return NULL;

    n = PyArray_DIM(v, 0);
    dim = PyArray_DIM(v, 1);
    float *indata = (float *) PyArray_DATA(v);

    // create a numpy array to hold the output
    npy_intp dims[2] = {n, k};
    PyObject *ret = PyArray_SimpleNew(2, dims, NPY_INT);
    int *outdata = (int *) PyArray_DATA(ret);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        self->ptr->query(Eigen::Map<VectorXf>(indata + i * dim, dim),
                         k, elect, branches, outdata + i * k);
    }

    return ret;
}

static PyObject *exact_search(mrptIndex *self, PyObject *args) {
    PyObject *v;
    int k, n, dim;

    if (!PyArg_ParseTuple(args, "Oi", &v, &k))
        return NULL;

    n = PyArray_DIM(v, 0);
    dim = PyArray_DIM(v, 1);
    float *indata = (float *) PyArray_DATA(v);

    // create a numpy array to hold the output
    npy_intp dims[2] = {n, k};
    PyObject *ret = PyArray_SimpleNew(2, dims, NPY_INT);
    int *outdata = (int *) PyArray_DATA(ret);

    VectorXi indices(self->ptr->get_n_samples());
    std::iota(indices.data(), indices.data() + self->ptr->get_n_samples(), 0);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        self->ptr->exact_knn(Eigen::Map<VectorXf>(indata + i * dim, dim),
                             k, indices, indices.size(), outdata + i * k);
    }

    return ret;
}

static PyObject *save(mrptIndex *self, PyObject *args) {
    char *fn;

    if (!PyArg_ParseTuple(args, "s", &fn))
        return NULL;

    try {
        self->ptr->save(fn);
    } catch (std::exception &e) {
        handle_exception(5);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *load(mrptIndex *self, PyObject *args) {
    char *fn;

    if (!PyArg_ParseTuple(args, "s", &fn))
        return NULL;

    try {
        self->ptr->load(fn);
    } catch (std::exception &e) {
        handle_exception(6);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef MrptMethods[] = {
    {"ann", (PyCFunction) ann, METH_VARARGS,
            "Return approximate nearest neighbors"},
    {"parallel_ann", (PyCFunction) parallel_ann, METH_VARARGS,
            "Return approximate nearest neighbors calculated in parallel"},
    {"exact_search", (PyCFunction) exact_search, METH_VARARGS,
            "Return exact nearest neighbors"},
    {"build", (PyCFunction) build, METH_VARARGS,
            "Build the index"},
    {"save", (PyCFunction) save, METH_VARARGS,
            "Save the index to a file"},
    {"load", (PyCFunction) load, METH_VARARGS,
            "Load the index from a file"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyTypeObject MrptIndexType = {
    PyObject_HEAD_INIT(NULL)
    0, /*ob_size*/
    "mrpt.MrptIndex", /*tp_name*/
    sizeof (mrptIndex), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor) mrpt_dealloc, /*tp_dealloc*/
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "Mrpt index object", /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    MrptMethods, /* tp_methods */
    0, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc) Mrpt_init, /* tp_init */
    0, /* tp_alloc */
    Mrpt_new, /* tp_new */
};

PyMODINIT_FUNC initmrptlib(void) {
    PyObject *m;
    if (PyType_Ready(&MrptIndexType) < 0)
        return;
    m = Py_InitModule("mrptlib", MrptMethods);

    if (m == NULL)
        return;

    import_array();

    Py_INCREF(&MrptIndexType);
    PyModule_AddObject(m, "MrptIndex", (PyObject*) &MrptIndexType);
}

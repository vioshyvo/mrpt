/*
 * This file wraps the C++11 Mrpt code to an extension module compatible with
 * Python 2.7.
 */

#include "Python.h"
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <sys/types.h>
#include <sys/stat.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include "Mrpt.h"
#include "numpy/arrayobject.h"

#include <Eigen/Dense>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;

typedef struct {
    PyObject_HEAD
    Mrpt *ptr;
    float *data;
    bool mmap;
    int n;
    int dim;
    int k;
} mrptIndex;

static PyObject *Mrpt_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    mrptIndex *self;
    self = reinterpret_cast<mrptIndex *>(type->tp_alloc(type, 0));
    if (self != NULL) {
        self->ptr = NULL;
        self->data = NULL;
    }
    return reinterpret_cast<PyObject *>(self);
}

float *read_memory(char *file, int n, int dim) {
    float *data = new float[n * dim];

    FILE *fd;
    if ((fd = fopen(file, "rb")) == NULL)
        return NULL;

    int read = fread(data, sizeof(float), n * dim, fd);
    if (read != n * dim)
        return NULL;

    fclose(fd);
    return data;
}

#ifndef _WIN32
float *read_mmap(char *file, int n, int dim) {
    FILE *fd;
    if ((fd = fopen(file, "rb")) == NULL)
        return NULL;

    float *data;

    if ((data = reinterpret_cast<float *> (
#ifdef MAP_POPULATE
            mmap(0, n * dim * sizeof(float), PROT_READ,
            MAP_SHARED | MAP_POPULATE, fileno(fd), 0))) == MAP_FAILED) {
#else
            mmap(0, n * dim * sizeof(float), PROT_READ,
            MAP_SHARED, fileno(fd), 0))) == MAP_FAILED) {
#endif
            return NULL;
    }

    fclose(fd);
    return data;
}
#endif

static int Mrpt_init(mrptIndex *self, PyObject *args) {
    PyObject *py_data;
    int n, dim, mmap;

    if (!PyArg_ParseTuple(args, "Oiii", &py_data, &n, &dim, &mmap))
        return -1;

    float *data;
#if PY_MAJOR_VERSION >= 3
    if (PyUnicode_Check(py_data)) {
        char *file = PyBytes_AsString(py_data);
#else
    if (PyString_Check(py_data)) {
        char *file = PyString_AsString(py_data);
#endif

        struct stat sb;
        if (stat(file, &sb) != 0) {
            PyErr_SetString(PyExc_IOError, strerror(errno));
            return -1;
        }

        if (sb.st_size != static_cast<unsigned>(sizeof(float) * dim * n)) {
            PyErr_SetString(PyExc_ValueError, "Size of the input is not N x dim");
            return -1;
        }

#ifndef _WIN32
        data = mmap ? read_mmap(file, n, dim) : read_memory(file, n, dim);
#else
        data = read_memory(file, n, dim);
#endif

        if (data == NULL) {
            PyErr_SetString(PyExc_IOError, "Unable to read data from file or allocate memory for it");
            return -1;
        }

        self->mmap = mmap;
        self->data = data;
    } else {
        data = reinterpret_cast<float *>(PyArray_DATA(py_data));
    }

    self->n = n;
    self->dim = dim;

    self->ptr = new Mrpt(data, dim, n);
    return 0;
}

static PyObject *build(mrptIndex *self, PyObject *args) {
    int n_trees, depth;
    float density;

    if (!PyArg_ParseTuple(args, "iif", &n_trees, &depth, &density))
        return NULL;

    try {
      self->ptr->grow(n_trees, depth, density);
    } catch (const std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }

    Py_RETURN_NONE;
}

static PyObject *build_autotune(mrptIndex *self, PyObject *args) {
    PyObject *py_data;
    float target_recall, density;
    int n_test, k, trees_max, depth_min, depth_max, votes_max;

    if (!PyArg_ParseTuple(args, "fOiiiiiif", &target_recall, &py_data, &n_test, &k, &trees_max,
                                             &depth_min, &depth_max, &votes_max, &density))
        return NULL;

    bool from_mem = false;

    float *data;
#if PY_MAJOR_VERSION >= 3
    if (PyUnicode_Check(py_data)) {
        char *file = PyBytes_AsString(py_data);
#else
    if (PyString_Check(py_data)) {
        char *file = PyString_AsString(py_data);
#endif

        struct stat sb;
        if (stat(file, &sb) != 0) {
            PyErr_SetString(PyExc_IOError, strerror(errno));
            return NULL;
        }

        if (sb.st_size != static_cast<unsigned>(sizeof(float) * self->dim * n_test)) {
            PyErr_SetString(PyExc_ValueError, "Size of the input is not n_test x dim");
            return NULL;
        }

        data = read_memory(file, n_test, self->dim);
        if (data == NULL) {
            PyErr_SetString(PyExc_IOError, "Unable to read data from file or allocate memory for it");
            return NULL;
        }

        from_mem = true;
    } else {
        data = reinterpret_cast<float *>(PyArray_DATA(py_data));
    }

    self->k = k;
    try {
      self->ptr->grow(target_recall, data, n_test, k, trees_max, depth_max, depth_min, votes_max, density);
    } catch (const std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }

    if (from_mem)
      delete[] data;

    Py_RETURN_NONE;
}

static void mrpt_dealloc(mrptIndex *self) {
    if (self->data) {
#ifndef _WIN32
        if (self->mmap)
            munmap(self->data, self->n * self->dim * sizeof(float));
        else
#endif
            delete[] self->data;
    }
    if (self->ptr)
        delete self->ptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *ann(mrptIndex *self, PyObject *args) {
    PyObject *v;
    int k, elect, dim, n, return_distances;

    if (!PyArg_ParseTuple(args, "Oiii", &v, &k, &elect, &return_distances))
        return NULL;

    float *indata = reinterpret_cast<float *>(PyArray_DATA(v));
    PyObject *nearest;

    if (k == -1)
      k = self->k;

    if (PyArray_NDIM(v) == 1) {
        dim = PyArray_DIM(v, 0);

        npy_intp dims[1] = {k};
        nearest = PyArray_SimpleNew(1, dims, NPY_INT);
        int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));

        if (return_distances) {
            PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            float *out_distances = reinterpret_cast<float *>(PyArray_DATA(distances));

            if (k == -1) {
              self->ptr->query(indata, outdata, out_distances);
            } else {
              self->ptr->query(indata, k, elect, outdata, out_distances);
            }

            PyObject *out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            if (k == -1) {
              self->ptr->query(indata, outdata);
            } else {
              self->ptr->query(indata, k, elect, outdata);
            }
            return nearest;
        }
    } else {
        n = PyArray_DIM(v, 0);
        dim = PyArray_DIM(v, 1);

        npy_intp dims[2] = {n, k};
        nearest = PyArray_SimpleNew(2, dims, NPY_INT);
        int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));

        if (return_distances) {
            npy_intp dims[2] = {n, k};
            PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            float *distances_out = reinterpret_cast<float *>(PyArray_DATA(distances));

            if (k == -1) {
              for (int i = 0; i < n; ++i) {
                  self->ptr->query(indata + i * dim, outdata + i * k, distances_out + i * k);
              }
            } else {
              for (int i = 0; i < n; ++i) {
                  self->ptr->query(indata + i * dim, k, elect, outdata + i * k, distances_out + i * k);
              }
            }

            PyObject *out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            if (k == -1) {
              for (int i = 0; i < n; ++i) {
                  self->ptr->query(indata + i * dim, outdata + i * k);
              }
            } else {
              for (int i = 0; i < n; ++i) {
                  self->ptr->query(indata + i * dim, k, elect, outdata + i * k);
              }
            }
            return nearest;
        }
    }
}

static PyObject *exact_search(mrptIndex *self, PyObject *args) {
    PyObject *v;
    int k, n, dim, return_distances;

    if (!PyArg_ParseTuple(args, "Oii", &v, &k, &return_distances))
        return NULL;

    float *indata = reinterpret_cast<float *>(PyArray_DATA(v));
    PyObject *nearest;

    if (PyArray_NDIM(v) == 1) {
        dim = PyArray_DIM(v, 0);

        npy_intp dims[1] = {k};
        nearest = PyArray_SimpleNew(1, dims, NPY_INT);
        int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));

        if (return_distances) {
            PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            float *out_distances = reinterpret_cast<float *>(PyArray_DATA(distances));
            self->ptr->exact_knn(indata, k, outdata, out_distances);

            PyObject *out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            self->ptr->exact_knn(indata, k, outdata);
            return nearest;
        }
    } else {
        n = PyArray_DIM(v, 0);
        dim = PyArray_DIM(v, 1);

        npy_intp dims[2] = {n, k};
        nearest = PyArray_SimpleNew(2, dims, NPY_INT);
        int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));

        if (return_distances) {
            npy_intp dims[2] = {n, k};
            PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            float *distances_out = reinterpret_cast<float *>(PyArray_DATA(distances));

            for (int i = 0; i < n; ++i) {
                self->ptr->exact_knn(indata + i * dim, k, outdata + i * k, distances_out + i * k);
            }
            PyObject *out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            for (int i = 0; i < n; ++i) {
                self->ptr->exact_knn(indata + i * dim, k, outdata + i * k);
            }
            return nearest;
        }
    }
}

static PyObject *save(mrptIndex *self, PyObject *args) {
    char *fn;

    if (!PyArg_ParseTuple(args, "s", &fn))
        return NULL;

    if (!self->ptr->save(fn)) {
        PyErr_SetString(PyExc_IOError, "Unable to save index to file");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *load(mrptIndex *self, PyObject *args) {
    char *fn;

    if (!PyArg_ParseTuple(args, "s", &fn))
        return NULL;

    if (!self->ptr->load(fn)) {
        PyErr_SetString(PyExc_IOError, "Unable to load index from file");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef MrptMethods[] = {
    {"ann", (PyCFunction) ann, METH_VARARGS,
            "Return approximate nearest neighbors"},
    {"exact_search", (PyCFunction) exact_search, METH_VARARGS,
            "Return exact nearest neighbors"},
    {"build", (PyCFunction) build, METH_VARARGS,
            "Build the index"},
    {"build_autotune", (PyCFunction) build_autotune, METH_VARARGS,
            "Build the index using autotuning"},
    {"save", (PyCFunction) save, METH_VARARGS,
            "Save the index to a file"},
    {"load", (PyCFunction) load, METH_VARARGS,
            "Load the index from a file"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyTypeObject MrptIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "mrpt.MrptIndex", /*tp_name*/
    sizeof(mrptIndex), /*tp_basicsize*/
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

static PyMethodDef module_methods[] = {
  {NULL}	/* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mrptlib",          /* m_name */
    "",                  /* m_doc */
    -1,                  /* m_size */
    module_methods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };

PyMODINIT_FUNC PyInit_mrptlib(void) {
    PyObject *m;
    if (PyType_Ready(&MrptIndexType) < 0)
        return NULL;

    m = PyModule_Create(&moduledef);

    if (m == NULL)
        return NULL;

    import_array();

    Py_INCREF(&MrptIndexType);
    PyModule_AddObject(m, "MrptIndex", reinterpret_cast<PyObject *>(&MrptIndexType));

    return m;
}
#else
PyMODINIT_FUNC initmrptlib(void) {
    PyObject *m;
    if (PyType_Ready(&MrptIndexType) < 0)
        return;

    m = Py_InitModule("mrptlib", module_methods);

    if (m == NULL)
        return;

    import_array();

    Py_INCREF(&MrptIndexType);
    PyModule_AddObject(m, "MrptIndex", reinterpret_cast<PyObject *>(&MrptIndexType));
}
#endif

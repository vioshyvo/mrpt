/*
 * This file wraps the C++11 Mrpt code to an extension module compatible with
 * Python 3.
 */

#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "Python.h"

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <Eigen/Dense>

#include "Mrpt.h"
#include "numpy/arrayobject.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;

typedef struct {
  PyObject_HEAD Mrpt *index;
  PyObject *py_data;
  float *data;
  int *subset_refs;
  bool mmap;
  int n;
  int dim;
  int k;
} mrptIndex;

static PyObject *Mrpt_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  mrptIndex *self = reinterpret_cast<mrptIndex *>(type->tp_alloc(type, 0));

  if (self != NULL) {
    self->index = NULL;
    self->data = NULL;
    self->py_data = NULL;
    self->subset_refs = new int(1);
  }

  return reinterpret_cast<PyObject *>(self);
}

float *read_memory(char *file, int n, int dim) {
  FILE *fd;
  if ((fd = fopen(file, "rb")) == NULL) {
    return NULL;
  }

  float *data = new float[n * dim];

  if (data == NULL) {
    fclose(fd);
    return NULL;
  }

  int read = fread(data, sizeof(float), n * dim, fd);
  fclose(fd);

  if (read != n * dim) {
    delete[] data;
    return NULL;
  }

  return data;
}

#ifndef _WIN32
float *read_mmap(char *file, int n, int dim) {
  FILE *fd;
  if ((fd = fopen(file, "rb")) == NULL) return NULL;

  float *data;

  if ((data = reinterpret_cast<float *>(
#ifdef MAP_POPULATE
           mmap(0, n * dim * sizeof(float), PROT_READ, MAP_SHARED | MAP_POPULATE, fileno(fd),
                0))) == MAP_FAILED) {
#else
           mmap(0, n * dim * sizeof(float), PROT_READ, MAP_SHARED, fileno(fd), 0))) == MAP_FAILED) {
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

  if (!PyArg_ParseTuple(args, "Oiii", &py_data, &n, &dim, &mmap)) return -1;

  float *data;
  if (PyUnicode_Check(py_data)) {
    char *file = PyBytes_AsString(py_data);

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
    data = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)py_data));
    self->py_data = py_data;
    Py_XINCREF(self->py_data);
  }

  self->n = n;
  self->dim = dim;
  self->index = new Mrpt(data, dim, n);

  return 0;
}

static PyObject *build(mrptIndex *self, PyObject *args) {
  int n_trees, depth;
  float density;

  if (!PyArg_ParseTuple(args, "iif", &n_trees, &depth, &density)) return NULL;

  try {
    Py_BEGIN_ALLOW_THREADS;
    self->index->grow(n_trees, depth, density);
    Py_END_ALLOW_THREADS;

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
  if (PyUnicode_Check(py_data)) {
    char *file = PyBytes_AsString(py_data);

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
    data = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)py_data));
  }

  self->k = k;

  try {
    if (target_recall < 0) {
      Py_BEGIN_ALLOW_THREADS;
      self->index->grow(data, n_test, k, trees_max, depth_max, depth_min, votes_max, density);
      Py_END_ALLOW_THREADS;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      self->index->grow(target_recall, data, n_test, k, trees_max, depth_max, depth_min, votes_max,
                        density);
      Py_END_ALLOW_THREADS;
    }
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }

  if (from_mem) delete[] data;

  Py_RETURN_NONE;
}

static PyObject *build_autotune_sample(mrptIndex *self, PyObject *args) {
  float target_recall, density;
  int n_test, k, trees_max, depth_min, depth_max, votes_max;

  if (!PyArg_ParseTuple(args, "fiiiiiif", &target_recall, &n_test, &k, &trees_max, &depth_min,
                        &depth_max, &votes_max, &density))
    return NULL;

  self->k = k;

  try {
    if (target_recall < 0) {
      Py_BEGIN_ALLOW_THREADS;
      self->index->grow_autotune(k, trees_max, depth_max, depth_min, votes_max, density, 0, n_test);
      Py_END_ALLOW_THREADS;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      self->index->grow_autotune(target_recall, k, trees_max, depth_max, depth_min, votes_max,
                                 density, 0, n_test);
      Py_END_ALLOW_THREADS;
    }
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }

  Py_RETURN_NONE;
}

static void mrpt_dealloc(mrptIndex *self) {
  (*self->subset_refs)--;
  if (!(*self->subset_refs)) {
    if (self->data) {
#ifndef _WIN32
      if (self->mmap)
        munmap(self->data, self->n * self->dim * sizeof(float));
      else
#endif
        delete[] self->data;

      self->data = NULL;
    }

    delete self->subset_refs;
    self->subset_refs = NULL;
  }

  if (self->index) {
    delete self->index;
    self->index = NULL;
  }

  Py_XDECREF(self->py_data);
  self->py_data = NULL;

  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *ann(mrptIndex *self, PyObject *args) {
  PyArrayObject *v;
  int k, elect, dim, n, return_distances;

  if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &v, &k, &elect, &return_distances))
    return NULL;

  float *indata = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)v));
  PyObject *nearest;

  if (k < 0) k = self->k;
  if (elect < 0) {
    elect = self->index->parameters().votes;
  }

  if (PyArray_NDIM(v) == 1) {
    dim = PyArray_DIM(v, 0);

    npy_intp dims[1] = {k};
    nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
      float *out_distances = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));
      Py_BEGIN_ALLOW_THREADS;
      self->index->query(indata, k, elect, outdata, out_distances);
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      self->index->query(indata, k, elect, outdata);
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  } else {
    n = PyArray_DIM(v, 0);
    dim = PyArray_DIM(v, 1);

    npy_intp dims[2] = {n, k};
    nearest = PyArray_SimpleNew(2, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      npy_intp dims[2] = {n, k};
      PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
      float *distances_out = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));

      Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < n; ++i) {
        self->index->query(indata + i * dim, k, elect, outdata + i * k, distances_out + i * k);
      }
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      for (int i = 0; i < n; ++i) {
        self->index->query(indata + i * dim, k, elect, outdata + i * k);
      }
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  }
}

static PyObject *exact_search(mrptIndex *self, PyObject *args) {
  PyArrayObject *v;
  int k, n, dim, return_distances;

  if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &v, &k, &return_distances)) return NULL;

  float *indata = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)v));
  PyObject *nearest;

  if (PyArray_NDIM(v) == 1) {
    dim = PyArray_DIM(v, 0);

    npy_intp dims[1] = {k};
    nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
      float *out_distances = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));
      Py_BEGIN_ALLOW_THREADS;
      self->index->exact_knn(indata, k, outdata, out_distances);
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      self->index->exact_knn(indata, k, outdata);
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  } else {
    n = PyArray_DIM(v, 0);
    dim = PyArray_DIM(v, 1);

    npy_intp dims[2] = {n, k};
    nearest = PyArray_SimpleNew(2, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      npy_intp dims[2] = {n, k};
      PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
      float *distances_out = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));

      Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < n; ++i) {
        self->index->exact_knn(indata + i * dim, k, outdata + i * k, distances_out + i * k);
      }
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      for (int i = 0; i < n; ++i) {
        self->index->exact_knn(indata + i * dim, k, outdata + i * k);
      }
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  }
}

static PyObject *subset(mrptIndex *self, PyObject *args) {
  float target_recall;

  if (!PyArg_ParseTuple(args, "f", &target_recall)) return NULL;

  mrptIndex *new_idx = (mrptIndex *)PyObject_New(mrptIndex, Py_TYPE(self));
  new_idx = (mrptIndex *)PyObject_Init((PyObject *)new_idx, Py_TYPE(self));

  new_idx->data = self->data;
  new_idx->subset_refs = self->subset_refs;
  (*new_idx->subset_refs)++;
  new_idx->mmap = self->mmap;
  new_idx->n = self->n;
  new_idx->dim = self->dim;
  new_idx->k = self->k;
  new_idx->py_data = self->py_data;
  Py_XINCREF(new_idx->py_data);

  new_idx->index = self->index->subset_pointer(target_recall);
  return reinterpret_cast<PyObject *>(new_idx);
}

static PyObject *parameters(mrptIndex *self, PyObject *args) {
  Mrpt_Parameters par = self->index->parameters();

  PyObject *tup = PyTuple_New(6);
  PyTuple_SetItem(tup, 0, PyLong_FromLong(par.n_trees));
  PyTuple_SetItem(tup, 1, PyLong_FromLong(par.depth));
  PyTuple_SetItem(tup, 2, PyLong_FromLong(par.votes));
  PyTuple_SetItem(tup, 3, PyLong_FromLong(par.k));
  PyTuple_SetItem(tup, 4, PyFloat_FromDouble(par.estimated_qtime));
  PyTuple_SetItem(tup, 5, PyFloat_FromDouble(par.estimated_recall));

  return tup;
}

static PyObject *save(mrptIndex *self, PyObject *args) {
  char *fn;

  if (!PyArg_ParseTuple(args, "s", &fn)) return NULL;

  if (!self->index->save(fn)) {
    PyErr_SetString(PyExc_IOError, "Unable to save index to file");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *load(mrptIndex *self, PyObject *args) {
  char *fn;

  if (!PyArg_ParseTuple(args, "s", &fn)) return NULL;

  if (!self->index->load(fn)) {
    PyErr_SetString(PyExc_IOError, "Unable to load index from file");
    return NULL;
  }

  self->k = self->index->parameters().k;

  Py_RETURN_NONE;
}

static PyObject *is_autotuned(mrptIndex *self, PyObject *args) {
  if (self->index->is_autotuned())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyMethodDef MrptMethods[] = {
    {"ann", (PyCFunction)ann, METH_VARARGS, "Return approximate nearest neighbors"},
    {"exact_search", (PyCFunction)exact_search, METH_VARARGS, "Return exact nearest neighbors"},
    {"build", (PyCFunction)build, METH_VARARGS, "Build the index"},
    {"build_autotune", (PyCFunction)build_autotune, METH_VARARGS,
     "Build the index using autotuning"},
    {"build_autotune_sample", (PyCFunction)build_autotune_sample, METH_VARARGS,
     "Build the index using autotuning"},
    {"subset", (PyCFunction)subset, METH_VARARGS, "Subset the trees"},
    {"parameters", (PyCFunction)parameters, METH_VARARGS, "Get the parameters of the index"},
    {"save", (PyCFunction)save, METH_VARARGS, "Save the index to a file"},
    {"load", (PyCFunction)load, METH_VARARGS, "Load the index from a file"},
    {"is_autotuned", (PyCFunction)is_autotuned, METH_VARARGS,
     "Get whether the index has been autotuned"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyTypeObject MrptIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) "mrpt.MrptIndex", /*tp_name*/
    sizeof(mrptIndex),                               /*tp_basicsize*/
    0,                                               /*tp_itemsize*/
    (destructor)mrpt_dealloc,                        /*tp_dealloc*/
    0,                                               /*tp_print*/
    0,                                               /*tp_getattr*/
    0,                                               /*tp_setattr*/
    0,                                               /*tp_compare*/
    0,                                               /*tp_repr*/
    0,                                               /*tp_as_number*/
    0,                                               /*tp_as_sequence*/
    0,                                               /*tp_as_mapping*/
    0,                                               /*tp_hash */
    0,                                               /*tp_call*/
    0,                                               /*tp_str*/
    0,                                               /*tp_getattro*/
    0,                                               /*tp_setattro*/
    0,                                               /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                              /*tp_flags*/
    "Mrpt index object",                             /* tp_doc */
    0,                                               /* tp_traverse */
    0,                                               /* tp_clear */
    0,                                               /* tp_richcompare */
    0,                                               /* tp_weaklistoffset */
    0,                                               /* tp_iter */
    0,                                               /* tp_iternext */
    MrptMethods,                                     /* tp_methods */
    0,                                               /* tp_members */
    0,                                               /* tp_getset */
    0,                                               /* tp_base */
    0,                                               /* tp_dict */
    0,                                               /* tp_descr_get */
    0,                                               /* tp_descr_set */
    0,                                               /* tp_dictoffset */
    (initproc)Mrpt_init,                             /* tp_init */
    0,                                               /* tp_alloc */
    Mrpt_new,                                        /* tp_new */
};

static PyMethodDef module_methods[] = {
    {NULL} /* Sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mrptlib",      /* m_name */
    "",             /* m_doc */
    -1,             /* m_size */
    module_methods, /* m_methods */
    NULL,           /* m_reload */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_mrptlib(void) {
  PyObject *m;
  if (PyType_Ready(&MrptIndexType) < 0) return NULL;

  m = PyModule_Create(&moduledef);

  if (m == NULL) return NULL;

  import_array();

  Py_INCREF(&MrptIndexType);
  PyModule_AddObject(m, "MrptIndex", reinterpret_cast<PyObject *>(&MrptIndexType));

  return m;
}

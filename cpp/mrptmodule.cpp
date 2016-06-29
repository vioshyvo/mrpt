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
#include <cstdlib>
#include <armadillo>
#include "Mrpt.h"


typedef struct {
    PyObject_HEAD
    Mrpt* ptr;
} mrptIndex;

static PyObject *
Mrpt_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    mrptIndex* self;
    self = (mrptIndex*) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ptr = NULL;
    }
    return (PyObject *) self;
}

static int
Mrpt_init(mrptIndex *self, PyObject *args, PyObject *kwds) {
    PyObject* py_data = NULL;
    int depth, n_trees, n, dim;
    if (!PyArg_ParseTuple(args, "Oii", &py_data, &depth, &n_trees))
        return -1;
    
    if (PyString_Check(py_data)){
        // Load the data matrix from file
        self->ptr = new Mrpt(std::string(PyString_AsString(py_data)), n_trees, depth, "genericTreeID");
    } else {
        // Load the data matrix from a nested python list
        n = PyList_Size(py_data);
        dim = PyList_Size(PyList_GetItem(py_data, 0));
        arma::fmat X(dim, n);
        for (int i = 0; i < dim; i++){
            for (int j = 0; j < n; j++){
                X(i, j) = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(py_data, j), i));
            }
        }
        self->ptr = new Mrpt(X, n_trees, depth, "genericTreeID");
    }
    self->ptr->grow();
    return 0;
}

static void
mrpt_dealloc(mrptIndex* self) {
    if (self->ptr) {
        delete self->ptr;
    }
    self->ob_type->tp_free((PyObject*) self);
}

static PyObject* ann(mrptIndex* self, PyObject* args) { 
    PyObject* v;
    int k, elect, branches, dim;
    if (!PyArg_ParseTuple(args, "Oiii", &v, &k, &elect, &branches))
        return NULL;
    dim = PyList_Size(v);
    arma::fvec w(dim);
    for (int i = 0; i < dim; i++){
        PyObject* elem = PyList_GetItem(v, i);
        w[i] = PyFloat_AsDouble(elem);
    }
    arma::uvec neighbors = self->ptr->query(w, k, elect, branches);
    
    PyObject* l = PyList_New(k);
    for (size_t i = 0; i < k; i++)
        PyList_SetItem(l, i, PyInt_FromLong(neighbors[i]));
    return l;
}

static PyMethodDef MrptMethods[] = {
    {"ann", (PyCFunction) ann, METH_VARARGS, 
            "Return approximate nearest neighbors"},
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

PyMODINIT_FUNC
initmrptlib(void) {
    PyObject* m;
    if (PyType_Ready(&MrptIndexType) < 0)
        return;
    m = Py_InitModule("mrptlib", MrptMethods);
    
    if (m == NULL)
        return;
    
    Py_INCREF(&MrptIndexType);
    PyModule_AddObject(m, "MrptIndex", (PyObject*)&MrptIndexType);
}
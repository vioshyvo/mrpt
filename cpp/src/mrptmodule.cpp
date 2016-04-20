/*
 Author: Teemu Henrikki Pitkänen
 Email: teemu.pitkanen@cs.helsinki.fi
 Helsinki Institute for Information Technology (HIIT) 2016
 University of Helsinki, Finland
 */


#include "Python/Python.h"
#include "Python/structmember.h"
#include <cstdlib>
#include <armadillo>
#include "mrpt.h"
#include "knn.h"


typedef struct {
    PyObject_HEAD
    Mrpt* ptr;
} mrptIndex;


static PyObject *
Mrpt_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    mrptIndex* self;
    self = (mrptIndex*) type->tp_alloc(type, 0);
    if (self != NULL) {
        //self->f = 0;
        self->ptr = NULL;
    }
    return (PyObject *) self;
}


static int
Mrpt_init(mrptIndex *self, PyObject *args, PyObject *kwds) {
    PyObject* data = NULL;
    int n0, n_trees, n, dim;
    if (!PyArg_ParseTuple(args, "Oii", &data, &n0, &n_trees))
        return -1;
    n = PyList_Size(data);
    dim = PyList_Size(PyList_GetItem(data, 0));
    arma::fmat X();
    self->ptr = new Mrpt(X, n_trees, n0, "genericTreeID");
    return 0;
}


static void
mrpt_dealloc(mrptIndex* self) {
    if (self->ptr) {
        delete self->ptr;
    }
    self->ob_type->tp_free((PyObject*) self);
}

// Copy-paste from the dummy implementation. Guaranteed to NOT work for now!!!
static PyObject* ann(PyObject* self, PyObject* args) { 
    PyObject* v;
    int k;
    if (!PyArg_ParseTuple(args, "Oi", &v, &k))
        return NULL;
    arma::fvec w(5);
    for (int i = 0; i < 5; i++){
        PyObject* elem = PyList_GetItem(v, i);
        
        w[i] = PyFloat_AsDouble(elem);
    }
    arma::uvec neighbors = query(w, 10); // <----- Fix the rest from here on
    
    PyObject* l = PyList_New(10);
    for (size_t i = 0; i < 10; i++)
        PyList_SetItem(l, i, PyInt_FromLong(neighbors[i]));
    return l;
}


static PyMethodDef MrptMethods[] = {
    {"ann", (PyCFunction) ann, METH_VARARGS, "Return approximate nearest neighbors"},
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
initmrpt(void) {
    PyObject* m;
    (void) Py_InitModule("mrpt", MrptMethods);
}
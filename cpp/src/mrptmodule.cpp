#include "Python/Python.h"
#include <cstdlib>
#include <armadillo>
#include <Python/structmember.h>
#include "mrpt.h"
#include "knn.h"

typedef struct {
    PyObject_HEAD
    Mrpt* ptr;
} mrptIndex;

static PyObject *
Mrpt_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  mrptIndex* self;

  self = (mrptIndex*)type->tp_alloc(type, 0);
  if (self != NULL) {
    //self->f = 0;
    self->ptr = NULL;
  }

  return (PyObject *)self;
}

// Copy-paste from noddy reference implementation. FIX!
static int
Mrpt_init(mrptIndex *self, PyObject *args, PyObject *kwds)
{
    PyObject *first=NULL, *last=NULL, *tmp;

    static char *kwlist[] = {"first", "last", "number", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|OOi", kwlist, 
                                      &first, &last, 
                                      &self->number))
        return -1; 

    if (first) {
        tmp = self->first;
        Py_INCREF(first);
        self->first = first;
        Py_XDECREF(tmp);
    }

    if (last) {
        tmp = self->last;
        Py_INCREF(last);
        self->last = last;
        Py_XDECREF(tmp);
    }

    return 0;
}


PyMODINIT_FUNC
initmrpt(void)
{   
    PyObject* m;
    (void) Py_InitModule("mrpt", MrptMethods);
}

static PyMethodDef MrptMethods[] = {
  {NULL, NULL, 0, NULL}		 /* Sentinel */
};

static PyTypeObject MrptIndexType={
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "mrpt.MrptIndex",          /*tp_name*/
    sizeof(mrptIndex),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    "Mrpt index object",       /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MrptMethods,               /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Mrpt_init,       /* tp_init */
    0,                         /* tp_alloc */
    Mrpt_new,                  /* tp_new */
};

static PyObject* ann(PyObject* self, PyObject* args){
    
}
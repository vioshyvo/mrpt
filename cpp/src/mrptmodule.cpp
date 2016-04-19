#include "Python/Python.h"
#include <cstdlib>
#include <armadillo>
#include "mrpt.h"
#include "knn.h"

//static PyObject* query(PyObject* self, PyObject* args){
//    PyObject* v;
//    int k;
//    if (!PyArg_ParseTuple(args, "Oi", &v, &k))
//        return NULL;
//    
//    arma::fvec w(5);
//    for (int i = 0; i < 5; i++){
//        PyObject* elem = PyList_GetItem(v, i);
//        
//        w[i] = PyFloat_AsDouble(elem);
//    }
//    arma::uvec neighbors = ann(w, 10);
//    
//    PyObject* l = PyList_New(10);
//    for (size_t i = 0; i < 10; i++)
//        PyList_SetItem(l, i, PyInt_FromLong(neighbors[i]));
//    return l;
//}
//
//static PyMethodDef MrptMethods[] = {
//    {"ann",  query, METH_VARARGS, ""},
//    {NULL, NULL, 0, NULL} /* Sentinel*/
//};
//
//PyMODINIT_FUNC
//initmrpt(void)
//{
//    (void) Py_InitModule("mrpt", MrptMethods);
//}


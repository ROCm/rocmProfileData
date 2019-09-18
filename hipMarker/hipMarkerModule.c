#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "roctracer_ext.h"


static PyObject *hipMarker_emitMarker(PyObject *self, PyObject *args)
{
    const char *eventString = "";
    if (PyArg_ParseTuple(args, "s", &eventString)) {
        roctracer_add_user_event(eventString);
        printf("MARKER: %s\n", eventString);
    }
    Py_INCREF(Py_None);
    return Py_None;
};

static PyMethodDef hipMarkerMethods[] = {
    {"emitMarker", hipMarker_emitMarker, METH_VARARGS, "Insert a hip user marker"}
    , {NULL, NULL, 0, NULL}
};

static struct PyModuleDef hipMarkermodule = {
    PyModuleDef_HEAD_INIT,
    "hipMarker",
    NULL,
    -1,
    hipMarkerMethods
};

PyMODINIT_FUNC PyInit_hipMarker(void)
{
    return PyModule_Create(&hipMarkermodule);
}


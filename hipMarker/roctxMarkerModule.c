#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "roctracer/roctx.h"


static PyObject *roctxMarker_emitMarker(PyObject *self, PyObject *args)
{
    const char *eventString = "";
    if (PyArg_ParseTuple(args, "s", &eventString)) {
	roctxMarkA(eventString);
        //printf("EMIT: %s\n", eventString);
    }
    Py_INCREF(Py_None);
    return Py_None;
};

static PyObject *roctxMarker_pushMarker(PyObject *self, PyObject *args)
{
    const char *eventString = "";
    if (PyArg_ParseTuple(args, "s", &eventString)) {
        roctxRangePushA(eventString);
        //printf("PUSH: %s\n", eventString);
    }
    Py_INCREF(Py_None);
    return Py_None;
};

static PyObject *roctxMarker_popMarker(PyObject *self, PyObject *args)
{
    roctxRangePop();
    //printf("POP:\n");
    Py_INCREF(Py_None);
    return Py_None;
};

static PyMethodDef roctxMarkerMethods[] = {
    {"emitMarker", roctxMarker_emitMarker, METH_VARARGS, "Insert a roxtx marker"}
    , {"pushMarker", roctxMarker_pushMarker, METH_VARARGS, "Start a roxtx range"}
    , {"popMarker", roctxMarker_popMarker, METH_VARARGS, "End most recent roxtx range"}
    , {NULL, NULL, 0, NULL}
};

static struct PyModuleDef roctxMarkermodule = {
    PyModuleDef_HEAD_INIT,
    "roctxMarker",
    NULL,
    -1,
    roctxMarkerMethods
};

PyMODINIT_FUNC PyInit_roctxMarker(void)
{
    return PyModule_Create(&roctxMarkermodule);
}


#include <Python.h>
#include <stdlib.h>
#include <vector>
#include "cppjieba/MixSegment.hpp"

using namespace std;
using namespace CppJieba;

PyObject *wrap_cppjieba_init(PyObject *self, PyObject *args)
{
    MixSegment *seg_hd;
    const char *dict_path;
    const char *hmm_path;
    if (!PyArg_ParseTuple(args, "ss", &dict_path, &hmm_path)) {
        return Py_BuildValue("K", 0);
    }
    
    LogInfo("test log info");
    seg_hd = new MixSegment(dict_path, hmm_path);
    
    return Py_BuildValue("K", seg_hd);
}

PyObject *wrap_cppjieba_close(PyObject *self, PyObject *args)
{
    MixSegment *seg_hd;
    unsigned long long seg_hdv;
    if (!PyArg_ParseTuple(args, "K", &seg_hdv)) {
        return Py_BuildValue("i", 0);
    }

    if (!seg_hdv) {
        return Py_BuildValue("i", 0);
    }

    seg_hd = (MixSegment *)seg_hdv;
    
    delete seg_hd;
    
    return Py_BuildValue("i", 0);
}

PyObject *wrap_cppjieba_cut(PyObject *self, PyObject *args)
{
    MixSegment *seg_hd;
    unsigned long long seg_hdv;
    const char *s;
    if (!PyArg_ParseTuple(args, "Ks", &seg_hdv, &s)) {
        return Py_BuildValue("[]");
    }

    if (!seg_hdv) {
        return Py_BuildValue("[]");
    }

    seg_hd = (MixSegment *)seg_hdv;

    vector<string> word_list;
    seg_hd->cut(s, word_list);

    int word_list_size = (int)word_list.size();

    PyObject *ret_list;

    ret_list = PyList_New(word_list_size);

    for(int i = 0; i < word_list_size; i++) {
        PyList_SET_ITEM(ret_list, i, Py_BuildValue("s", word_list[i].c_str()));
    }

    return ret_list;
}

//短字符串在前
PyObject *wrap_distance_compare(PyObject *self, PyObject *args)
{
    const char *s;
    const char *t;
    int slen;
    int tlen;
    int *v0;
    int *v1;
    int i, j;
    int cost = 0;
    int a,b,c;
    int ret;

    if (!PyArg_ParseTuple(args, "ssii", &s, &t, &slen, &tlen)) {
        return Py_BuildValue("i", 0);
    }

    if (slen == 0) {
        return Py_BuildValue("i", tlen);
    }

    if (tlen == 0) {
        return Py_BuildValue("i", slen);
    }

    v0 = (int *)malloc(sizeof(int) * (tlen + 1));

    for (i = 0; i < tlen + 1; i++) {
        v0[i] = i;
    }

    v1 = (int *)calloc(sizeof(int), (tlen + 1));

    for (i = 1; i < slen + 1; i++) {
        v1[0] = i;
        for (j = 1; j < tlen + 1; j++) {
            if (s[i - 1] == t[j - 1]) {
                cost = 0;
            } else {
                cost = 1;
            }

            a = v0[j] + 1;
            b = v1[j - 1] + 1;
            c = v0[j - 1] + cost;
            if (a > b) {
                a = b;
            }

            if (a > c) {
                a = c;
            }

            v1[j] = a;
        }
        memcpy(v0, v1, sizeof(int) * tlen);
    }

    ret = v1[tlen];
    free(v0);
    free(v1);

    return Py_BuildValue("i", ret);
}

static PyMethodDef tf_idf_tk_methods[] = 
{
    {"distance_compare" , wrap_distance_compare , METH_VARARGS , "distance_compare"} ,
    {"cppjieba_init"    , wrap_cppjieba_init    , METH_VARARGS , "init cppjieba"}    ,
    {"cppjieba_cut"     , wrap_cppjieba_cut     , METH_VARARGS , "cppjieba cut"}     ,
    {"cppjieba_close"   , wrap_cppjieba_close   , METH_VARARGS , "cppjieba close"}   ,
    {NULL               , NULL}
};

extern "C" {
void initc_tf_idf_tk(void)
{
    Py_InitModule("c_tf_idf_tk", tf_idf_tk_methods);
}
}

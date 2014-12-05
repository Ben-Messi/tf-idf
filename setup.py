#setup.py
from distutils.core import setup, Extension
example_mod = Extension('c_tf_idf_tk', sources = ['wrapper.cpp'])
setup(name = "c_tf_idf_tk",
    version = "1.0",
    description = "A tf idf toolkit",
    ext_modules = [example_mod],
)

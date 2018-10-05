from distutils.core import setup
from Cython.Build import cythonize
import numpy
# Use  python set_cython_fun.py build_ext --inplace

setup(
    ext_modules=cythonize("LIF_AdaptiveThreshold_cy.pyx"),
    extra_compile_args=["-O3", "fopenmp"],
    extra_link_args=["fopenmp"],
    include_dirs=[numpy.get_include()]
)


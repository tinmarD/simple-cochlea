from distutils.core import setup
from Cython.Build import cythonize
import numpy
# Use  python setup_cochlea_cython_fun.py build_ext --inplace

setup(
    ext_modules=cythonize("cochlea_fun_cy.pyx"),
    extra_compile_args=["-O3", "fopenmp"],
    extra_link_args=["fopenmp"],
    include_dirs=[numpy.get_include()]
)


# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
#
# ext_modules = [
#     Extension(
#         "cochlea_fun_cy",
#         ["cochlea_fun_cy.pyx"],
#         extra_compile_args=['-fopenmp'],
#         extra_link_args=['-fopenmp'],
#     )
# ]
#
# setup(
#     name='cochlea_fun_cy',
#     ext_modules=cythonize(ext_modules),
# )

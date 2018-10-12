# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

install_requires = [
    "numpy",
    "pandas",
    "scipy",
    "seaborn",
    "scikit-learn",
    "tqdm",
    "cython"
]

extra_requires = {
    "librosa": ['librosa'],
    "pymuvr": ['pymuvr'],

    "peakutils": ['peakutils'],
}

cmdclass = {}
ext_modules = [Extension("simplecochlea.cython.cochlea_fun_cy",
                         ["simplecochlea/cython/cochlea_fun_cy.pyx"],
                         include_dirs=[numpy.get_include()]),
               Extension("simplecochlea.cython.lif_adaptthresh_fun_cy",
                         ["simplecochlea/cython/lif_adaptthresh_fun_cy.pyx"],
                         include_dirs=[numpy.get_include()]),
               ]
cmdclass.update({'build_ext': build_ext})

setup(
    name='simplecochlea',
    version='0.1.13',
    description='Simple cochlea model for sound-to-spikes conversion',
    long_description='',
    author='Martin Deudon',
    author_email='martin.deudon@protonmail.com',
    cmdclass=cmdclass,
    ext_modules=cythonize(ext_modules),
    url='',
    license='MIT',
    packages=find_packages(exclude=('docs', 'examples')),
    setup_requires=['numpy', 'cython'],
    install_requires=install_requires,
    python_requires='>=3.0',
)

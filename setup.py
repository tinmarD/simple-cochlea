# -*- coding: utf-8 -*-
import numpy

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext

USE_CYTHON = False

install_requires = [
    "numpy",
    "pandas",
    "scipy",
    "seaborn",
    "scikit-learn",
    "tqdm",
    "pymuvr",
]

ext = '.pxy' if USE_CYTHON else '.c'

cmdclass = {}
if USE_CYTHON:
    cmdclass.update({'build_ext': build_ext})
    ext_modules = [
        Extension("simplecochlea.cochlea_fun_cy", ["simplecochlea/cochlea_fun_cy" + ext]),
        Extension("simplecochlea.LIF_AdaptiveThreshold_cy", ["simplecochlea/LIF_AdaptiveThreshold_cy" + ext]),
    ]
else:
    ext_modules = [
        Extension("simplecochlea.cochlea_fun_cy", ["simplecochlea/cochlea_fun_cy" + ext],
                  include_dirs=[numpy.get_include()]),
        Extension("simplecochlea.LIF_AdaptiveThreshold_cy", ["simplecochlea/LIF_AdaptiveThreshold_cy" + ext],
                  include_dirs=[numpy.get_include()]),
    ]


setup(
    name='simple-cochlea',
    version='0.1.13',
    description='Simple cochlea model for sound-to-spikes conversion',
    long_description='',
    author='Martin Deudon',
    author_email='martin.deudon@protonmail.com',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    url='',
    license='MIT',
    packages=find_packages(exclude=('docs', 'examples')),
    install_requires=install_requires
)

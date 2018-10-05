# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='simple-cochlea',
    version='0.1.1',
    description='Simple cochlea model for sound-to-spikes conversion',
    long_description=readme,
    author='Martin Deudon',
    author_email='martin.deudon@protonmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('docs', 'examples'))
)


Installation
============

1. Install a Python Interpreter
################################

The simplecochlea package runs with Python 3.
The `Anaconda <https://www.anaconda.com/download/>`_ Python 3 distrubition is recommended :

If you already have Anaconda installed, you can create a new environment to avoid any troubles with your current setup :

 .. code-block:: console

    $ conda create --name simplecochlea_env python=3.6 numpy cython
    # Activate the new environment
    $ activate simplecochlea_env    # on Windows systems
    $ source activate simplecochlea_env    # on Unix systems

2. Install a C compiler
########################

You will need a C compiler to complete the installation, as simplecochlea uses Cython code to boost performances.
 * In Unix systems, the gcc compiler should already be installed.
 * Windows users can download Microsoft Visual Studio which contain a compiler. See `WindowsCompilers <https://wiki.python.org/moin/WindowsCompilers>`_ to know which version is more suited to your Python version.

3. Install simplecochlea package
################################

With pip and git :

  .. code-block:: console

    $ pip install git+https://github.com/tinmarD/simplecochlea.git@master

From github :

Download the zip file of the simplecochlea project from `github <https://github.com/tinmarD/simplecochlea>`_

  .. image:: ./../_static/images/install_github.png


Extract the archive and in a terminal (or in the Anaconda prompt), go to the simplecochlea root directory and run :

  .. code-block:: console

    $ python setup.py install

4. Test the installation
##########################

Open a Python prompt and import simplecochlea

  .. code-block:: python

    >>> import simplecochlea

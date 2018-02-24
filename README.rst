SPORCO-CUDA
===========


.. image:: https://readthedocs.org/projects/sporco-cuda/badge/?version=latest
    :target: http://sporco-cuda.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://badge.fury.io/py/sporco-cuda.svg
    :target: https://badge.fury.io/py/sporco-cuda
    :alt: PyPi Release
.. image:: https://img.shields.io/pypi/pyversions/sporco-cuda.svg
    :target: https://github.com/bwohlberg/sporco-cuda
    :alt: Supported Python Versions
.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://github.com/bwohlberg/sporco-cuda
    :alt: Package License


SPORCO-CUDA provides GPU-accelerated versions of selected convolutional sparse coding algorithms in the `SPORCO <https://github.com/bwohlberg/sporco>`__ package. It is a component of `SPORCO`, and is subject to the same license, but is made available as an optional extension to avoid complicating the prerequisites and build/install procedure for the main part of `SPORCO`. If you use this software for published work, please `cite it <http://sporco.readthedocs.io/en/latest/overview.html#citing>`__.



Documentation
-------------

Documentation is available online at `Read the Docs <http://sporco-cuda.rtfd.io/>`_, or can be built from the root directory of the source distribution by the command

::

   python setup.py build_sphinx

in which case the HTML documentation can be found in the ``build/sphinx/html`` directory (the top-level document is ``index.html``).




Usage
-----

Scripts illustrating usage of the package can be found in the ``examples`` directory of the source distribution. These examples can be run from the root directory of the package by, for example

::

   python examples/cmp_cbpdn.py


To run these scripts prior to installing the package, it is necessary to build it in place, which involves the following steps:

* Install the requirements described below

* If ``nvcc`` is not already in the executable search path, add it; e.g

  ::

    export PATH=$PATH:/usr/local/cuda-9.1/bin

  where ``/usr/local/cuda-9.1/bin`` is the path for CUDA compiler ``nvcc``.

* Build the ``sporco-cuda`` package in place:

  ::

    python setup.py build_ext --inplace

* Set the ``PYTHONPATH`` environment variable to include the root directory of the package. For example, in a ``bash`` shell

  ::

    export PYTHONPATH=$PYTHONPATH:`pwd`

  from the root directory of the package.

* If the ``sporco`` package is not installed, create a symlink from the SPORCO-CUDA package root directory to the ``sporco`` directory in the SPORCO package.



Requirements
------------

The primary requirements are Python, `sporco <https://github.com/bwohlberg/sporco>`__ and its requirements, `Cython <http://cython.org/>`_, and the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_.



Installation
------------

See the `installation instructions <http://sporco-cuda.rtfd.io/en/latest/install.html>`_ in the `online documentation <http://sporco-cuda.rtfd.io/>`_.



License
-------

SPORCO-CUDA is part of the `SPORCO <https://github.com/bwohlberg/sporco>`__ package and is distributed with the same 3-Clause BSD license; see the ``LICENSE`` file for details.

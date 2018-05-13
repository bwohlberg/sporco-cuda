Overview
========

SPORCO-CUDA is an extension package to the `SPORCO <http://sporco.rtfd.io>`__ package, providing GPU acceleration for selected algorithms. It is a component of `SPORCO`, and is subject to the same license, but is made available as an optional extension to avoid complicating the prerequisites and build/install procedure for the main part of `SPORCO`. If you use this software for published work, please `cite it <http://sporco.readthedocs.io/en/latest/overview.html#citing>`__.


.. _usage-section:

Using SPORCO-CUDA
-----------------

The recommended way of using SPORCO-CUDA is to install it as described in :ref:`installation-section`, and then access it via the `sporco.cuda <http://sporco.readthedocs.io/en/latest/sporco.cuda.html>`__ interface sub-package provided within the main `SPORCO <http://sporco.rtfd.io>`__ package.

SPORCO-CUDA can also be used directly, via its own interface. A collection of scripts illustrating such usage can be found in the ``examples`` directory of the source distribution. These examples can be run from the root directory of the package by, for example

::

   python examples/cmp_cbpdn.py

To run these scripts prior to installing the package, it is necessary to build it in place, which involves the following steps:

* Install the required packages as described in :ref:`requirements-section`.

* If the CUDA compiler ``nvcc`` is not already in the executable search path, add it, e.g.

  ::

    export PATH=$PATH:/usr/local/cuda-9.0/bin

  where ``/usr/local/cuda-9.0/bin`` is the path for ``nvcc``, or set the ``CUDAHOME`` environment variable to the root of the CUDA installation, e.g.

  ::

    export CUDAHOME=/usr/local/cuda-9.0

  where ``/usr/local/cuda-9.0`` is the root of the CUDA installation.

* Build the ``sporco-cuda`` package in place:

  ::

    python setup.py build_ext --inplace

* Set the ``PYTHONPATH`` environment variable to include the root directory of the package. For example, in a ``bash`` shell

  ::

    export PYTHONPATH=$PYTHONPATH:`pwd`

  from the root directory of the package.

* If the ``sporco`` package is not installed, create a symlink from the SPORCO-CUDA package root directory to the ``sporco`` directory in the SPORCO package.



If SPORCO-CUDA has been installed via ``pip``, the examples can be found in the directory in which ``pip`` installs documentation, e.g. ``/usr/local/share/doc/sporco-cuda-x.y.z/examples/``.



Contact
-------

Please submit bug reports, comments, etc. to brendt@ieee.org. Bugs and feature requests can also be reported via the `GitHub Issues interface <https://github.com/bwohlberg/sporco-cuda/issues>`_.



BSD License
-----------

This library was developed at Los Alamos National Laboratory, and has been approved for public release under the approval number LA-CC-14-057. It is made available under the terms of the BSD 3-Clause License; please see the ``LICENSE`` file for further details.

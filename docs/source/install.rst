Installation
============

The simplest way to install the most recent release of SPORCO from
`PyPI <https://pypi.python.org/pypi/sporco-cuda/>`_ is

::

    pip install sporco-cuda


SPORCO can also be installed from source, either from the development
version from `GitHub <https://github.com/bwohlberg/sporco-cuda>`_, or from
a release source package downloaded from `PyPI
<https://pypi.python.org/pypi/sporco-cuda/>`_.

To install the development version from `GitHub
<https://github.com/bwohlberg/sporco-cuda>`_, do

::

    git clone https://github.com/bwohlberg/sporco-cuda.git

followed by

::

   cd sporco-cuda
   python setup.py build
   python setup.py test
   python setup.py install

Please report any test failures. The install command will usually have to be performed with root permissions, e.g. on Ubuntu Linux

::

   sudo -H pip install sporco-cuda

or

::

   sudo python setup.py install

The procedure for installing from a source package downloaded from `PyPI
<https://pypi.python.org/pypi/sporco-cuda/>`_ is similar.


A summary of the most significant changes between SPORCO-CUDA releases can
be found in the ``CHANGES.rst`` file. It is strongly recommended to
consult this summary when updating from a previous version.


.. _requirements-section:

Requirements
------------

The primary requirements are Python itself, `Cython <http://cython.org/>`_, the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_, and modules `future
<http://python-future.org>`_, `numpy <http://www.numpy.org>`_, and `sporco <https://github.com/bwohlberg/sporco>`__


Installation of these requirements is system dependent.

.. tabs::

   .. group-tab:: :fa:`linux` Linux

      If the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ is not already installed, install it following the `instructions from Nvidia <http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation>`_.

      Under Ubuntu Linux, the remaining requirements can be installed via the package manager and `pip`.  Under Ubuntu 16.04, the following commands should be sufficient for Python 2

      ::

	sudo apt-get -y install cython python-numpy python-pip python-future python-pytest
	sudo -H pip install pytest-runner sporco

      or Python 3

      ::

	sudo apt-get -y install cython python3-numpy python3-pip python3-future python3-pytest
	sudo -H pip3 install pytest-runner sporco


      Some additional dependencies are required for building the
      documentation from the package source, for which Python 3.3 or
      later is required. For example, under Ubuntu Linux 16.04, the
      following commands should be sufficient

      ::

	sudo apt-get -y install python3-sphinx python3-numpydoc
	sudo -H pip3 install sphinx_tabs sphinx_fontawesome


   .. group-tab:: :fa:`apple` Mac OS

      Not yet tested under Mac OS



   .. group-tab:: :fa:`windows` Windows

      Not yet tested under Windows

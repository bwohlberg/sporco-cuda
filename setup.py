#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from future.utils import iteritems
from builtins import next
from builtins import filter

import os
from os.path import join as pjoin
from glob import glob
from ast import parse
from setuptools import setup
from distutils.extension import Extension
from distutils.command.clean import clean
from Cython.Distutils import build_ext
import subprocess
import numpy


# The approach used in this file is copied from the cython/CUDA setup.py
# example at https://github.com/rmcgibbo/npcuda-example

def find_in_path(name, path):
    "Find a file in a search path"

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iteritems(cudaconfig):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig



def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile



# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



class custom_clean(clean):
    def run(self):
        super(custom_clean, self).run()
        for f in glob(os.path.join('sporco_cuda', '*.pyx')):
            os.unlink(os.path.splitext(f)[0] + '.c')
        for f in glob(os.path.join('sporco_cuda', '*.so')):
            os.unlink(f)



CUDA = locate_cuda()

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()



gcc_flags = ['-shared', '-O2', '-fno-strict-aliasing']
nvcc_flags = [
    '-Xcompiler', "'-D__builtin_stdarg_start=__builtin_va_start'",
    '--compiler-options', "'-fno-inline'",
    '--compiler-options', "'-fno-strict-aliasing'",
    '--compiler-options', "'-Wall'",
    '-gencode', 'arch=compute_30,code=sm_30',
    '-gencode', 'arch=compute_35,code=sm_35',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_62,code=sm_62',
    '-Xcompiler', "'-fPIC'"
    ]


ext_util = Extension('sporco_cuda.util',
        sources= ['sporco_cuda/src/utils.cu',
                  'sporco_cuda/util.pyx'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cuda', 'cudart'],
        language = 'c',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args = {
            'gcc': gcc_flags,
            'nvcc': nvcc_flags
            },
        include_dirs = [numpy_include, CUDA['include'], 'sporco_cuda/src'])


ext_cbpdn = Extension('sporco_cuda.cbpdn',
        sources= ['sporco_cuda/src/utils.cu',
                  'sporco_cuda/src/cbpdn_kernels.cu',
                  'sporco_cuda/src/cbpdn.cu',
                  'sporco_cuda/src/cbpdn_grd.cu',
                  'sporco_cuda/cbpdn.pyx'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cuda', 'cudart', 'cufft', 'cublas', 'm'],
        language = 'c',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args = {
            'gcc': gcc_flags,
            'nvcc': nvcc_flags
            },
        include_dirs = [numpy_include, CUDA['include'], 'sporco_cuda/src'])




name = 'sporco-cuda'
pname = 'sporco_cuda'

# Get version number from sporco/__init__.py
# See http://stackoverflow.com/questions/2058802
with open(os.path.join(pname, '__init__.py')) as f:
    version = parse(next(filter(
        lambda line: line.startswith('__version__'),
        f))).body[0].value.s


longdesc = \
"""
SPORCO-CUDA is an extension package to Sparse Optimisation Research
Code (SPORCO), providing GPU accelerated versions for some
convolutional sparse coding problems.
"""

setup(
    author           = 'Gustavo Silva, Brendt Wohlberg',
    author_email     = 'gustavo.silva@pucp.edu.pe, brendt@ieee.org',
    name             = name,
    description      = 'SPORCO-CUDA: A CUDA extension package for SPORCO',
    long_description = longdesc,
    keywords         = ['Convolutional Sparse Representations',
                        'Convolutional Sparse Coding', 'CUDA'],
    url              = 'https://github.com/bwohlberg/sporco-cuda',
    version          = version,
    platforms        = 'Linux',
    license          = 'BSD',
    setup_requires   = ['cython', 'future', 'numpy'],
    #tests_require    = ['pytest', 'pytest-runner', 'sporco'],
    tests_require    = ['pytest', 'pytest-runner'],
    install_requires = ['future', 'numpy'],
    classifiers = [
    'License :: OSI Approved :: BSD License',
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules'
    ],

    # extension module specification
    ext_modules = [ext_util, ext_cbpdn],
    # inject our custom trigger
    cmdclass = {'build_ext': custom_build_ext,
                'clean': custom_clean},
    # since the package has c code, the egg cannot be zipped
    zip_safe = False)

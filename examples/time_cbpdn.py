#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO-CUDA package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: compute initialisation time and solve time per iteration for
CUDA ConvBPDN solver"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco import util
from sporco import plot
import sporco.metric as sm
from sporco.admm import cbpdn
import sporco_cuda.cbpdn as cucbpdn


# Load demo image
img = util.ExampleImages().image('barbara.png', scaled=True, gray=True)


# Highpass filter test image
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(img, fltlmbd, npd)


# Load dictionary
D = util.convdicts()['G:12x12x72']


# Set up ConvBPDN options
lmbda = 1e-2
opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 0,
                    'HighMemSolve': True, 'LinSolveCheck': True,
                    'RelStopTol': 1e-5, 'AuxVarObj': False,
                    'AutoRho': {'Enabled': False}})


# Compute initialisation time: solve with 0 iterations
t0 = util.Timer()
with util.ContextTimer(t0):
    X = cucbpdn.cbpdn(D, sh, lmbda, opt)


# Solve with Niter iterations
Niter = 200
opt['MaxMainIter'] = Niter
t1 = util.Timer()
with util.ContextTimer(t1):
    X = cucbpdn.cbpdn(D, sh, lmbda, opt)


# Print run time information
print("GPU ConvBPDN init time: %.3fs" % t0.elapsed())
print("GPU ConvBPDN solve time per iteration: %.3fs" % (t1.elapsed() / Niter))

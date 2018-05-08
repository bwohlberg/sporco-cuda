#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO-CUDA package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: compare solve times for Python and CUDA AddMaskSim/ConvBPDN
solvers"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco import util
from sporco import plot
from sporco.admm import cbpdn
import sporco.metric as sm

import sporco_cuda.cbpdn as cucbpdn


# Load demo image
img = util.ExampleImages().image('barbara.png', scaled=True, gray=True)


# Highpass filter test image
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(img, fltlmbd, npd)


# Apply random mask to highpass component
frc = 0.5
np.random.seed(12345)
msk = util.rndmask(img.shape, frc, dtype=np.float32)
shw = msk * sh


# Load dictionary
D = util.convdicts()['G:12x12x72']


# Set up ConvBPDN options
lmbda = 1e-2
opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 20,
                    'HighMemSolve': True, 'LinSolveCheck': False,
                    'RelStopTol': 2e-3, 'AuxVarObj': False,
                    'rho': 1.5e0, 'AutoRho': {'Enabled': False}})


# Initialise and run AddMaskSim/ConvBPDN object
b = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, shw, msk, lmbda, opt=opt)
X1 = b.solve()
print("AddMaskSim/ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'))


# Time CUDA AddMaskSim/ConvBPDN solve
t = util.Timer()
with util.ContextTimer(t):
    X2 = cucbpdn.cbpdnmsk(D, shw, msk, lmbda, opt)


# Solve time comparison
print("GPU AddMaskSim/ConvBPDN solve time: %.2fs" % t.elapsed())
print("GPU time improvement factor: %.1f" % (b.timer.elapsed('solve') /
                                             t.elapsed()))


# Compare CPU and GPU solutions
print("CPU solution:  min: %.4e  max: %.4e   l1: %.4e" %
          (X1.min(), X1.max(), np.sum(np.abs(X1))))
print("GPU solution:  min: %.4e  max: %.4e   l1: %.4e" %
          (X2.min(), X2.max(), np.sum(np.abs(X2))))
print("CPU/GPU MSE: %.2e  SNR: %.2f dB" % (sm.mse(X1, X2), sm.snr(X1, X2)))

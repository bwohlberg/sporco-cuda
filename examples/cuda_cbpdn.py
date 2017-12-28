#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO-CUDA package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: CUDA cbpdn of a large image"""

from __future__ import print_function
from builtins import input
from builtins import range

import os
import tempfile
import numpy as np

from sporco import util
from sporco import plot
from sporco.admm import cbpdn
import sporco.metric as sm

import sporco_cuda.cbpdn as cucbpdn


# Get test image
url = 'https://www.math.purdue.edu/~lucier/PHOTO_CD/BMP_IMAGES/IMG0023.bmp'
dir = os.path.join(tempfile.gettempdir(), 'images')
if not os.path.exists(dir):
    os.mkdir(dir)
pth = os.path.join(dir, 'kodim23.bmp')
if not os.path.isfile(pth):
    img = util.netgetdata(url)
    f = open(pth, 'wb')
    f.write(img.read())
    f.close()


# Load demo image
ei = util.ExampleImages(pth=dir)
img = ei.image('kodim23.bmp', scaled=True, gray=True, zoom=0.5)


# Highpass filter test image
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(img, fltlmbd, npd)


# Load dictionary
D = util.convdicts()['G:12x12x72']


# Set up ConvBPDN options
lmbda = 1e-2
opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                    'HighMemSolve': True, 'RelStopTol': 1e-3,
                    'AuxVarObj': False, 'AutoRho': {'Enabled': False}})

# Time CUDA cbpdn solve
t = util.Timer()
with util.ContextTimer(t):
    X = cucbpdn.cbpdn(D, sh, lmbda, opt)
print("GPU ConvBPDN solve time: %.2fs" % t.elapsed())
print("Image size: %d x %d" % sh.shape)

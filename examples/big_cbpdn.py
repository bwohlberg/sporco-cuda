#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO-CUDA package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: CUDA ConvBPDN solver applied to a large image"""

from __future__ import print_function
from builtins import input
from builtins import range

import os
import tempfile
import numpy as np

from sporco import util
from sporco import plot
import sporco.linalg as spl
from sporco.admm import cbpdn
import sporco_cuda.cbpdn as cucbpdn


# Get test image
url = 'http://www.math.purdue.edu/~lucier/PHOTO_CD/D65_GREY_TIFF_IMAGES/'\
      'IMG0023.tif'
dir = os.path.join(tempfile.gettempdir(), 'images')
if not os.path.exists(dir):
    os.mkdir(dir)
pth = os.path.join(dir, 'IMG0023.tif')
if not os.path.isfile(pth):
    img = util.netgetdata(url)
    f = open(pth, 'wb')
    f.write(img.read())
    f.close()


# Load demo image
ei = util.ExampleImages(pth=dir)
img = ei.image('IMG0023.tif', scaled=True, zoom=0.5)


# Highpass filter test image
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(img, fltlmbd, npd)


# Load dictionary
D = util.convdicts()['G:12x12x72']


# Set up ConvBPDN options
lmbda = 1e-2
opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                    'HighMemSolve': True, 'RelStopTol': 5e-3,
                    'AuxVarObj': False, 'AutoRho': {'Enabled': False},
                    'rho': 5.0})


# Time CUDA cbpdn solve
t = util.Timer()
with util.ContextTimer(t):
    X = cucbpdn.cbpdn(D, sh, lmbda, opt)
print("Image size: %d x %d" % sh.shape)
print("GPU ConvBPDN solve time: %.2fs" % t.elapsed())


# Reconstruct the image from the sparse representation
shr = np.sum(spl.fftconv(D, X), axis=2)
imgr = sl + shr


#Display representation and reconstructed image.
fig = plot.figure(figsize=(14, 14))
plot.subplot(2, 2, 1)
plot.imview(sl, fig=fig, title='Lowpass component')
plot.subplot(2, 2, 2)
plot.imview(np.sum(abs(X), axis=2).squeeze(), fig=fig,
            cmap=plot.cm.Blues, title='Main representation')
plot.subplot(2, 2, 3)
plot.imview(imgr, fig=fig, title='Reconstructed image')
plot.subplot(2, 2, 4)
plot.imview(imgr - img, fig=fig, fltscl=True,
            title='Reconstruction difference')
fig.show()


# Wait for enter on keyboard
input()

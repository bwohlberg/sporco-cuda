#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO-CUDA package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: CUDA ConvBPDNGradReg solver applied to a large image"""

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


# Load dictionary
Db = util.convdicts()['G:12x12x72']
# Append impulse filter for lowpass component representation
di = np.zeros(Db.shape[0:2] + (1,), dtype=np.float32)
di[0, 0] = 1
D = np.concatenate((di, Db), axis=2)
# Weights for l1 norm: no regularization on impulse filter
wl1 = np.ones((1,)*2 + (D.shape[2:]), dtype=np.float32)
wl1[..., 0] = 0.0
# Weights for l2 norm of gradient: regularization only on impulse filter
wgr = np.zeros((D.shape[2]), dtype=np.float32)
wgr[0] = 1.0


# Set up ConvBPDNGradReg options
lmbda = 1e-2
mu = 1e1
opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': True, 'MaxMainIter': 200,
                    'HighMemSolve': True, 'RelStopTol': 5e-3,
                    'AuxVarObj': False, 'AutoRho': {'Enabled': False},
                    'rho': 0.2, 'L1Weight': wl1, 'GradWeight': wgr})


# Time CUDA cbpdn solve
t = util.Timer()
with util.ContextTimer(t):
    X = cucbpdn.cbpdngrd(D, img, lmbda, mu, opt)
print("Image size: %d x %d" % img.shape)
print("GPU ConvBPDNGradReg solve time: %.2fs" % t.elapsed())


# Reconstruct the image from the sparse representation
imgr = np.sum(spl.fftconv(D, X), axis=2)


#Display representation and reconstructed image.
fig = plot.figure(figsize=(14, 14))
plot.subplot(2, 2, 1)
plot.imview(X[..., 0].squeeze(), fig=fig, title='Lowpass component')
plot.subplot(2, 2, 2)
plot.imview(np.sum(abs(X[..., 1:]), axis=2).squeeze(), fig=fig,
            cmap=plot.cm.Blues, title='Main representation')
plot.subplot(2, 2, 3)
plot.imview(imgr, fig=fig, title='Reconstructed image')
plot.subplot(2, 2, 4)
plot.imview(imgr - img, fig=fig, fltscl=True,
            title='Reconstruction difference')
fig.show()


# Wait for enter on keyboard
input()

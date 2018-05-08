#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO-CUDA package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: CUDA utility functions"""

from __future__ import print_function
from builtins import input
from builtins import range

import sporco_cuda.util as cu


ndev = cu.device_count()
print('Found %d CUDA device(s)' % ndev)
if ndev > 0:
    print('Current device id: %d' % cu.current_device())
    mbc = 1024.0**2
    print('Id   Model                 Total memory     Free Memory')
    for n in range(ndev):
        cu.current_device(n)
        mf, mt = cu.memory_info()
        nm = cu.device_name(n)
        print('%2d   %-20s   %8.0f MB     %8.0f MB' % (n, nm, mt/mbc, mf/mbc))

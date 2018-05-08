from __future__ import division
from builtins import object

import pytest
import numpy as np

import sporco_cuda.util as cu


class TestSet01(object):

    def test_01(self):
        assert(cu.device_count() >= 0)


    def test_02(self):
        assert(cu.current_device() >= 0)
        assert(cu.current_device(0) == 0)


    def test_03(self):
        f, t = cu.memory_info()
        assert(f >= 0 and t > 0)


    def test_04(self):
        nm = cu.device_name()
        assert(nm is not None and nm != '')

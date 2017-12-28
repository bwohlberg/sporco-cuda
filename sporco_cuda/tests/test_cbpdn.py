from __future__ import division
from builtins import object

import pytest
import numpy as np

from sporco.admm import cbpdn
import sporco_cuda.cbpdn as cucbpdn
import sporco.metric as sm



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        lmbda = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 50,
                                 'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDN(D, s, lmbda, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdn(D, s, lmbda, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_02(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        lmbda = 1e-1
        Wl1 = np.random.randn(1, 1, M).astype(np.float32)
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 50,
                    'L1Weight': Wl1, 'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDN(D, s, lmbda, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdn(D, s, lmbda, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_03(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        lmbda = 1e-1
        Wl1 = np.random.randn(Nr, Nc, M).astype(np.float32)
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 50,
                    'L1Weight': Wl1, 'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDN(D, s, lmbda, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdn(D, s, lmbda, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_04(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        lmbda = 1e-1
        mu = 1e-2
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False,
                'MaxMainIter': 50, 'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdngrd(D, s, lmbda, mu, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_05(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        lmbda = 1e-1
        mu = 1e-2
        Wgrd = np.random.randn(M).astype(np.float32)
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False,
                        'MaxMainIter': 50, 'GradWeight': Wgrd,
                        'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdngrd(D, s, lmbda, mu, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_06(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        lmbda = 1e-1
        mu = 1e-2
        Wl1 = np.random.randn(1, 1, M).astype(np.float32)
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False,
                'MaxMainIter': 50, 'L1Weight': Wl1,
                'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdngrd(D, s, lmbda, mu, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_07(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        lmbda = 1e-1
        mu = 1e-2
        Wl1 = np.random.randn(Nr, Nc, M).astype(np.float32)
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False,
                'MaxMainIter': 50, 'L1Weight': Wl1,
                'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdngrd(D, s, lmbda, mu, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_08(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        lmbda = 1e-1
        mu = 1e-2
        Wl1 = np.random.randn(Nr, Nc, M).astype(np.float32)
        Wgrd = np.random.randn(M).astype(np.float32)
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False,
                'MaxMainIter': 50, 'L1Weight': Wl1, 'GradWeight': Wgrd,
                'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdngrd(D, s, lmbda, mu, opt)
        assert(sm.mse(X1, X2) < 1e-10)

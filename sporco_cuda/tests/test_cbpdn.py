from __future__ import division
from builtins import object

import pytest
import numpy as np

from sporco.admm import cbpdn
import sporco_cuda.cbpdn as cucbpdn
import sporco.metric as sm
import sporco.util as su



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
        Wl1 = np.random.randn(1, 1, M).astype(np.float32)
        Wl1[0] = 0.0
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 50,
                    'L1Weight': Wl1, 'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDN(D, s, lmbda, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdn(D, s, lmbda, opt)
        assert(sm.mse(X1, X2) < 1e-8)



    def test_04(self):
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



    def test_05(self):
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



    def test_06(self):
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



    def test_07(self):
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
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False,
                'MaxMainIter': 50, 'L1Weight': Wl1,
                'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu, opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdngrd(D, s, lmbda, mu, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_09(self):
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



    def test_10(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        frc = 0.5
        msk = su.rndmask(s.shape, frc, dtype=np.float32)
        s *= msk
        lmbda = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 50,
                                      'AutoRho': {'Enabled': False}})
        b = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, s, msk, lmbda, opt=opt)
        X1 = b.solve()
        X2 = cucbpdn.cbpdnmsk(D, s, msk, lmbda, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_11(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        frc = 0.5
        msk = su.rndmask(s.shape, frc, dtype=np.float32)
        s *= msk
        lmbda = 1e-1
        # Create a random ℓ1 term weighting array. There is no need to
        # extend this array to account for the AMS impulse filter since
        # this is taken care of automatically by cucbpdn.cbpdnmsk
        Wl1 = np.random.randn(1, 1, M).astype(np.float32)
        # Append a zero entry to the L1Weight array, corresponding to
        # the impulse filter appended to the dictionary by cbpdn.AddMaskSim,
        # since this is not done automatically by cbpdn.AddMaskSim
        Wl1i = np.concatenate((Wl1, np.zeros(Wl1.shape[0:-1] + (1,))),
                              axis=-1)
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 50,
                                      'AutoRho': {'Enabled': False}})
        opt['L1Weight'] = Wl1i
        b = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, s, msk, lmbda, opt=opt)
        X1 = b.solve()
        opt['L1Weight'] = Wl1
        X2 = cucbpdn.cbpdnmsk(D, s, msk, lmbda, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_12(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        frc = 0.5
        msk = su.rndmask(s.shape, frc, dtype=np.float32)
        s *= msk
        lmbda = 1e-1
        mu = 1e-2
        # Since cucbpdn.cbpdngrdmsk automatically ensures that the ℓ2 of
        # gradient term is not applied to the AMS impulse filter, while
        # cbpdn.AddMaskSim does not, we have to pass a GradWeight array
        # with a zero entry corresponding to the AMS impulse filter to
        # cbpdn.AddMaskSim
        Wgrdi = np.hstack((np.ones(M,), np.zeros((1,))))
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False,
                'MaxMainIter': 50, 'AutoRho': {'Enabled': False}})
        opt['GradWeight'] = Wgrdi
        b = cbpdn.AddMaskSim(cbpdn.ConvBPDNGradReg, D, s, msk, lmbda, mu, opt)
        X1 = b.solve()
        opt['GradWeight'] = 1.0
        X2 = cucbpdn.cbpdngrdmsk(D, s, msk, lmbda, mu, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_13(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        frc = 0.5
        msk = su.rndmask(s.shape, frc, dtype=np.float32)
        s *= msk
        lmbda = 1e-1
        mu = 1e-2
        # Create a random ℓ1 term weighting array. There is no need to
        # extend this array to account for the AMS impulse filter since
        # this is taken care of automatically by cucbpdn.cbpdngrdmsk
        Wl1 = np.random.randn(1, 1, M).astype(np.float32)
        # Append a zero entry to the L1Weight array, corresponding to
        # the impulse filter appended to the dictionary by cbpdn.AddMaskSim,
        # since this is not done automatically by cbpdn.AddMaskSim
        Wl1i = np.concatenate((Wl1, np.zeros(Wl1.shape[0:-1] + (1,))),
                              axis=-1)
        # Since cucbpdn.cbpdngrdmsk automatically ensures that the ℓ2 of
        # gradient term is not applied to the AMS impulse filter, while
        # cbpdn.AddMaskSim does not, we have to pass a GradWeight array
        # with a zero entry corresponding to the AMS impulse filter to
        # cbpdn.AddMaskSim
        Wgrdi = np.hstack((np.ones(M,), np.zeros((1,))))
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False,
                'MaxMainIter': 50, 'AutoRho': {'Enabled': False}})
        opt['L1Weight'] = Wl1i
        opt['GradWeight'] = Wgrdi
        b = cbpdn.AddMaskSim(cbpdn.ConvBPDNGradReg, D, s, msk, lmbda, mu, opt)
        X1 = b.solve()
        opt['L1Weight'] = Wl1
        opt['GradWeight'] = 1.0
        X2 = cucbpdn.cbpdngrdmsk(D, s, msk, lmbda, mu, opt)
        assert(sm.mse(X1, X2) < 1e-10)



    def test_14(self):
        Nr = 32
        Nc = 31
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M).astype(np.float32)
        s = np.random.randn(Nr, Nc).astype(np.float32)
        frc = 0.5
        msk = su.rndmask(s.shape, frc, dtype=np.float32)
        s *= msk
        lmbda = 1e-1
        mu = 1e-2
        # Create a random ℓ2 of gradient term weighting array. There is no
        # need to extend this array to account for the AMS impulse filter
        # since this is taken care of automatically by cucbpdn.cbpdngrdmsk
        Wgrd = np.random.randn(M).astype(np.float32)
        # Append a zero entry to the GradWeight array, corresponding to
        # the impulse filter appended to the dictionary by cbpdn.AddMaskSim,
        # since this is not done automatically by cbpdn.AddMaskSim
        Wgrdi = np.hstack((Wgrd, np.zeros((1,))))
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False,
                'MaxMainIter': 50, 'AutoRho': {'Enabled': False}})
        opt['GradWeight'] = Wgrdi
        b = cbpdn.AddMaskSim(cbpdn.ConvBPDNGradReg, D, s, msk, lmbda, mu, opt)
        X1 = b.solve()
        opt['GradWeight'] = Wgrd
        X2 = cucbpdn.cbpdngrdmsk(D, s, msk, lmbda, mu, opt)
        assert(sm.mse(X1, X2) < 1e-10)

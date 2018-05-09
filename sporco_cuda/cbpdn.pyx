# cython: embedsignature=True

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


cdef extern from "algopt.h":
    struct AlgOpt:
       int     MaxMainIter
       float   rho
       float   RhoRsdlRatio
       float   RhoScaling
       float   RhoRsdlTarget
       float   RelaxParam
       float   AbsStopTol
       float   RelStopTol
       float*  L1Weight
       int     L1_WEIGHT_M_SIZE
       int     L1_WEIGHT_ROW_SIZE
       int     L1_WEIGHT_COL_SIZE
       int*    Weight
       int     nWeight
       int     Verbose
       int     NonNegCoef
       int     NoBndryCross
       int     AuxVarObj
       int     HighMemSolve
       int     AutoRho
       int     AutoRhoScaling
       int     AutoRhoPeriod
       int     StdResiduals
       int     IMG_ROW_SIZE
       int     IMG_COL_SIZE
       int     DICT_M_SIZE
       int     DICT_ROW_SIZE
       int     DICT_COL_SIZE
       float*  GrdWeight
       int     WEIGHT_SIZE
       int     device



cdef extern void cuda_wrapper_CBPDN(float* D, float* S, float lmbda,
                                    void* opt, float *Y)

cdef extern void cuda_wrapper_CBPDN_GR(float *D, float *S, float lmbda,
                                       float mu, void *opt, float *Y)

cdef extern void clear_opts(void *data)



cdef set_algopt(AlgOpt* algopt, opt):

    algopt.Verbose = int(opt['Verbose'])

    algopt.MaxMainIter = int(opt['MaxMainIter'])

    algopt.RelaxParam = DTYPE(opt['RelaxParam'])

    algopt.AbsStopTol = DTYPE(opt['AbsStopTol'])
    algopt.RelStopTol = DTYPE(opt['RelStopTol'])

    algopt.AuxVarObj = int(opt['AuxVarObj'])
    algopt.HighMemSolve = int(opt['HighMemSolve'])
    algopt.NonNegCoef = int(opt['NonNegCoef'])
    algopt.NoBndryCross = int(opt['NoBndryCross'])

    algopt.StdResiduals = int(opt['AutoRho']['StdResiduals'])
    algopt.AutoRho = int(opt['AutoRho']['Enabled'])
    algopt.AutoRhoScaling = DTYPE(opt['AutoRho']['AutoScaling'])
    algopt.AutoRhoPeriod = int(opt['AutoRho']['Period'])
    algopt.RhoRsdlRatio = DTYPE(opt['AutoRho']['RsdlRatio'])
    algopt.RhoScaling = DTYPE(opt['AutoRho']['Scaling'])
    algopt.RhoRsdlTarget = DTYPE(opt['AutoRho']['RsdlTarget'])




@cython.boundscheck(False)
@cython.wraparound(False)
def cbpdn(np.ndarray[DTYPE_t, ndim=3] D not None,
          np.ndarray[DTYPE_t, ndim=2] S not None,
          DTYPE_t lmbda, opt, int dev=0):

    # No spatial mask for this problem so we pass a 1 x 1 array with a
    # unit entry as the corresponding mask array
    W = np.ones((1, 1), dtype=np.dtype("i"))

    # Call main implementation for CBPDN problem. The use of np.rollaxis
    # is required because the CUDA implementation assumes Matlab array
    # layout.
    X = _cbpdn(np.ascontiguousarray(np.rollaxis(D, 2, 0)),
               np.ascontiguousarray(S), np.ascontiguousarray(W),
               lmbda, dict(opt), dev)

    # Return the coefficient map array
    return np.rollaxis(X, 0, 3)



@cython.boundscheck(False)
@cython.wraparound(False)
def cbpdnmsk(np.ndarray[DTYPE_t, ndim=3] D not None,
             np.ndarray[DTYPE_t, ndim=2] S not None,
             np.ndarray W, DTYPE_t lmbda, opt, int dev=0):

    # Prepend an impulse filter as the initial dictionary filter, for use
    # within the Additive Mask Simulation (AMS) method for implementing a
    # spatial mask in the data fidelity term (see
    # doi: 10.1109/ICIP.2016.7532675)
    d0 = np.zeros((D.shape[0], D.shape[1], 1), dtype=DTYPE)
    d0[0, 0] = 1.0
    Di = np.dstack((d0, D))

    # Check whether the L1Weight option is an array or a scalar
    if hasattr(opt['L1Weight'], 'ndim'):
        # If the L1Weight option is an array, prepend a zero array of the
        # appropriate shape so that ℓ1 regularization is not applied to
        # the initial AMS impulse filter in the dictionary
        w0 = np.zeros(opt['L1Weight'].shape[0:-1] + (1,), dtype=DTYPE)
        opt['L1Weight'] = np.dstack((w0, opt['L1Weight']))
    else:
        # If the L1Weight option is a scalar, set it to a constant array
        # of the same value and then set the first entry to zero so that
        # ℓ1 regularization is not applied to the initial AMS impulse
        # filter in the dictionary
        opt['L1Weight'] = opt['L1Weight'] * np.ones((1, 1, Di.shape[2]),
                                                    dtype=DTYPE)
        opt['L1Weight'][..., 0] = 0.0

    # Call main implementation for CBPDN problem. The use of np.rollaxis
    # is required because the CUDA implementation assumes Matlab array
    # layout.
    X = _cbpdn(np.ascontiguousarray(np.rollaxis(Di, 2, 0)),
               np.ascontiguousarray(S),
               np.ascontiguousarray(W, dtype=np.dtype("i")),
               lmbda, dict(opt), dev)

    # Return the coefficient map array, slicing off the initial coefficient
    # map corresponding to the AMS impulse filter
    return np.rollaxis(X, 0, 3)[..., 1:]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef _cbpdn(np.ndarray[DTYPE_t, ndim=3, mode="c"] D,
          np.ndarray[DTYPE_t, ndim=2, mode="c"] S,
          np.ndarray[int, ndim=2, mode="c"] W,
          DTYPE_t lmbda, dict opt, int dev):

    cdef AlgOpt algopt
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] X = np.zeros(
        (D.shape[0], S.shape[0], S.shape[1]), dtype=DTYPE, order='C')

    clear_opts(&algopt)
    set_algopt(&algopt, opt)

    algopt.device = dev

    algopt.IMG_ROW_SIZE = S.shape[0]
    algopt.IMG_COL_SIZE = S.shape[1]

    algopt.DICT_ROW_SIZE = D.shape[1]
    algopt.DICT_COL_SIZE= D.shape[2]
    algopt.DICT_M_SIZE = D.shape[0]

    algopt.rho = 5e1*lmbda+1.0 if opt['rho'] is None else DTYPE(opt['rho'])

    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] gl1w = np.asarray([1.0],
            dtype=DTYPE).reshape((1,1,1))
    if hasattr(opt['L1Weight'], 'ndim'):
        gl1w = np.ascontiguousarray(np.rollaxis(opt['L1Weight'], 2, 0))
        algopt.L1_WEIGHT_ROW_SIZE = gl1w.shape[1]
        algopt.L1_WEIGHT_COL_SIZE = gl1w.shape[2]
        algopt.L1_WEIGHT_M_SIZE = gl1w.shape[0]
    else:
        gl1w[0] = opt['L1Weight']
        algopt.L1_WEIGHT_ROW_SIZE = 1
        algopt.L1_WEIGHT_COL_SIZE = 1
        algopt.L1_WEIGHT_M_SIZE = 1
    algopt.L1Weight = &gl1w[0,0,0]

    if W.size > 1:
        algopt.Weight = &W[0,0]
        algopt.nWeight = W.size

    cuda_wrapper_CBPDN(&D[0,0,0], &S[0,0], lmbda, &algopt, &X[0,0,0])

    return X



@cython.boundscheck(False)
@cython.wraparound(False)
def cbpdngrd(np.ndarray[DTYPE_t, ndim=3] D not None,
          np.ndarray[DTYPE_t, ndim=2] S not None,
          DTYPE_t lmbda, DTYPE_t mu, opt, int dev=0):

    # No spatial mask for this problem so we pass a 1 x 1 array with a
    # unit entry as the corresponding mask array
    W = np.ones((1, 1), dtype=np.dtype("i"))

    # Call main implementation for CBPDNGradReg problem. The use of
    # np.rollaxis is required because the CUDA implementation assumes
    # Matlab array layout.
    X = _cbpdngrd(np.ascontiguousarray(np.rollaxis(D, 2, 0)),
               np.ascontiguousarray(S), np.ascontiguousarray(W),
               lmbda, mu, dict(opt), dev)

    # Return the coefficient map array
    return np.rollaxis(X, 0, 3)



@cython.boundscheck(False)
@cython.wraparound(False)
def cbpdngrdmsk(np.ndarray[DTYPE_t, ndim=3] D not None,
          np.ndarray[DTYPE_t, ndim=2] S not None,
          np.ndarray W, DTYPE_t lmbda, DTYPE_t mu, opt, int dev=0):

    # Prepend an impulse filter as the initial dictionary filter, for use
    # within the Additive Mask Simulation (AMS) method for implementing a
    # spatial mask in the data fidelity term (see
    # doi: 10.1109/ICIP.2016.7532675)
    d0 = np.zeros((D.shape[0], D.shape[1], 1), dtype=DTYPE)
    d0[0, 0] = 1.0
    Di = np.dstack((d0, D))

    # Check whether the L1Weight option is an array or a scalar
    if hasattr(opt['L1Weight'], 'ndim'):
        # If the L1Weight option is an array, prepend a zero array of the
        # appropriate shape so that ℓ1 regularization is not applied to
        # the initial AMS impulse filter in the dictionary
        w0 = np.zeros(opt['L1Weight'].shape[0:-1] + (1,), dtype=DTYPE)
        opt['L1Weight'] = np.dstack((w0, opt['L1Weight']))
    else:
        # If the L1Weight option is a scalar, set it to a constant array
        # of the same value and then set the first entry to zero so that
        # ℓ1 regularization is not applied to the initial AMS impulse
        # filter in the dictionary
        opt['L1Weight'] = opt['L1Weight'] * np.ones((1, 1, Di.shape[2]),
                                                    dtype=DTYPE)
        opt['L1Weight'][..., 0] = 0.0

    # Check whether the GradWeight option is an array or a scalar
    if hasattr(opt['GradWeight'], 'ndim'):
        # If the GradWeight option is an array, prepend a zero entry
        # so that ℓ2 of gradient regularization is not applied to the
        # initial AMS impulse filter in the dictionary
        w0 = np.zeros((1,), dtype=DTYPE)
        opt['GradWeight'] = np.hstack((w0, opt['GradWeight']))
    else:
        # If the GradWeight option is a scalar, set it to a constant array
        # of the same value and then set the first entry to zero so that
        # ℓ2 of gradient regularization is not applied to the initial AMS
        # impulse filter in the dictionary
        opt['GradWeight'] = opt['GradWeight'] * np.ones((Di.shape[2],),
                                                        dtype=DTYPE)
        opt['GradWeight'][..., 0] = 0.0

    # Call main implementation for CBPDNGradReg problem. The use of
    # np.rollaxis is required because the CUDA implementation assumes
    # Matlab array layout.
    X = _cbpdngrd(np.ascontiguousarray(np.rollaxis(Di, 2, 0)),
                  np.ascontiguousarray(S),
                  np.ascontiguousarray(W, dtype=np.dtype("i")),
                  lmbda, mu, dict(opt), dev)

    # Return the coefficient map array, slicing off the initial coefficient
    # map corresponding to the AMS impulse filter
    return np.rollaxis(X, 0, 3)[..., 1:]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef _cbpdngrd(np.ndarray[DTYPE_t, ndim=3, mode="c"] D,
             np.ndarray[DTYPE_t, ndim=2, mode="c"] S,
             np.ndarray[int, ndim=2, mode="c"] W,
             DTYPE_t lmbda, DTYPE_t mu, dict opt, int dev):

    cdef AlgOpt algopt
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] X = np.zeros(
        (D.shape[0], S.shape[0], S.shape[1]), dtype=DTYPE, order='C')

    clear_opts(&algopt)
    set_algopt(&algopt, opt)

    algopt.device = dev

    algopt.IMG_ROW_SIZE = S.shape[0]
    algopt.IMG_COL_SIZE = S.shape[1]

    algopt.DICT_ROW_SIZE = D.shape[1]
    algopt.DICT_COL_SIZE= D.shape[2]
    algopt.DICT_M_SIZE = D.shape[0]

    algopt.rho = 5e1*lmbda+1.0 if opt['rho'] is None else DTYPE(opt['rho'])

    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] gl1w = np.asarray([1.0],
            dtype=DTYPE).reshape((1,1,1))
    if hasattr(opt['L1Weight'], 'ndim'):
        gl1w = np.ascontiguousarray(np.rollaxis(opt['L1Weight'], 2, 0))
        algopt.L1_WEIGHT_ROW_SIZE = gl1w.shape[1]
        algopt.L1_WEIGHT_COL_SIZE = gl1w.shape[2]
        algopt.L1_WEIGHT_M_SIZE = gl1w.shape[0]
    else:
        gl1w[0] = opt['L1Weight']
        algopt.L1_WEIGHT_ROW_SIZE = 1
        algopt.L1_WEIGHT_COL_SIZE = 1
        algopt.L1_WEIGHT_M_SIZE = 1
    algopt.L1Weight = &gl1w[0,0,0]

    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] gw = np.asarray([1.0],
            dtype=DTYPE)
    if hasattr(opt['GradWeight'], 'ndim'):
        gw = opt['GradWeight']
    else:
        gw[0] = opt['GradWeight']
    algopt.WEIGHT_SIZE = gw.shape[0]
    algopt.GrdWeight = &gw[0]

    if W.size > 1:
        algopt.Weight = &W[0,0]
        algopt.nWeight = W.size

    cuda_wrapper_CBPDN_GR(&D[0,0,0], &S[0,0], lmbda, mu, &algopt, &X[0,0,0])

    return X

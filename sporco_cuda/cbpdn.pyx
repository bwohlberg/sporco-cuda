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

    X = _cbpdn(np.ascontiguousarray(np.rollaxis(D, 2, 0)),
               np.ascontiguousarray(S), lmbda, dict(opt), dev)
    return np.rollaxis(X, 0, 3)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _cbpdn(np.ndarray[DTYPE_t, ndim=3, mode="c"] D,
          np.ndarray[DTYPE_t, ndim=2, mode="c"] S,
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

    cuda_wrapper_CBPDN(&D[0,0,0], &S[0,0], lmbda, &algopt, &X[0,0,0])

    return X



@cython.boundscheck(False)
@cython.wraparound(False)
def cbpdngrd(np.ndarray[DTYPE_t, ndim=3] D not None,
          np.ndarray[DTYPE_t, ndim=2] S not None,
          DTYPE_t lmbda, DTYPE_t mu, opt, int dev=0):

    X = _cbpdngrd(np.ascontiguousarray(np.rollaxis(D, 2, 0)),
               np.ascontiguousarray(S), lmbda, mu, dict(opt), dev)
    return np.rollaxis(X, 0, 3)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _cbpdngrd(np.ndarray[DTYPE_t, ndim=3, mode="c"] D,
             np.ndarray[DTYPE_t, ndim=2, mode="c"] S,
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

    cuda_wrapper_CBPDN_GR(&D[0,0,0], &S[0,0], lmbda, mu, &algopt, &X[0,0,0])

    return X

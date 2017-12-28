sporco_cuda.cbpdn module
========================


.. py:function:: cbpdn(D, S, lmbda, opt, dev=0)

   A GPU-accelerated version of :class:`sporco.admm.cbpdn.ConvBPDN`.

   Parameters
   ----------
   D : array_like(float32, ndim=3)
     Dictionary array (three dimensional)
   S : array_like(ndim=2)
     Signal array (two dimensional)
   lmbda : float32
     Regularisation parameter
   opt : dict or :class:`sporco.admm.cbpdn.ConvBPDN.Options` object
     Algorithm options (see :ref:`algorithm-options`)
   dev : int
     Device number of GPU device to use

   Returns
   -------
   X : ndarray
     Coefficient map array (sparse representation)


.. py:function:: cbpdngrd(D, s, lmbda, mu, opt, dev=0)

   A GPU-accelerated version of :class:`sporco.admm.cbpdn.ConvBPDNGradReg`.

   Parameters
   ----------
   D : array_like(float32, ndim=3)
     Dictionary array (three dimensional)
   S : array_like(ndim=2)
     Signal array (two dimensional)
   lmbda : float32
     Regularisation parameter (l1)
   mu : float
     Regularisation parameter (gradient)
   opt : dict or :class:`sporco.admm.cbpdn.ConvBPDNGradReg.Options` object
     Algorithm options (see :ref:`algorithm-options`)
   dev : int
     Device number of GPU device to use

   Returns
   -------
   X : ndarray
     Coefficient map array (sparse representation)



.. _algorithm-options:

Algorithm Options
-----------------

The algorithm options parameter may either be an appropriate ``sporco`` options object (:class:`sporco.admm.cbpdn.ConvBPDN.Options` or :class:`sporco.admm.cbpdn.ConvBPDNGradReg.Options`), or a `dict` with the following entries:

    ``Verbose`` : Flag determining whether iteration status is displayed.

    ``MaxMainIter`` : Maximum main iterations.

    ``AbsStopTol`` : Absolute convergence tolerance (see the docs for
    :class:`sporco.admm.admm.ADMM.Options`).

    ``RelStopTol`` : Relative convergence tolerance (see the docs for
    :class:`sporco.admm.admm.ADMM.Options`).

    ``RelaxParam`` : Relaxation parameter (see the docs for
    :class:`sporco.admm.admm.ADMM.Options`). Relaxation is
    disabled by setting this value to 1.0.

    ``rho`` : ADMM penalty parameter :math:`\rho`.

    ``AutoRho`` : Options for adaptive rho strategy. The value of this
    dict key should itself be a dict with the following entries (see the
    docs for :class:`sporco.admm.admm.ADMM.Options` for more detail):

	``Enabled`` : Flag determining whether adaptive penalty parameter
	strategy is enabled.

	``Period`` : Iteration period on which rho is updated. If set to
	1, the rho update test is applied at every iteration.

	``Scaling`` : Multiplier applied to rho when updated.

	``RsdlRatio`` : Primal/dual residual ratio in rho update test.

	``RsdlTarget`` : Residual ratio targeted by auto rho update
	policy.

	``AutoScaling`` : Flag determining whether RhoScaling value is
	adaptively determined. If enabled, ``Scaling`` specifies a maximum a
	llowed multiplier instead of a fixed multiplier.

	``StdResiduals`` : Flag determining whether standard residual
	definitions are used instead of normalised residuals.

    ``AuxVarObj`` : Flag indicating whether the objective
    function should be evaluated using variable X (``False``) or
    Y (``True``) as its argument. Setting this flag to ``True``
    often gives a better estimate of the objective function, but
    at additional computational cost.

    ``HighMemSolve`` : Flag indicating whether to use a slightly
    faster algorithm at the expense of higher memory usage.

    ``NonNegCoef`` : Flag indicating whether to force solution to
    be non-negative.

    ``NoBndryCross`` : Flag indicating whether all solution
    coefficients corresponding to filters crossing the image
    boundary should be forced to zero.

    ``L1Weight`` : An array of weights for the :math:`\ell_1`
    norm (see the docs for :class:`sporco.admm.cbpdn.GenericConvBPDN.Options`
    for more detail).

    ``GradWeight`` : An array of weights :math:`w_m` for the term
    penalising the gradient of the coefficient maps (see the docs for
    :class:`sporco.admm.cbpdn.ConvBPDNGradReg.Options` for more detail).
    **NB**: This option is only relevant to :func:`.cbpdngrd`.

Note that entries in the ``sporco`` options objects that are not listed above are silently ignored.

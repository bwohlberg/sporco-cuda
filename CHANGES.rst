=========================
SPORCO-CUDA Release Notes
=========================


Version 0.0.6   (not yet released)
----------------------------------

• Update cuda arch codes in setup
• Resolve deprecation warnings due to changes in sporco


Version 0.0.5   (2020-06-22)
----------------------------

• Fixed omission of a source file from MANIFEST.in


Version 0.0.4   (2019-12-06)
----------------------------

• Fixed a number of bugs, including kernels not running on some GPU
  architectures (e.g. GTX Titan X)


Version 0.0.3   (2018-08-10)
----------------------------

• Fixed bug encountered when making multiple calls to cbpdnmsk or
  cbpdngrdmsk
• Added missing memory deallocation in cbpdn.cu and cbpdn_grd.cu


Version 0.0.2   (2018-06-22)
----------------------------

• Improvements to docs
• Fixed bug in handling of l1 weighting option
• New solvers for problems with a spatial mask in the data fidelity term


Version 0.0.1   (2018-04-06)
----------------------------

• Initial release

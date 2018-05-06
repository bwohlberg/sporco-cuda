//  Author: Gustavo Silva <gustavo.silva@pucp.edu.pe>

#ifndef _CUDA_CBPDN_H_
#define _CUDA_CUCBPDN_H_

#ifdef __cplusplus
extern "C" {
#endif

void cuda_wrapper_CBPDN(float *D, float *S, float lambda, void *opt, float *Y);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _END_  _CUDA_CBPDN_H_*/

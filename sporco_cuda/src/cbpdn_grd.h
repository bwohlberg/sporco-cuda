//  Author: Gustavo Silva <gustavo.silva@pucp.edu.pe>

#ifndef _CUDA_CBPDN_GR_H_
#define _CUDA_CUCBPDN_GR_H_

#ifdef __cplusplus
extern "C" {
#endif

void cuda_wrapper_CBPDN_GR(float *D, float *S, float lambda, float mu,
                           void *opt, float *Y);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _END_  _CUDA_CBPDN_GR_H_*/

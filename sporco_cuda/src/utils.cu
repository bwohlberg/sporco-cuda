//  Author: Gustavo Silva <gustavo.silva@pucp.edu.pe>

#include "algopt.h"
#include "common.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void default_opts(void *data) {

  AlgOpt *opt = ((AlgOpt *)data);

  // -1 was given in place of a necessarily nonzero or NULL value

  if (opt->device == -1)
    opt->device = 0;

  if (opt->rho == -1)
    opt->rho = 0;

  if (opt->Verbose == -1)
    opt->Verbose = 0;
  if (opt->AutoRho == -1)
    opt->AutoRho = 1;

  if (opt->nWeight == -1)
    opt->nWeight = 1;

  if (opt->AuxVarObj == -1)
    opt->AuxVarObj = 0;

  if (opt->AbsStopTol == -1)
    opt->AbsStopTol = 0;
  if (opt->NonNegCoef == -1)
    opt->NonNegCoef = 0;
  if (opt->RelaxParam == -1)
    opt->RelaxParam = 1.8;
  if (opt->RhoScaling == -1)
    opt->RhoScaling = 100;

  if (opt->RelStopTol == -1)
    opt->RelStopTol = 1e-4;
  if (opt->MaxMainIter == -1)
    opt->MaxMainIter = 250;

  if (opt->NoBndryCross == -1)
    opt->NoBndryCross = 0;
  if (opt->HighMemSolve == -1)
    opt->HighMemSolve = 0;
  if (opt->StdResiduals == -1)
    opt->StdResiduals = 0;
  if (opt->RhoRsdlRatio == -1)
    opt->RhoRsdlRatio = 1.2;

  if (opt->RhoRsdlTarget == -1)
    opt->RhoRsdlTarget = -1;
  if (opt->AutoRhoPeriod == -1)
    opt->AutoRhoPeriod = 1;
  if (opt->AutoRhoScaling == -1)
    opt->AutoRhoScaling = 1;
}

void clear_opts(void *data) {

  AlgOpt *opt = ((AlgOpt *)data);

  opt->device = -1;

  opt->rho = -1;
  opt->Verbose = -1;
  opt->AutoRho = -1;

  opt->L1Weight = ((float *)calloc(1, sizeof(float)));
  opt->Weight = ((int *)calloc(1, sizeof(int)));
  opt->nWeight = -1;

  opt->AuxVarObj = -1;

  opt->AbsStopTol = -1;
  opt->NonNegCoef = -1;
  opt->RelaxParam = -1;
  opt->RhoScaling = -1;

  opt->RelStopTol = -1;
  opt->MaxMainIter = -1;

  opt->NoBndryCross = -1;
  opt->HighMemSolve = -1;
  opt->StdResiduals = -1;
  opt->RhoRsdlRatio = -1;

  opt->RhoRsdlTarget = -1;
  opt->AutoRhoPeriod = -1;
  opt->AutoRhoScaling = -1;
}

int check_cuda_req(int device) {

  int devCount;
  cudaError_t cuError;
  struct cudaDeviceProp properties;

  cuError = cudaGetDeviceCount(&devCount);
  if (cuError == cudaSuccess) {
    // printf("Number of CUDA Devices : %d \n\n", devCount);
  } else if (cuError == cudaErrorNoDevice) {
    printf("ERROR - There are not CUDA Devices \n\n");
    exit(EXIT_FAILURE);
  } else if (cuError == cudaErrorInsufficientDriver) {
    printf("ERROR - Insufficient Driver \n\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaGetDeviceProperties(&properties, device));

  if (properties.major >= 3) {
    // printf("Running Sample on GPU %d \n\n", device);
  } else {
    printf("Error - Non CUDA GPU with architecture SM 3.0 or higher \n\n");
    printf("Select other GPU  \n\n");
    exit(EXIT_FAILURE);
  }

  // Iterate through devices
  /*
  for (k = 0; k < devCount; ++k) {
    // Get device properties
    checkCudaErrors(cudaGetDeviceProperties(&properties, k));
    if ( (properties.major >= 3) && (properties.computeMode == 1) ) {
      printf("Running Sample on GPU %d \n\n", k);
      break;
    } else if (k == (devCount-1)) {
        printf("Non CUDA Device avalible with architecture SM 3.0 or "
               "higher \n\n");
        exit(EXIT_FAILURE);
      }
  }
  */

  return device;
}

int get_device_count(void) {
  int devCount;
  cudaError_t cuError;

  cuError = cudaGetDeviceCount(&devCount);
  if (cuError == cudaSuccess)
    return devCount;
  else
    return 0;
}

cudaError_t get_current_device(int *dev) { return cudaGetDevice(dev); }

cudaError_t set_current_device(int dev) { return cudaSetDevice(dev); }

cudaError_t get_memory_info(size_t *free, size_t *total) {
  return cudaMemGetInfo(free, total);
}

char *get_device_name(int dev) {
  static cudaDeviceProp cdp;
  cudaError_t cuError;

  cuError = cudaGetDeviceProperties(&cdp, dev);
  if (cuError == cudaSuccess)
    return cdp.name;
  else
    return NULL;
}

#ifdef __cplusplus
} // extern "C"
#endif

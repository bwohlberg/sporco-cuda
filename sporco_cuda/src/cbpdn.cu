/*******************************************************************/
/****       CBPDN - Convolutional Basis Pursuit DeNoising       ****/
/*******************************************************************/

/*
         argmin_{x_m} (1/2)||\sum_m d_m * x_m - s||_2^2 +
                            lambda \sum_m ||x_m||_1

         The solution is computed using an ADMM approach (see
         boyd-2010-distributed) with efficient solution of the main
         linear systems (see wohlberg-2016-efficient).


  Usage:
        cuda_wrapper_CBPDN(float *D, float *S, float lambda, void *opt,
                           float *Y);

  Input:
         D           Dictionary filter set (3D array)
         S           Input image
         lambda      Regularization parameter
         opt         Algorithm parameters structure
         Y           Dictionary coefficient map set (3D array)


  Options structure fields:

     Verbose          Flag determining whether iteration status is displayed.
                      Fields are iteration number, functional value,
                      data fidelity term, l1 regularisation term, and
                      primal and dual residuals (see Sec. 3.3 of
                      boyd-2010-distributed). The value of rho is also
                      displayed if options request that it is automatically
                      adjusted.
     MaxMainIter      Maximum main iterations
     AbsStopTol       Absolute convergence tolerance (see Sec. 3.3.1 of
                      boyd-2010-distributed)
     RelStopTol       Relative convergence tolerance (see Sec. 3.3.1 of
                      boyd-2010-distributed)
     L1Weight         Weighting array for coefficients in l1 norm of X
     rho              ADMM penalty parameter
     AutoRho          Flag determining whether rho is automatically updated
                      (see Sec. 3.4.1 of boyd-2010-distributed)
     AutoRhoPeriod    Iteration period on which rho is updated
     RhoRsdlRatio     Primal/dual residual ratio in rho update test
     RhoScaling       Multiplier applied to rho when updated
     AutoRhoScaling   Flag determining whether RhoScaling value is
                      adaptively determined (see wohlberg-2015-adaptive). If
                      enabled, RhoScaling specifies a maximum allowed
                      multiplier instead of a fixed multiplier.
     RhoRsdlTarget    Residual ratio targeted by auto rho update policy.
     StdResiduals     Flag determining whether standard residual definitions
                      (see Sec 3.3 of boyd-2010-distributed) are used instead
                      of normalised residuals (see wohlberg-2015-adaptive)
     RelaxParam       Relaxation parameter (see Sec. 3.4.3 of
                      should be forced to zero.
     AuxVarObj        Flag determining whether objective function is computed
                      using the auxiliary (split) variable
     HighMemSolve     Use more memory for a slightly faster solution


  Author: Gustavo Silva <gustavo.silva@pucp.edu.pe>

*/

// stdlib includes
#include <complex.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA library includes
#include "cublas_v2.h" // perform linear algebra operations on the GPU
#include "cufft.h"     // perform FFT on the GPU

// local includes
#include "algopt.h" // Contains the parameters structure
#include "common.h"
#include "utils.h"
// extern "C" {
#include "cbpdn.h"
//}
#include "cbpdn_kernels.h"

// extern "C"
void cuda_wrapper_CBPDN(float *D, float *S, float lambda, void *vopt,
                        float *Y) {
  AlgOpt *opt = ((AlgOpt *)vopt);

  // Checks and sets default options
  default_opts(opt);

  /*************************************/
  /****   Check CUDA requirements   ****/
  /*************************************/

  int k = check_cuda_req(opt->device);

  checkCudaErrors(cudaSetDevice(k));

  /******************************/
  /****   Define Variables   ****/
  /******************************/

  float *d_S, *d_D, *d_Y, *d_X, *d_Xr, *d_U, *d_aux, *d_Yprv, *d_L1Weight;
  int *d_Weight;

  cufftComplex *d_auxf, *d_Df, *d_Dsf, *d_C, *d_Sf;

  dim3 threadsPerBlock;
  dim3 blocksPerGrid;
  dim3 blocksPerGrid_vec4;
  dim3 blocksPerGrid_fft;
  dim3 blocksPerGrid_Padding;
  dim3 blocksPerGrid_radix2_fft;

  int IMG_ROW_SIZE = opt->IMG_ROW_SIZE;
  int IMG_COL_SIZE = opt->IMG_COL_SIZE;
  int DICT_M_SIZE = opt->DICT_M_SIZE;
  int DICT_ROW_SIZE = opt->DICT_ROW_SIZE;
  int DICT_COL_SIZE = opt->DICT_COL_SIZE;

  int L1_WEIGHT_M_SIZE = opt->L1_WEIGHT_M_SIZE;
  int L1_WEIGHT_ROW_SIZE = opt->L1_WEIGHT_ROW_SIZE;
  int L1_WEIGHT_COL_SIZE = opt->L1_WEIGHT_COL_SIZE;
  int nL1Weight = L1_WEIGHT_ROW_SIZE * L1_WEIGHT_COL_SIZE * L1_WEIGHT_M_SIZE;

  int nWeight = opt->nWeight;
  int W_flag = (nWeight > 1) ? 1 : 0;
  int offset_W;

  int MAX_ITER = opt->MaxMainIter;

  int BATCH = DICT_M_SIZE;
  int LIMIT_X = IMG_COL_SIZE / 2 + 1;
  int n[2] = {IMG_ROW_SIZE, IMG_COL_SIZE};

  int inembed[2] = {IMG_ROW_SIZE, IMG_COL_SIZE};
  int onembed[2] = {IMG_ROW_SIZE, LIMIT_X};

  float rho = opt->rho;

  int SIZE_S = IMG_ROW_SIZE * IMG_COL_SIZE;
  int SIZE_X = IMG_ROW_SIZE * IMG_COL_SIZE * DICT_M_SIZE;
  int SIZE_D = DICT_ROW_SIZE * DICT_COL_SIZE * DICT_M_SIZE;

  int SIZE_Sf = IMG_ROW_SIZE * LIMIT_X;
  int SIZE_Xf = IMG_ROW_SIZE * LIMIT_X * DICT_M_SIZE;

  float *d_r, *d_ss, *d_nX, *d_nY, *d_nU;
  float nX, nY, nU, rhomlt, rsf;
  float *d_JL1, *d_Jdf;
  float r = FLT_MAX;
  float s = FLT_MAX;
  float epri = 0;
  float edua = 0;

  if (opt->StdResiduals == 1)
    opt->RhoRsdlTarget = 1;
  else
    opt->RhoRsdlTarget = 1 + powf(18.3, log10(lambda) + 1);

  cublasHandle_t cublas_handle;
  checkCudaErrors(cublasCreate(&cublas_handle));

  cufftHandle planfft_forward_many;
  cufftHandle planfft_forward_many_Sf;
  cufftHandle planfft_reverse_many;

  checkCudaErrors(cufftPlanMany(&planfft_forward_many, 2, n, inembed, 1,
                                IMG_COL_SIZE * IMG_ROW_SIZE, onembed, 1,
                                LIMIT_X * IMG_ROW_SIZE, CUFFT_R2C, BATCH));

  checkCudaErrors(cufftPlanMany(&planfft_forward_many_Sf, 2, n, inembed, 1,
                                IMG_COL_SIZE * IMG_ROW_SIZE, onembed, 1,
                                LIMIT_X * IMG_ROW_SIZE, CUFFT_R2C, 1));

  checkCudaErrors(cufftPlanMany(&planfft_reverse_many, 2, n, NULL, 1, 0, NULL,
                                1, 0, CUFFT_C2R, BATCH));

  /***************************/
  /****   Allocate data   ****/
  /***************************/

  // Unified Memory (necessary variables)
  checkCudaErrors(cudaMallocManaged(&d_Y, ((int)SIZE_X * sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_S, SIZE_S * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_D, SIZE_D * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(&d_r, ((int)1 * sizeof(float))));
  checkCudaErrors(cudaMallocManaged(&d_ss, ((int)1 * sizeof(float))));
  checkCudaErrors(cudaMallocManaged(&d_nX, ((int)1 * sizeof(float))));
  checkCudaErrors(cudaMallocManaged(&d_nY, ((int)1 * sizeof(float))));
  checkCudaErrors(cudaMallocManaged(&d_nU, ((int)1 * sizeof(float))));

  // Global Memory (necessary variables)
  checkCudaErrors(cudaMalloc((void **)&d_X, SIZE_X * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_U, SIZE_X * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_Yprv, SIZE_X * sizeof(float)));

  checkCudaErrors(cudaMalloc((void **)&d_C, SIZE_Xf * sizeof(float2)));
  checkCudaErrors(cudaMalloc((void **)&d_Df, SIZE_Xf * sizeof(float2)));
  checkCudaErrors(cudaMalloc((void **)&d_Dsf, SIZE_Xf * sizeof(float2)));

  // Global Memory (auxiliary variables)
  checkCudaErrors(cudaMalloc((void **)&d_Sf, SIZE_Sf * sizeof(float2)));
  checkCudaErrors(cudaMalloc((void **)&d_aux, SIZE_X * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_auxf, SIZE_Xf * sizeof(float2)));

  checkCudaErrors(cudaMalloc((void **)&d_L1Weight, nL1Weight * sizeof(float)));

  if (nWeight > 1) {
    checkCudaErrors(cudaMalloc((void **)&d_Weight, nWeight * sizeof(int)));
  }

  /******************************************/
  /****   Define Block and Thread Size   ****/
  /******************************************/

  threadsPerBlock.x = 64;
  threadsPerBlock.y = 8;
  threadsPerBlock.z = 1;

  blocksPerGrid.x = ((IMG_COL_SIZE - 1) / threadsPerBlock.x) + 1;
  blocksPerGrid.y = ((IMG_ROW_SIZE - 1) / threadsPerBlock.y) + 1;
  blocksPerGrid.z = 1;

  blocksPerGrid_vec4.x = ((IMG_COL_SIZE - 1) / threadsPerBlock.x) + 1;
  blocksPerGrid_vec4.y = ((IMG_ROW_SIZE / 4 - 1) / threadsPerBlock.y) + 1;
  blocksPerGrid_vec4.z = 1;

  blocksPerGrid_fft.x = ((LIMIT_X - 1) / threadsPerBlock.x) + 1;
  blocksPerGrid_fft.y = ((IMG_ROW_SIZE - 1) / threadsPerBlock.y) + 1;
  blocksPerGrid_fft.z = 1;

  blocksPerGrid_radix2_fft.x = ((LIMIT_X - 1) / threadsPerBlock.x) + 1;
  blocksPerGrid_radix2_fft.y = ((IMG_ROW_SIZE / 2 - 1) / threadsPerBlock.y) + 1;
  blocksPerGrid_radix2_fft.z = 1;

  blocksPerGrid_Padding.x = ((DICT_COL_SIZE - 1) / threadsPerBlock.x) + 1;
  blocksPerGrid_Padding.y = ((DICT_ROW_SIZE - 1) / threadsPerBlock.y) + 1;
  blocksPerGrid_Padding.z = 1;

  /***********************************************/
  /****   Transfer data from host to device   ****/
  /***********************************************/

  checkCudaErrors(
      cudaMemcpy(d_S, S, SIZE_S * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_D, D, SIZE_D * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_L1Weight, opt->L1Weight,
                             nL1Weight * sizeof(float),
                             cudaMemcpyHostToDevice));

  if (nWeight > 1) {
    checkCudaErrors(cudaMemcpy(d_Weight, opt->Weight, nWeight * sizeof(int),
                               cudaMemcpyHostToDevice));
  }

  checkCudaErrors(cudaDeviceSynchronize());

  /****************************************/
  /****   Calculates Fixed Variables   ****/
  /****************************************/

  // Share the same memory space in order to reduce expenses
  d_Xr = d_aux;

  checkCudaErrors(cudaMemset(d_aux, 0, SIZE_X * sizeof(float)));
  checkCudaErrors(cudaMemset(d_Yprv, 0, SIZE_X * sizeof(float)));

  // Sets padded dictionary
  cuda_Pad_Dict<<<blocksPerGrid_Padding, threadsPerBlock>>>(
      d_aux, d_D, DICT_ROW_SIZE, DICT_COL_SIZE, DICT_M_SIZE, IMG_ROW_SIZE,
      IMG_COL_SIZE);

  // Calculates "Sf'" = FFT2(S)
  checkCudaErrors(cufftExecR2C(planfft_forward_many_Sf, ((cufftReal *)d_S),
                               ((cufftComplex *)d_Sf)));

  // Calculates '"Df" = FFT2(D_padded)
  checkCudaErrors(cufftExecR2C(planfft_forward_many, ((cufftReal *)d_aux),
                               ((cufftComplex *)d_Df)));

  if (IMG_ROW_SIZE % 2) {
    // Calculates "Dsf" and "C" (both in a kernel)
    cuda_Cal_Dsf_C<<<blocksPerGrid_fft, threadsPerBlock>>>(
        d_Dsf, d_C, d_Sf, d_Df, rho, IMG_ROW_SIZE, LIMIT_X, DICT_M_SIZE);
  } else {
    cuda_Cal_Dsf_C_vec4<<<blocksPerGrid_radix2_fft, threadsPerBlock>>>(
        d_Dsf, d_C, d_Sf, d_Df, rho, IMG_ROW_SIZE, LIMIT_X, DICT_M_SIZE);
  }

  checkCudaErrors(cudaDeviceSynchronize());

  d_nX[0] = 0.0;
  d_nY[0] = 0.0;
  d_nU[0] = 0.0;
  d_ss[0] = 0.0;
  d_r[0] = 0.0;

  const char *header = "Itn   Fnc       DFid      Regℓ1     r         "
                       "s         ρ";
  const char *sepstr = "----------------------------------------------"
                       "------------------";

  /*******************************/
  /****   Main Loop (CBPDN)   ****/
  /*******************************/

  int i = 1;
  while (i <= MAX_ITER && (r > epri || s > edua)) {

    if (i == 1) {
      // Defines U and FFT(Y-U) initial values as zero arrays
      checkCudaErrors(cudaMemset(d_auxf, 0, SIZE_Xf * sizeof(float2)));
      checkCudaErrors(cudaMemset(d_U, 0, SIZE_X * sizeof(float)));
      checkCudaErrors(cudaMemset(d_Y, 0, SIZE_X * sizeof(float)));
    } else {
      // Calculates Y-U
      cuda_CalYU_vec4<<<blocksPerGrid, threadsPerBlock>>>(
          d_aux, d_Y, d_U, IMG_ROW_SIZE, IMG_COL_SIZE, DICT_M_SIZE);

      // Calculates FFT2(Y-U)
      checkCudaErrors(cufftExecR2C(planfft_forward_many, ((cufftReal *)d_aux),
                                   ((cufftComplex *)d_auxf)));
    }

    // Calculates Xf
    if (IMG_ROW_SIZE % 2) {
      cuda_solvedbi_sm<<<blocksPerGrid_fft, threadsPerBlock>>>(
          d_auxf, d_Df, d_Dsf, rho, d_auxf, d_C, IMG_ROW_SIZE, LIMIT_X,
          DICT_M_SIZE);
    } else {
      cuda_solvedbi_sm_vec4<<<blocksPerGrid_radix2_fft, threadsPerBlock>>>(
          d_auxf, d_Df, d_Dsf, rho, d_auxf, d_C, IMG_ROW_SIZE, LIMIT_X,
          DICT_M_SIZE);
    }

    // Calculates  X = IFFT(Xf)
    checkCudaErrors(cufftExecC2R(planfft_reverse_many, ((cufftComplex *)d_auxf),
                                 ((cufftReal *)d_X)));

    // Calculates Xr (over-relaxation of X)
    // See pg. 21 of boyd-2010-distributed
    if (opt->RelaxParam != 1.0) {
      cuda_OverRelax_vec4<<<blocksPerGrid, threadsPerBlock>>>(
          d_Xr, d_X, d_Y, opt->RelaxParam, IMG_ROW_SIZE, IMG_COL_SIZE,
          DICT_M_SIZE);
    } else {
      d_Xr = d_X;
    }

    // Calculates Y = shinkage(X + U, L1Weight*lamdba/rho) and U = U + X - Y
    // (both steps in a single kernel in order to reduce memory access cost)

    offset_W = IMG_ROW_SIZE * IMG_COL_SIZE * W_flag;

    if (nL1Weight == DICT_M_SIZE) {
      if ((IMG_ROW_SIZE % 4) == 0) {
        cuda_Shrink_CalU_vec4_Vector<<<blocksPerGrid_vec4, threadsPerBlock>>>(
            &d_Y[offset_W], &d_U[offset_W], &d_Xr[offset_W], lambda / rho,
            &d_L1Weight[W_flag], IMG_ROW_SIZE, IMG_COL_SIZE,
            DICT_M_SIZE - W_flag);
      } else {
        cuda_Shrink_CalU_Vector<<<blocksPerGrid, threadsPerBlock>>>(
            &d_Y[offset_W], &d_U[offset_W], &d_Xr[offset_W], lambda / rho,
            &d_L1Weight[W_flag], IMG_ROW_SIZE, IMG_COL_SIZE,
            DICT_M_SIZE - W_flag);
      }
    } else if (nL1Weight == 1) {
      cuda_Shrink_CalU_vec4_Scalar<<<blocksPerGrid, threadsPerBlock>>>(
          &d_Y[offset_W], &d_U[offset_W], &d_Xr[offset_W], lambda / rho,
          d_L1Weight, IMG_ROW_SIZE, IMG_COL_SIZE, DICT_M_SIZE - W_flag);
    } else {
      cuda_Shrink_CalU_vec4_Array<<<blocksPerGrid, threadsPerBlock>>>(
          &d_Y[offset_W], &d_U[offset_W], &d_Xr[offset_W], lambda / rho,
          &d_L1Weight[offset_W], IMG_ROW_SIZE, IMG_COL_SIZE,
          DICT_M_SIZE - W_flag);
    }

    if (W_flag == 1) {
      cuda_Cal_X_minus_U_W<<<blocksPerGrid, threadsPerBlock>>>(
          d_Y, d_U, d_Xr, d_Weight, IMG_ROW_SIZE, IMG_COL_SIZE);
    }

    // compute distances between X and Y,  Y and Yprv
    // See pp. 19-20 of boyd-2010-distributed
    cuda_Cal_residuals_norms_vec4<<<blocksPerGrid, threadsPerBlock>>>(
        d_ss, d_r, d_nX, d_nY, d_nU, d_X, d_Y, d_U, d_Yprv, rho, IMG_ROW_SIZE,
        IMG_COL_SIZE, DICT_M_SIZE);

    checkCudaErrors(cudaDeviceSynchronize());

    // nX, nY, nU, r, s are divided by 1e+6 since kernels use atomic
    // functions (big calculation error if the reduction values are small)
    nX = sqrtf(d_nX[0] / 1e+6);
    nY = sqrtf(d_nY[0] / 1e+6);
    nU = sqrtf(d_nU[0] / 1e+6);

    r = sqrtf(d_r[0] / 1e+6);
    s = sqrtf(d_ss[0] / 1e+6);

    d_nX[0] = 0.0;
    d_nY[0] = 0.0;
    d_nU[0] = 0.0;
    d_ss[0] = 0.0;
    d_r[0] = 0.0;

    if (opt->StdResiduals != 1) {
      if (nU == 0.0)
        nU = 1.0;
    }

    epri = sqrtf(nX) * opt->AbsStopTol + max(nX, nY) * opt->RelStopTol;
    edua = sqrtf(nX) * opt->AbsStopTol + rho * nU * opt->RelStopTol;

    if (opt->StdResiduals != 1) {
      // See wohlberg-2015-adaptive
      r /= max(nX, nY);
      s /= (rho * nU);
      epri /= max(nX, nY);
      edua /= (rho * nU);
    }

    if (opt->Verbose == 1) {

      if (i == 1) {
        printf("%s\n%s\n", header, sepstr);
        checkCudaErrors(cudaMallocManaged(&d_JL1, ((int)sizeof(float))));
        checkCudaErrors(cudaMallocManaged(&d_Jdf, ((int)sizeof(float))));
      }

      d_JL1[0] = 0.0;
      d_Jdf[0] = 0.0;

      // Compute data fidelity term in Fourier domain (note normalisation)
      if (opt->AuxVarObj == 1) {
        // Calculates FFT2(Y)
        checkCudaErrors(cufftExecR2C(planfft_forward_many, ((cufftReal *)d_Y),
                                     ((cufftComplex *)d_auxf)));

        if (IMG_ROW_SIZE % 2)
          cuda_Fidelity_Term<<<blocksPerGrid_fft, threadsPerBlock>>>(
              d_Jdf, d_Df, d_auxf, d_Sf, IMG_ROW_SIZE, LIMIT_X, DICT_M_SIZE);
        else
          cuda_Fidelity_Term_vec4<<<blocksPerGrid_radix2_fft,
                                    threadsPerBlock>>>(
              d_Jdf, d_Df, d_auxf, d_Sf, IMG_ROW_SIZE, LIMIT_X, DICT_M_SIZE);

        if (nL1Weight == DICT_M_SIZE) {
          if ((IMG_ROW_SIZE % 4) == 0) {
            cuda_L1_Term_vec4_Vector<<<blocksPerGrid_vec4, threadsPerBlock>>>(
                d_JL1, d_Y, d_L1Weight, 1, IMG_ROW_SIZE, IMG_COL_SIZE,
                DICT_M_SIZE);
          } else {
            cuda_L1_Term_Vector<<<blocksPerGrid, threadsPerBlock>>>(
                d_JL1, d_Y, d_L1Weight, 1, IMG_ROW_SIZE, IMG_COL_SIZE,
                DICT_M_SIZE);
          }
        } else {
          cuda_L1_Term_vec4_Scalar_Array<<<blocksPerGrid, threadsPerBlock>>>(
              d_JL1, d_Y, d_L1Weight, 1, IMG_ROW_SIZE, IMG_COL_SIZE,
              DICT_M_SIZE, nL1Weight);
        }

      } else {

        if (IMG_ROW_SIZE % 2)
          cuda_Fidelity_Term<<<blocksPerGrid_fft, threadsPerBlock>>>(
              d_Jdf, d_Df, d_auxf, d_Sf, IMG_ROW_SIZE, LIMIT_X, DICT_M_SIZE);
        else
          cuda_Fidelity_Term_vec4<<<blocksPerGrid_radix2_fft,
                                    threadsPerBlock>>>(
              d_Jdf, d_Df, d_auxf, d_Sf, IMG_ROW_SIZE, LIMIT_X, DICT_M_SIZE);

        if (nL1Weight == DICT_M_SIZE) {
          if ((IMG_ROW_SIZE % 4) == 0) {
            cuda_L1_Term_vec4_Vector<<<blocksPerGrid_vec4, threadsPerBlock>>>(
                d_JL1, d_X, d_L1Weight, (IMG_ROW_SIZE * IMG_COL_SIZE),
                IMG_ROW_SIZE, IMG_COL_SIZE, DICT_M_SIZE);
          } else {
            cuda_L1_Term_Vector<<<blocksPerGrid, threadsPerBlock>>>(
                d_JL1, d_X, d_L1Weight, (IMG_ROW_SIZE * IMG_COL_SIZE),
                IMG_ROW_SIZE, IMG_COL_SIZE, DICT_M_SIZE);
          }
        } else {
          cuda_L1_Term_vec4_Scalar_Array<<<blocksPerGrid, threadsPerBlock>>>(
              d_JL1, d_X, d_L1Weight, (IMG_ROW_SIZE * IMG_COL_SIZE),
              IMG_ROW_SIZE, IMG_COL_SIZE, DICT_M_SIZE, nL1Weight);
        }
      }

      checkCudaErrors(cudaDeviceSynchronize());

      float Jdf = d_Jdf[0] / (2 * IMG_ROW_SIZE * LIMIT_X);
      float Jl1 = d_JL1[0];
      float Jfn = Jdf + lambda * Jl1;

      // Display iteration details
      printf("%4d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", i - 1, Jfn, Jdf, Jl1,
             r, s, rho);
    }

    // See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed

    if ((opt->AutoRho == 1) && (i > 1) && (i % opt->AutoRhoPeriod == 0)) {

      if (opt->AutoRhoScaling == 1) {
        rhomlt = sqrtf(r / (s * opt->RhoRsdlTarget));

        if (rhomlt < 1)
          rhomlt = 1 / rhomlt;
        if (rhomlt > opt->RhoScaling)
          rhomlt = opt->RhoScaling;
      } else {
        rhomlt = opt->RhoScaling;
      }

      rsf = 1;

      if (r > opt->RhoRsdlTarget * opt->RhoRsdlRatio * s)
        rsf = rhomlt;
      if (s > (opt->RhoRsdlRatio / opt->RhoRsdlTarget) * r)
        rsf = 1 / rhomlt;

      rho = rsf * rho;

      // U = U / rsf
      float zero = 0;
      float rsf_inv = 1 / rsf;
      checkCudaErrors(cublasSgeam(
          cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, (IMG_ROW_SIZE * DICT_M_SIZE),
          IMG_ROW_SIZE, &rsf_inv, d_U, (IMG_ROW_SIZE * DICT_M_SIZE), &zero, d_U,
          (IMG_ROW_SIZE * DICT_M_SIZE), d_U, (IMG_ROW_SIZE * DICT_M_SIZE)));

      if ((opt->HighMemSolve == 1) && (rsf != 1)) {
        if (IMG_ROW_SIZE % 2)
          cuda_Cal_C<<<blocksPerGrid_fft, threadsPerBlock>>>(
              d_C, d_Df, rho, IMG_ROW_SIZE, LIMIT_X, DICT_M_SIZE);
        else
          cuda_Cal_C_vec4<<<blocksPerGrid_radix2_fft, threadsPerBlock>>>(
              d_C, d_Df, rho, IMG_ROW_SIZE, LIMIT_X, DICT_M_SIZE);
      }
    }

    checkCudaErrors(cudaMemcpy(d_Yprv, d_Y, SIZE_X * sizeof(float),
                               cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaDeviceSynchronize());

    i++;
  }

  if (opt->Verbose == 1) {
    printf("%s\n", sepstr);
  }

  checkCudaErrors(cudaDeviceSynchronize());

  memcpy(Y, d_Y, (int)SIZE_X * sizeof(float));

  /****************************/
  /****   Release Memory   ****/
  /****************************/

  checkCudaErrors(cufftDestroy(planfft_forward_many_Sf));
  checkCudaErrors(cufftDestroy(planfft_forward_many));
  checkCudaErrors(cufftDestroy(planfft_reverse_many));

  checkCudaErrors(cudaFree(d_X));
  checkCudaErrors(cudaFree(d_U));
  checkCudaErrors(cudaFree(d_Y));
  checkCudaErrors(cudaFree(d_Yprv));
  checkCudaErrors(cudaFree(d_S));
  checkCudaErrors(cudaFree(d_D));

  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaFree(d_Df));
  checkCudaErrors(cudaFree(d_Sf));
  checkCudaErrors(cudaFree(d_Dsf));

  checkCudaErrors(cudaFree(d_aux));
  checkCudaErrors(cudaFree(d_auxf));

  checkCudaErrors(cudaFree(d_r));
  checkCudaErrors(cudaFree(d_ss));
  checkCudaErrors(cudaFree(d_nX));
  checkCudaErrors(cudaFree(d_nY));
  checkCudaErrors(cudaFree(d_nU));

  if (nWeight > 1)
    checkCudaErrors(cudaFree(d_Weight));
  checkCudaErrors(cudaFree(d_L1Weight));
}

//  Author: Gustavo Silva <gustavo.silva@pucp.edu.pe>

#include <cuComplex.h>

#include "common.h"

/*****************************/
/****   Sherman Morrison  ****/
/*****************************/

// This method provides the solution for x (Output) to
//     (a a^H + rho I) x = b
// Where b = rho*fft2(Y-U) and inner products are taken along the 3rd dimension.
// Only works for even image size and is faster than ypncuda_sherman kernel.

__global__ void cuda_solvedbi_sm_vec4(float2 *Out, float2 *ah, float2 *Dsf,
                                      float rho, float2 *YUf, float2 *c,
                                      int nRows, int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k, half_nRows = nRows / 2;

  float4 a, cb, cba, x, b, YUf_temp;

  cb = make_float4(0, 0, 0, 0);

  if ((Tidx < nCols) & (Tidy < half_nRows)) {

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;

      YUf_temp = reinterpret_cast<float4 *>(YUf)[index];
      YUf_temp = cuCmulf_by_const_vec4(YUf_temp, rho);

      b = cuCaddf_vec4(reinterpret_cast<float4 *>(Dsf)[index], YUf_temp);
      cb = cuCaddf_vec4(cb,
                        cuCmulf_vec4(b, reinterpret_cast<float4 *>(c)[index]));
    }

    a = cuConjf_vec4(reinterpret_cast<float4 *>(ah)[index]);
    cba = cuCmulf_vec4(cb, a);
    x = cuCsubf_vec4(b, cba);

    x = cuCdivf_by_const_vec4(x, rho);
    reinterpret_cast<float4 *>(Out)[index] = x;

    for (k = 0; k < nFilts - 1; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      a = cuConjf_vec4(reinterpret_cast<float4 *>(ah)[index]);
      cba = cuCmulf_vec4(cb, a);

      YUf_temp = reinterpret_cast<float4 *>(YUf)[index];
      YUf_temp = cuCmulf_by_const_vec4(YUf_temp, rho);

      b = cuCaddf_vec4(reinterpret_cast<float4 *>(Dsf)[index], YUf_temp);

      x = cuCsubf_vec4(b, cba);
      x = cuCdivf_by_const_vec4(x, rho);

      reinterpret_cast<float4 *>(Out)[index] = x;
    }
  }
}

// Works for every image size
// This kernel does not work with vectorized data

__global__ void cuda_solvedbi_sm(float2 *Out, float2 *ah, float2 *Dsf,
                                 float rho, float2 *YUf, float2 *c, int nRows,
                                 int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  float2 a, cb, cba, x, b, YUf_temp;

  cb = make_cuFloatComplex(0, 0);

  if ((Tidx < nCols) & (Tidy < nRows)) {

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;

      YUf_temp = YUf[index];
      YUf_temp.x *= rho;
      YUf_temp.y *= rho;

      b = cuCaddf(Dsf[index], YUf_temp);

      cb = cuCaddf(cb, cuCmulf(b, c[index]));
    }

    a = cuConjf(ah[index]);
    cba = cuCmulf(cb, a);

    x = cuCsubf(b, cba);
    x.x /= rho;
    x.y /= rho;
    Out[index] = x;

    for (k = 0; k < nFilts - 1; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      a = cuConjf(ah[index]);
      cba = cuCmulf(cb, a);

      YUf_temp = YUf[index];
      YUf_temp.x *= rho;
      YUf_temp.y *= rho;

      b = cuCaddf(Dsf[index], YUf_temp);

      x = cuCsubf(b, cba);
      x.x /= rho;
      x.y /= rho;
      Out[index] = x;
    }
  }
}

/*****************************************/
/****   Shrinkage and U Calculation   ****/
/*****************************************/

// sets Y[index] = shrink(Xr[index]/(nRows*nCols) + U[index], lambda)
// sets U[index] = U[index] + Xr[index]/(nRows*nCols)  - Y[index]

// Xr[index] is divided by nRows*nCols due to the above operation (cuIFFT)
// This kernel is working with vectorized data (float4)

__global__ void cuda_Shrink_CalU_vec4_Scalar(float *Y, float *U, float *X,
                                             float lambda, float *L1Weight,
                                             int nRows, int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float absxV1, X_tempV1, U_tempV1, Y_tempV1, WLambdaV1;
  float4 absxV4, X_temp, U_temp, Y_temp, WLambda;

  WLambdaV1 = lambda * L1Weight[0];
  WLambda = make_float4(WLambdaV1, WLambdaV1, WLambdaV1, WLambdaV1);

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;

      U_temp = reinterpret_cast<float4 *>(U)[index];
      X_temp = reinterpret_cast<float4 *>(X)[index];

      Y_temp = operator_vec4(X_temp, U_temp, (nRows * nCols));

      absxV4 = fabs_minus_bf4_vec4(Y_temp, WLambda);

      Y_temp.x = signbit(-absxV4.x) * copysign(absxV4.x, Y_temp.x);
      Y_temp.y = signbit(-absxV4.y) * copysign(absxV4.y, Y_temp.y);
      Y_temp.z = signbit(-absxV4.z) * copysign(absxV4.z, Y_temp.z);
      Y_temp.w = signbit(-absxV4.w) * copysign(absxV4.w, Y_temp.w);

      U_temp = operator2_vec4(X_temp, Y_temp, U_temp, (nRows * nCols));

      reinterpret_cast<float4 *>(Y)[index] = Y_temp;
      reinterpret_cast<float4 *>(U)[index] = U_temp;
    }

    // Process remaining elements
    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;

      X_tempV1 = (X[index] / (nRows * nCols));
      U_tempV1 = U[index];

      Y_tempV1 = X_tempV1 + U_tempV1;
      absxV1 = fabs(Y_tempV1) - WLambdaV1;

      Y_tempV1 = signbit(-absxV1) * copysign(absxV1, Y_tempV1);

      Y[index] = Y_tempV1;
      U[index] = U_tempV1 + X_tempV1 - Y_tempV1;
    }
  }
}

__global__ void cuda_Shrink_CalU_vec4_Array(float *Y, float *U, float *X,
                                            float lambda, float *L1Weight,
                                            int nRows, int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float absxV1, X_tempV1, U_tempV1, Y_tempV1, WLambdaV1;
  float4 absxV4, X_temp, U_temp, Y_temp, WLambda;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;

      U_temp = reinterpret_cast<float4 *>(U)[index];
      X_temp = reinterpret_cast<float4 *>(X)[index];

      WLambda = reinterpret_cast<float4 *>(L1Weight)[index];
      WLambda = cuCmulf_by_const_vec4(WLambda, lambda);

      Y_temp = operator_vec4(X_temp, U_temp, (nRows * nCols));

      absxV4 = fabs_minus_bf4_vec4(Y_temp, WLambda);

      Y_temp.x = signbit(-absxV4.x) * copysign(absxV4.x, Y_temp.x);
      Y_temp.y = signbit(-absxV4.y) * copysign(absxV4.y, Y_temp.y);
      Y_temp.z = signbit(-absxV4.z) * copysign(absxV4.z, Y_temp.z);
      Y_temp.w = signbit(-absxV4.w) * copysign(absxV4.w, Y_temp.w);

      U_temp = operator2_vec4(X_temp, Y_temp, U_temp, (nRows * nCols));

      reinterpret_cast<float4 *>(Y)[index] = Y_temp;
      reinterpret_cast<float4 *>(U)[index] = U_temp;
    }

    // Process remaining elements
    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;

      X_tempV1 = (X[index] / (nRows * nCols));
      U_tempV1 = U[index];

      WLambdaV1 = lambda * L1Weight[index];

      Y_tempV1 = X_tempV1 + U_tempV1;
      absxV1 = fabs(Y_tempV1) - WLambdaV1;

      Y_tempV1 = signbit(-absxV1) * copysign(absxV1, Y_tempV1);

      Y[index] = Y_tempV1;
      U[index] = U_tempV1 + X_tempV1 - Y_tempV1;
    }
  }
}

__global__ void cuda_Shrink_CalU_vec4_Vector(float *Y, float *U, float *X,
                                             float lambda, float *L1Weight,
                                             int nRows, int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y, index;

  int Part_nRows = nRows / 4;

  float WLambda;
  float4 absxV4, X_temp, U_temp, Y_temp;

  if ((Tidx < nCols) & (Tidy < Part_nRows)) {

    for (int k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + Part_nRows * k) * nCols;

      U_temp = reinterpret_cast<float4 *>(U)[index];
      X_temp = reinterpret_cast<float4 *>(X)[index];

      Y_temp = operator_vec4(X_temp, U_temp, (nRows * nCols));

      WLambda = lambda * L1Weight[k];

      absxV4 = fabs_minus_b_vec4(Y_temp, WLambda);

      Y_temp.x = signbit(-absxV4.x) * copysign(absxV4.x, Y_temp.x);
      Y_temp.y = signbit(-absxV4.y) * copysign(absxV4.y, Y_temp.y);
      Y_temp.z = signbit(-absxV4.z) * copysign(absxV4.z, Y_temp.z);
      Y_temp.w = signbit(-absxV4.w) * copysign(absxV4.w, Y_temp.w);

      U_temp = operator2_vec4(X_temp, Y_temp, U_temp, (nRows * nCols));

      reinterpret_cast<float4 *>(Y)[index] = Y_temp;
      reinterpret_cast<float4 *>(U)[index] = U_temp;
    }
  }
}

__global__ void cuda_Shrink_CalU_Vector(float *Y, float *U, float *X,
                                        float lambda, float *L1Weight,
                                        int nRows, int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y, index;

  float WLambda;
  float absxV1, X_temp, U_temp, Y_temp;

  if ((Tidx < nCols) & (Tidy < nRows)) {

    for (int k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;

      X_temp = (X[index] / (nRows * nCols));
      U_temp = U[index];

      WLambda = lambda * L1Weight[k];

      Y_temp = X_temp + U_temp;
      absxV1 = fabs(Y_temp) - WLambda;

      Y_temp = signbit(-absxV1) * copysign(absxV1, Y_temp);

      Y[index] = Y_temp;
      U[index] = U_temp + X_temp - Y_temp;
    }
  }
}

__global__ void cuda_Shrink_CalU_vec4(float *Y, float *U, float *X,
                                      float lambda, int nRows, int nCols,
                                      int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float absxV1, X_tempV1, U_tempV1, Y_tempV1;
  float4 absxV4, X_temp, U_temp, Y_temp;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;

      U_temp = reinterpret_cast<float4 *>(U)[index];
      X_temp = reinterpret_cast<float4 *>(X)[index];

      Y_temp = operator_vec4(X_temp, U_temp, (nRows * nCols));

      absxV4 = fabs_minus_b_vec4(Y_temp, lambda);

      Y_temp.x = signbit(-absxV4.x) * copysign(absxV4.x, Y_temp.x);
      Y_temp.y = signbit(-absxV4.y) * copysign(absxV4.y, Y_temp.y);
      Y_temp.z = signbit(-absxV4.z) * copysign(absxV4.z, Y_temp.z);
      Y_temp.w = signbit(-absxV4.w) * copysign(absxV4.w, Y_temp.w);

      U_temp = operator2_vec4(X_temp, Y_temp, U_temp, (nRows * nCols));

      reinterpret_cast<float4 *>(Y)[index] = Y_temp;
      reinterpret_cast<float4 *>(U)[index] = U_temp;
    }

    // Process remaining elements
    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;

      X_tempV1 = (X[index] / (nRows * nCols));
      U_tempV1 = U[index];

      Y_tempV1 = X_tempV1 + U_tempV1;
      absxV1 = fabs(Y_tempV1) - lambda;

      Y_tempV1 = signbit(-absxV1) * copysign(absxV1, Y_tempV1);

      Y[index] = Y_tempV1;
      U[index] = U_tempV1 + X_tempV1 - Y_tempV1;
    }
  }
}

/******************************/
/****   Shrinkage Kernel   ****/
/******************************/

__global__ void cuda_Shrink_vec4(float *Y, float *X, float *U, float lambda,
                                 int nRows, int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float absxV1, tempV1;
  float4 absxV4, X_temp, U_temp;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;

      U_temp = reinterpret_cast<float4 *>(U)[index];
      X_temp = reinterpret_cast<float4 *>(X)[index];
      X_temp = operator_vec4(X_temp, U_temp, (nRows * nCols));

      absxV4 = fabs_minus_b_vec4(X_temp, lambda);

      X_temp.x = signbit(-absxV4.x) * copysign(absxV4.x, X_temp.x);
      X_temp.y = signbit(-absxV4.y) * copysign(absxV4.y, X_temp.y);
      X_temp.z = signbit(-absxV4.z) * copysign(absxV4.z, X_temp.z);
      X_temp.w = signbit(-absxV4.w) * copysign(absxV4.w, X_temp.w);

      reinterpret_cast<float4 *>(Y)[index] = X_temp;
    }

    // Process remaining elements
    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;

      tempV1 = (X[index] / (nRows * nCols)) + U[index];
      absxV1 = fabs(tempV1) - lambda;

      tempV1 = signbit(-absxV1) * copysign(absxV1, tempV1);

      Y[index] = tempV1;
    }
  }
}

/***************************/
/****   U Calculation   ****/
/***************************/

__global__ void cuda_CalU_vec4(float *U, float *X, float *Y, int nRows,
                               int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float4 X_temp, Y_temp, U_temp;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;

      X_temp = reinterpret_cast<float4 *>(X)[index];
      Y_temp = reinterpret_cast<float4 *>(Y)[index];
      U_temp = reinterpret_cast<float4 *>(U)[index];

      U_temp = operator2_vec4(X_temp, Y_temp, U_temp, (nRows * nCols));

      reinterpret_cast<float4 *>(U)[index] = U_temp;
    }

    // Process remaining elements
    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;
      U[index] = U[index] + (X[index] / (nRows * nCols)) - Y[index];
    }
  }
}

/***********************************/
/****   Dsf and C Calculation   ****/
/***********************************/

__global__ void cuda_Cal_Dsf_C_vec4(float2 *Dsf, float2 *C, float2 *Sf,
                                    float2 *Df, float rho, int nRows, int nCols,
                                    int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  unsigned int half_nRows = nRows / 2;

  float4 a, ah, sum_Df, b;

  sum_Df = make_float4(0, 0, 0, 0);

  if ((Tidx < nCols) & (Tidy < half_nRows)) {
    index = Tidx + Tidy * nCols;
    b = reinterpret_cast<float4 *>(Sf)[index];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      ah = reinterpret_cast<float4 *>(Df)[index];
      a = cuConjf_vec4(ah);
      reinterpret_cast<float4 *>(Dsf)[index] = cuCmulf_vec4(b, a);

      sum_Df = cuCaddf_vec4(sum_Df, cuCmulf_vec4(a, ah));
    }

    sum_Df.x += rho;
    sum_Df.z += rho;

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      reinterpret_cast<float4 *>(C)[index] =
          cuCdivf_vec4(reinterpret_cast<float4 *>(Df)[index], sum_Df);
    }
  }
}

// This kernel is not working with vectorized data

__global__ void cuda_Cal_Dsf_C(float2 *Dsf, float2 *C, float2 *Sf, float2 *Df,
                               float rho, int nRows, int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  float2 a, ah, sum_Df, b;

  sum_Df = make_cuFloatComplex(0, 0);

  if ((Tidx < nCols) & (Tidy < nRows)) {
    index = Tidx + Tidy * nCols;
    b = Sf[index];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      ah = Df[index];
      a = cuConjf(ah);
      Dsf[index] = cuCmulf(b, a);

      sum_Df = cuCaddf(sum_Df, cuCmulf(a, ah));
    }

    sum_Df.x += rho;

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      C[index] = cuCdivf(Df[index], sum_Df);
    }
  }
}

/***************************/
/****   C Calculation   ****/
/***************************/

__global__ void cuda_Cal_C_vec4(float2 *C, float2 *Df, float rho, int nRows,
                                int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  unsigned int half_nRows = nRows / 2;

  float4 a, ah, sum_Df;

  sum_Df = make_float4(0, 0, 0, 0);

  if ((Tidx < nCols) & (Tidy < half_nRows)) {

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      ah = reinterpret_cast<float4 *>(Df)[index];
      a = cuConjf_vec4(ah);
      sum_Df = cuCaddf_vec4(sum_Df, cuCmulf_vec4(a, ah));
    }

    sum_Df.x += rho;
    sum_Df.z += rho;

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      reinterpret_cast<float4 *>(C)[index] =
          cuCdivf_vec4(reinterpret_cast<float4 *>(Df)[index], sum_Df);
    }
  }
}

// This kernel does not work with vectorized data

__global__ void cuda_Cal_C(float2 *C, float2 *Df, float rho, int nRows,
                           int nCols, int nFilts) {

  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  float2 a, ah, sum_Df;

  sum_Df = make_cuFloatComplex(0, 0);

  if ((Tidx < nCols) & (Tidy < nRows)) {

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      ah = Df[index];
      a = cuConjf(ah);
      sum_Df = cuCaddf(sum_Df, cuCmulf(a, ah));
    }

    sum_Df.x += rho;

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      C[index] = cuCdivf(Df[index], sum_Df);
    }
  }
}

/*******************************/
/****   Padded Dictionary   ****/
/*******************************/

// Sets padded Dictionary in order to get the same size of nCols and nRows
// input image (required size for FFT point wise multiplication)

__global__ void cuda_Pad_Dict(float *PadD, float *D, int nCols_D, int nRows_D,
                              int nFilts, int nRows, int nCols) {
  unsigned int Tidx_D = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy_D = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int Tidy_PadD = threadIdx.y + blockIdx.y * blockDim.y;

  int Dim_D = nRows_D * nFilts;

  if ((Tidx_D < nCols_D) & (Tidy_D < nRows_D)) {

    for (; Tidy_D < Dim_D; Tidy_D += nRows_D, Tidy_PadD += nRows)
      PadD[Tidx_D + Tidy_PadD * nCols] = D[Tidx_D + Tidy_D * nCols_D];
  }
}

/******************************/
/****   Y-U  Calculation   ****/
/******************************/

__global__ void cuda_CalYU_vec4(float *YU, float *Y, float *U, int nRows,
                                int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float4 Y_temp, U_temp;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;

      Y_temp = reinterpret_cast<float4 *>(Y)[index];
      U_temp = reinterpret_cast<float4 *>(U)[index];

      Y_temp = sub_vec4(Y_temp, U_temp);

      reinterpret_cast<float4 *>(YU)[index] = Y_temp;
    }

    // Process remaining elements
    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;
      YU[index] = Y[index] - U[index];
    }
  }
}

/*****************************/
/****   Over-relaxation   ****/
/*****************************/

__global__ void cuda_OverRelax_vec4(float *Xr, float *X, float *Y,
                                    float RelaxParam, int nRows, int nCols,
                                    int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float4 X_temp, Y_temp;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;

      X_temp = reinterpret_cast<float4 *>(X)[index];
      Y_temp = reinterpret_cast<float4 *>(Y)[index];
      Y_temp = cuCmulf_by_const_vec4(Y_temp, nRows * nCols);

      reinterpret_cast<float4 *>(Xr)[index] =
          OverRelax_vec4(X_temp, Y_temp, RelaxParam);
    }

    // Process remaining elements
    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;
      Xr[index] =
          RelaxParam * X[index] + (1 - RelaxParam) * Y[index] * (nRows * nCols);
    }
  }
}

/*********************************************/
/****   Residuals and norms Calculation   ****/
/*********************************************/

__global__ void cuda_Cal_residuals_norms_vec4(float *s, float *r, float *nX,
                                              float *nY, float *nU, float *X,
                                              float *Y, float *U, float *Yprv,
                                              float rho, int nRows, int nCols,
                                              int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float sumXY = 0.0, sumYY = 0.0, sumX = 0.0, sumY = 0.0, sumU = 0.0;
  float4 X_temp, Y_temp, U_temp, Yprv_temp;

  rho = rho * rho;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;
      X_temp = reinterpret_cast<float4 *>(X)[index];
      X_temp = cuCdivf_by_const_vec4(X_temp, nRows * nCols);

      Y_temp = reinterpret_cast<float4 *>(Y)[index];
      U_temp = reinterpret_cast<float4 *>(U)[index];
      Yprv_temp = reinterpret_cast<float4 *>(Yprv)[index];

      sumX += ((X_temp.x * X_temp.x + X_temp.y * X_temp.y) +
               (X_temp.z * X_temp.z + X_temp.w * X_temp.w));
      sumY += ((Y_temp.x * Y_temp.x + Y_temp.y * Y_temp.y) +
               (Y_temp.z * Y_temp.z + Y_temp.w * Y_temp.w));
      sumU += ((U_temp.x * U_temp.x + U_temp.y * U_temp.y) +
               (U_temp.z * U_temp.z + U_temp.w * U_temp.w));

      X_temp = sub_vec4(X_temp, Y_temp);
      Y_temp = sub_vec4(Yprv_temp, Y_temp);

      sumXY += ((X_temp.x * X_temp.x + X_temp.y * X_temp.y) +
                (X_temp.z * X_temp.z + X_temp.w * X_temp.w));
      sumYY += (((rho * Y_temp.x * Y_temp.x) + (rho * Y_temp.y * Y_temp.y)) +
                ((rho * Y_temp.z * Y_temp.z) + (rho * Y_temp.w * Y_temp.w)));
    }

    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;
      X_temp.x = X[index] / (nRows * nCols);
      Y_temp.x = Y[index];
      U_temp.x = U[index];
      Yprv_temp.x = Yprv[index];

      sumX += (X_temp.x * X_temp.x);
      sumY += (Y_temp.x * Y_temp.x);
      sumU += (U_temp.x * U_temp.x);

      X_temp.x = X_temp.x - Y_temp.x;
      Y_temp.x = Yprv_temp.x - Y_temp.x;

      sumXY += (X_temp.x * X_temp.x);
      sumYY += (rho * Y_temp.x * Y_temp.x);
    }
  }

  sumX = blockReduceSum(sumX);
  sumY = blockReduceSum(sumY);
  sumU = blockReduceSum(sumU);

  sumXY = blockReduceSum(sumXY);
  sumYY = blockReduceSum(sumYY);

  if (threadIdx.x == 0) {
    // sumX, sumY, sumU, sumXY and sumYY are multiplied by 1e+6 to
    // avoid calculation error
    atomicAdd(nX, 1e+6 * sumX);
    atomicAdd(nY, 1e+6 * sumY);
    atomicAdd(nU, 1e+6 * sumU);

    atomicAdd(r, 1e+6 * sumXY);
    atomicAdd(s, 1e+6 * sumYY);
  }
}

/*******************************/
/****   Norm Calculations   ****/
/*******************************/

__global__ void cuda_norm_vec4(float *nX, float *nY, float *nU, float *X,
                               float *Y, float *U, int nRows, int nCols,
                               int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float sumX = 0.0, sumY = 0.0, sumU = 0.0;
  float4 X_temp, Y_temp, U_temp;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;
      X_temp = reinterpret_cast<float4 *>(X)[index];
      X_temp = cuCdivf_by_const_vec4(X_temp, nRows * nCols);

      Y_temp = reinterpret_cast<float4 *>(Y)[index];
      U_temp = reinterpret_cast<float4 *>(U)[index];

      sumX += ((X_temp.x * X_temp.x + X_temp.y * X_temp.y) +
               (X_temp.z * X_temp.z + X_temp.w * X_temp.w));
      sumY += ((Y_temp.x * Y_temp.x + Y_temp.y * Y_temp.y) +
               (Y_temp.z * Y_temp.z + Y_temp.w * Y_temp.w));
      sumU += ((U_temp.x * U_temp.x + U_temp.y * U_temp.y) +
               (U_temp.z * U_temp.z + U_temp.w * U_temp.w));
    }

    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;
      X_temp.x = X[index];
      Y_temp.x = Y[index];
      U_temp.x = U[index];

      sumX += (X_temp.x * X_temp.x);
      sumY += (Y_temp.x * Y_temp.x);
      sumX += (X_temp.x * X_temp.x);
    }
  }

  sumX = blockReduceSum(sumX);
  sumY = blockReduceSum(sumY);
  sumU = blockReduceSum(sumU);

  if (threadIdx.x == 0) {
    // sumX, sumY and sumU are multiplied by 1e+6 to avoid calculation error
    atomicAdd(nX, 1e+6 * sumX);
    atomicAdd(nX, 1e+6 * sumY);
    atomicAdd(nU, 1e+6 * sumU);
  }
}

/**********************************/
/****   Residual Calculation   ****/
/**********************************/

__global__ void cuda_Cal_residuals_vec4(float *s, float *r, float *X, float *Y,
                                        float *Yprv, float rho, int nRows,
                                        int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float sumXY = 0.0, sumYY = 0.0;

  float4 X_temp, Y_temp, Yprv_temp;

  rho = rho * rho;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;
      X_temp = reinterpret_cast<float4 *>(X)[index];
      Y_temp = reinterpret_cast<float4 *>(Y)[index];
      Yprv_temp = reinterpret_cast<float4 *>(Yprv)[index];

      X_temp = cuCdivf_by_const_vec4(X_temp, nRows * nCols);
      X_temp = sub_vec4(X_temp, Y_temp);
      Y_temp = sub_vec4(Yprv_temp, Y_temp);

      sumXY += ((X_temp.x * X_temp.x + X_temp.y * X_temp.y) +
                (X_temp.z * X_temp.z + X_temp.w * X_temp.w));
      sumYY += (((rho * Y_temp.x * Y_temp.x) + (rho * Y_temp.y * Y_temp.y)) +
                ((rho * Y_temp.z * Y_temp.z) + (rho * Y_temp.w * Y_temp.w)));
    }

    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;
      X_temp.x = X[index] / (nRows * nCols) - Y[index];
      Y_temp.x = Yprv[index] - Y[index];
      sumXY += (X_temp.x * X_temp.x);
      sumYY += (rho * Y_temp.x * Y_temp.x);
    }
  }

  sumXY = blockReduceSum(sumXY);
  sumYY = blockReduceSum(sumYY);

  if (threadIdx.x == 0) {
    // sumXY and sumYY are multiplied by 1e+6 to avoid calculation error
    atomicAdd(r, 1e+6 * sumXY);
    atomicAdd(s, 1e+6 * sumYY);
  }
}

/***************************************/
/****   Fidelity Term Calculation   ****/
/***************************************/

__global__ void cuda_Fidelity_Term_vec4(float *Jdf, float2 *Df, float2 *Xf,
                                        float2 *Sf, int nRows, int nCols,
                                        int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  unsigned int half_nRows = nRows / 2;

  float4 Df_temp, Xf_temp, sum_Df;
  float sum = 0.0;

  sum_Df = make_float4(0, 0, 0, 0);

  if ((Tidx < nCols) & (Tidy < half_nRows)) {

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      Df_temp = reinterpret_cast<float4 *>(Df)[index];
      Xf_temp = reinterpret_cast<float4 *>(Xf)[index];

      sum_Df = cuCaddf_vec4(sum_Df, cuCmulf_vec4(Df_temp, Xf_temp));
    }

    index = Tidx + Tidy * nCols;
    sum_Df = cuCsubf_vec4(sum_Df, reinterpret_cast<float4 *>(Sf)[index]);

    sum += sum_sqrt_cuCabsf_vec4(sum_Df);
  }

  sum = blockReduceSum(sum);

  if (threadIdx.x == 0)
    atomicAdd(Jdf, sum);
}

// This kernel does not work with vectorized data

__global__ void cuda_Fidelity_Term(float *Jdf, float2 *Df, float2 *Xf,
                                   float2 *Sf, int nRows, int nCols,
                                   int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  float2 sum_Df;
  float sum = 0.0, aux;

  sum_Df = make_cuFloatComplex(0, 0);

  if ((Tidx < nCols) & (Tidy < nRows)) {

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      sum_Df = cuCaddf(sum_Df, cuCmulf(Df[index], Xf[index]));
    }

    index = Tidx + Tidy * nCols;
    sum_Df = cuCsubf(sum_Df, Sf[index]);

    aux = cuCabsf(sum_Df);
    sum += (aux * aux);
  }

  sum = blockReduceSum(sum);

  if (threadIdx.x == 0)
    atomicAdd(Jdf, sum);
}

/*********************************/
/****   L1 Term Calculation   ****/
/*********************************/

__global__ void cuda_L1_Term_vec4(float *d_JL1, float *X, float L1Weight,
                                  int factor, int nRows, int nCols,
                                  int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float sum = 0.0;
  float4 X_temp;

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;
      X_temp = reinterpret_cast<float4 *>(X)[index];

      sum += (abs(X_temp.x / factor) + abs(X_temp.y / factor)) +
             (abs(X_temp.z / factor) + abs(X_temp.w / factor));
    }

    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;
      sum += abs(X[index] / factor);
    }
  }

  sum = blockReduceSum(sum);
  sum *= abs(L1Weight);

  if (threadIdx.x == 0)
    atomicAdd(d_JL1, sum);
}

__global__ void cuda_L1_Term_vec4_Scalar_Array(float *d_JL1, float *X,
                                               float *L1Weight, int factor,
                                               int nRows, int nCols, int nFilts,
                                               int nL1Weight) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int stride = blockDim.y * gridDim.y, index;

  int Part_Dim = (nRows * nFilts) / 4;
  int Full_Dim = nRows * nFilts;

  float WLambdaV1, sum = 0.0;
  float4 X_temp, WLambda;

  if (nL1Weight == 1) {
    WLambdaV1 = L1Weight[0];
    WLambda = make_float4(WLambdaV1, WLambdaV1, WLambdaV1, WLambdaV1);
  }

  if (Tidx < nCols) {

    for (int i = Tidy; i < Part_Dim; i += stride) {
      index = Tidx + i * nCols;
      X_temp = reinterpret_cast<float4 *>(X)[index];

      if (nL1Weight > 1)
        WLambda = reinterpret_cast<float4 *>(L1Weight)[index];

      sum += (abs(WLambda.x * X_temp.x / factor) +
              abs(WLambda.y * X_temp.y / factor)) +
             (abs(WLambda.z * X_temp.z / factor) +
              abs(WLambda.w * X_temp.w / factor));
    }

    for (int i = Tidy + 4 * Part_Dim; i < Full_Dim; i += stride) {
      index = Tidx + i * nCols;

      if (nL1Weight > 1)
        WLambdaV1 = L1Weight[index];

      sum += abs(WLambdaV1 * X[index] / factor);
    }
  }

  sum = blockReduceSum(sum);

  if (threadIdx.x == 0)
    atomicAdd(d_JL1, sum);
}

__global__ void cuda_L1_Term_vec4_Vector(float *d_JL1, float *X,
                                         float *L1Weight, int factor, int nRows,
                                         int nCols, int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y, index;

  int Part_nRows = nRows / 4;

  float WLambda, sum = 0.0;
  float4 X_temp;

  if ((Tidx < nCols) & (Tidy < Part_nRows)) {

    for (int k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + Part_nRows * k) * nCols;

      X_temp = reinterpret_cast<float4 *>(X)[index];
      WLambda = L1Weight[k];

      sum +=
          (abs(WLambda * X_temp.x / factor) +
           abs(WLambda * X_temp.y / factor)) +
          (abs(WLambda * X_temp.z / factor) + abs(WLambda * X_temp.w / factor));
    }
  }

  sum = blockReduceSum(sum);

  if (threadIdx.x == 0)
    atomicAdd(d_JL1, sum);
}

__global__ void cuda_L1_Term_Vector(float *d_JL1, float *X, float *L1Weight,
                                    int factor, int nRows, int nCols,
                                    int nFilts) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y, index;

  float X_temp, WLambda, sum = 0.0;

  if ((Tidx < nCols) & (Tidy < nRows)) {

    for (int k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;

      X_temp = abs(X[index]) / factor;
      WLambda = L1Weight[k];

      sum += (WLambda * X_temp);
    }
  }

  sum = blockReduceSum(sum);

  if (threadIdx.x == 0)
    atomicAdd(d_JL1, sum);
}

__global__ void cuda_Cal_X_minus_U_W(float *Y, float *U, float *X, int *Weight,
                                     int nRows, int nCols) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y, index;

  float X_temp, U_temp, Y_temp;

  if ((Tidx < nCols) & (Tidy < nRows)) {
    index = Tidx + Tidy * nCols;

    X_temp = (X[index] / (nRows * nCols));
    U_temp = U[index];

    Y_temp = (1 - Weight[index]) * (X_temp + U_temp);

    Y[index] = Y_temp;
    U[index] = U_temp + X_temp - Y_temp;
  }
}

/******************************************/
/******************************************/
/********  Kernels for cbpdn_grd  *********/
/******************************************/
/******************************************/

/****************************/
/****   GfW Calculation  ****/
/****************************/

__global__ void cuda_Cal_Gfw(float *GfW, float2 *Grf, float2 *Gcf, int nRows,
                             int nCols) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index;

  float GfW_temp;
  float2 Grf_temp, Gcf_temp;

  if ((Tidx < nCols) & (Tidy < nRows)) {

    index = Tidx + Tidy * nCols;

    Grf_temp = Grf[index];
    Gcf_temp = Gcf[index];
    GfW_temp = Grf_temp.x * Grf_temp.x + Grf_temp.y * Grf_temp.y +
               Gcf_temp.x * Gcf_temp.x + Gcf_temp.y * Gcf_temp.y;

    GfW[index] = GfW_temp;
  }
}

/******************************/
/****   Sherman Morrison   ****/
/******************************/

// This method provides the solution for x (Output) to
//     (a a^H + rho I) x = b
// Where b = rho*fft2(Y-U) and inner products are taken along the 3rd dimesion.
// Only works for even image size and is faster than ypncuda_sherman kernel.

__global__ void cuda_solvedbd_sm_vec4(float2 *Out, float2 *ah, float2 *Dsf,
                                      float *GfW, float *GrdWeight, float rho,
                                      float mu, float2 *YUf, float2 *c,
                                      int nRows, int nCols, int nFilts,
                                      int nWeights) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k, half_nRows = nRows / 2;

  float4 a, cb, cba, x, b, YUf_temp;
  float2 GfW_temp, aux;
  float Weight;

  cb = make_float4(0, 0, 0, 0);

  if ((Tidx < nCols) & (Tidy < half_nRows)) {

    index = Tidx + Tidy * nCols;
    GfW_temp = reinterpret_cast<float2 *>(GfW)[index];
    GfW_temp.x *= mu;
    GfW_temp.y *= mu;
    Weight = GrdWeight[0];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;

      YUf_temp = reinterpret_cast<float4 *>(YUf)[index];
      YUf_temp = cuCmulf_by_const_vec4(YUf_temp, rho);

      b = cuCaddf_vec4(reinterpret_cast<float4 *>(Dsf)[index], YUf_temp);
      cb = cuCaddf_vec4(cb,
                        cuCmulf_vec4(b, reinterpret_cast<float4 *>(c)[index]));
    }

    a = cuConjf_vec4(reinterpret_cast<float4 *>(ah)[index]);
    cba = cuCmulf_vec4(cb, a);
    x = cuCsubf_vec4(b, cba);

    if (nWeights > 1)
      Weight = GrdWeight[nFilts - 1];
    aux.x = Weight * GfW_temp.x + rho;
    aux.y = Weight * GfW_temp.y + rho;

    x = make_float4((x.x / aux.x), (x.y / aux.x), (x.z / aux.y), (x.w / aux.y));
    reinterpret_cast<float4 *>(Out)[index] = x;

    for (k = 0; k < nFilts - 1; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      a = cuConjf_vec4(reinterpret_cast<float4 *>(ah)[index]);
      cba = cuCmulf_vec4(cb, a);

      YUf_temp = reinterpret_cast<float4 *>(YUf)[index];
      YUf_temp = cuCmulf_by_const_vec4(YUf_temp, rho);

      b = cuCaddf_vec4(reinterpret_cast<float4 *>(Dsf)[index], YUf_temp);
      x = cuCsubf_vec4(b, cba);

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux.x = Weight * GfW_temp.x + rho;
      aux.y = Weight * GfW_temp.y + rho;

      x = make_float4((x.x / aux.x), (x.y / aux.x), (x.z / aux.y),
                      (x.w / aux.y));

      reinterpret_cast<float4 *>(Out)[index] = x;
    }
  }
}

// Works for every image size
// This kernel does not work with vectorized data

__global__ void cuda_solvedbd_sm(float2 *Out, float2 *ah, float2 *Dsf,
                                 float *GfW, float *GrdWeight, float rho,
                                 float mu, float2 *YUf, float2 *c, int nRows,
                                 int nCols, int nFilts, int nWeights) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  float2 a, cb, cba, x, b, YUf_temp;
  float aux, Weight, GfW_temp;

  cb = make_cuFloatComplex(0, 0);

  if ((Tidx < nCols) & (Tidy < nRows)) {

    index = Tidx + Tidy * nCols;
    GfW_temp = mu * GfW[index];
    Weight = GrdWeight[0];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;

      YUf_temp = YUf[index];
      YUf_temp.x *= rho;
      YUf_temp.y *= rho;

      b = cuCaddf(Dsf[index], YUf_temp);

      cb = cuCaddf(cb, cuCmulf(b, c[index]));
    }

    a = cuConjf(ah[index]);
    cba = cuCmulf(cb, a);

    x = cuCsubf(b, cba);

    if (nWeights > 1)
      Weight = GrdWeight[nFilts - 1];
    aux = Weight * GfW_temp + rho;

    x.x /= aux;
    x.y /= aux;
    Out[index] = x;

    for (k = 0; k < nFilts - 1; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      a = cuConjf(ah[index]);
      cba = cuCmulf(cb, a);

      YUf_temp = YUf[index];
      YUf_temp.x *= rho;
      YUf_temp.y *= rho;

      b = cuCaddf(Dsf[index], YUf_temp);

      x = cuCsubf(b, cba);

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux = Weight * GfW_temp + rho;

      x.x /= aux;
      x.y /= aux;
      Out[index] = x;
    }
  }
}

/***************************************/
/****   Fidelity Term Calculation   ****/
/***************************************/

__global__ void cuda_Fidelity_Gr_Term_vec4(float *Jdf, float *Jgr, float2 *Df,
                                           float2 *Xf, float2 *Sf, float *GfW,
                                           float *GrdWeight, int nRows,
                                           int nCols, int nFilts,
                                           int nWeights) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  unsigned int half_nRows = nRows / 2;

  float4 Df_temp, Xf_temp, sum_Df;
  float sum = 0.0, sum_Gr = 0.0, Weight;
  float2 GfW_temp, aux;

  sum_Df = make_float4(0, 0, 0, 0);

  if ((Tidx < nCols) & (Tidy < half_nRows)) {
    index = Tidx + Tidy * nCols;
    GfW_temp = reinterpret_cast<float2 *>(GfW)[index];
    Weight = GrdWeight[0];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      Df_temp = reinterpret_cast<float4 *>(Df)[index];
      Xf_temp = reinterpret_cast<float4 *>(Xf)[index];

      if (nWeights > 1)
        Weight = GrdWeight[k];

      aux.x =
          GfW_temp.x * Weight * (Xf_temp.x * Xf_temp.x + Xf_temp.y * Xf_temp.y);
      aux.y =
          GfW_temp.y * Weight * (Xf_temp.z * Xf_temp.z + Xf_temp.w * Xf_temp.w);

      sum_Gr += (aux.x + aux.y);
      sum_Df = cuCaddf_vec4(sum_Df, cuCmulf_vec4(Df_temp, Xf_temp));
    }

    index = Tidx + Tidy * nCols;
    sum_Df = cuCsubf_vec4(sum_Df, reinterpret_cast<float4 *>(Sf)[index]);

    sum += sum_sqrt_cuCabsf_vec4(sum_Df);
  }

  sum = blockReduceSum(sum);
  sum_Gr = blockReduceSum(sum_Gr);

  if (threadIdx.x == 0) {
    atomicAdd(Jdf, sum);
    atomicAdd(Jgr, sum_Gr);
  }
}

// This kernel does not work with vectorized data

__global__ void cuda_Fidelity_Gr_Term(float *Jdf, float *Jgr, float2 *Df,
                                      float2 *Xf, float2 *Sf, float *GfW,
                                      float *GrdWeight, int nRows, int nCols,
                                      int nFilts, int nWeights) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  float2 sum_Df, Xf_temp;
  float sum = 0.0, sum_Gr = 0.0, aux, Weight, GfW_temp;

  sum_Df = make_cuFloatComplex(0, 0);

  if ((Tidx < nCols) & (Tidy < nRows)) {

    index = Tidx + Tidy * nCols;
    GfW_temp = GfW[index];
    Weight = GrdWeight[0];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      Xf_temp = Xf[index];

      if (nWeights > 1)
        Weight = GrdWeight[k];

      aux = GfW_temp * Weight * (Xf_temp.x * Xf_temp.x + Xf_temp.y * Xf_temp.y);

      sum_Gr += aux;
      sum_Df = cuCaddf(sum_Df, cuCmulf(Df[index], Xf_temp));
    }

    index = Tidx + Tidy * nCols;
    sum_Df = cuCsubf(sum_Df, Sf[index]);

    aux = cuCabsf(sum_Df);
    sum += (aux * aux);
  }

  sum = blockReduceSum(sum);
  sum_Gr = blockReduceSum(sum_Gr);

  if (threadIdx.x == 0) {
    atomicAdd(Jdf, sum);
    atomicAdd(Jgr, sum_Gr);
  }
}

/***********************************/
/****   Dsf and C Calculation   ****/
/***********************************/

__global__ void cuda_Cal_Dsf_grd_C_vec4(float2 *Dsf, float2 *C, float2 *Sf,
                                        float2 *Df, float *GfW,
                                        float *GrdWeight, float rho, float mu,
                                        int nRows, int nCols, int nFilts,
                                        int nWeights) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  unsigned int half_nRows = nRows / 2;

  float4 a, ah, b;
  float2 aux, sum_Df, GfW_temp;
  float Weight;

  sum_Df = make_float2(0, 0);

  if ((Tidx < nCols) & (Tidy < half_nRows)) {
    index = Tidx + Tidy * nCols;
    b = reinterpret_cast<float4 *>(Sf)[index];
    GfW_temp = reinterpret_cast<float2 *>(GfW)[index];
    GfW_temp.x *= mu;
    GfW_temp.y *= mu;
    Weight = GrdWeight[0];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      ah = reinterpret_cast<float4 *>(Df)[index];
      a = cuConjf_vec4(ah);
      reinterpret_cast<float4 *>(Dsf)[index] = cuCmulf_vec4(b, a);

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux.x = Weight * GfW_temp.x + rho;
      aux.y = Weight * GfW_temp.y + rho;

      sum_Df =
          cuCaddf(sum_Df, make_float2((ah.x * ah.x + ah.y * ah.y) / aux.x,
                                      (ah.z * ah.z + ah.w * ah.w) / aux.y));
    }

    sum_Df.x += 1;
    sum_Df.y += 1;

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      ah = reinterpret_cast<float4 *>(Df)[index];

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux.x = sum_Df.x * (Weight * GfW_temp.x + rho);
      aux.y = sum_Df.y * (Weight * GfW_temp.y + rho);

      reinterpret_cast<float4 *>(C)[index] =
          make_float4(ah.x / aux.x, ah.y / aux.x, ah.z / aux.y, ah.w / aux.y);
    }
  }
}

// This kernel does not work with vectorized data

__global__ void cuda_Cal_Dsf_grd_C(float2 *Dsf, float2 *C, float2 *Sf,
                                   float2 *Df, float *GfW, float *GrdWeight,
                                   float rho, float mu, int nRows, int nCols,
                                   int nFilts, int nWeights) {

  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  float2 a, ah, b;
  float Weight, GfW_temp, aux, sum_Df = 0;

  if ((Tidx < nCols) & (Tidy < nRows)) {
    index = Tidx + Tidy * nCols;
    b = Sf[index];
    GfW_temp = mu * GfW[index];
    Weight = GrdWeight[0];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      ah = Df[index];
      a = cuConjf(ah);
      Dsf[index] = cuCmulf(b, a);

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux = Weight * GfW_temp + rho;

      sum_Df += ((ah.x * ah.x + ah.y * ah.y) / aux);
    }

    sum_Df += 1;

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      ah = Df[index];

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux = sum_Df * (Weight * GfW_temp + rho);

      C[index] = make_cuFloatComplex(ah.x / aux, ah.y / aux);
    }
  }
}

/***************************/
/****   C Calculation   ****/
/***************************/

__global__ void cuda_Cal_grd_C_vec4(float2 *C, float2 *Df, float *GfW,
                                    float *GrdWeight, float rho, float mu,
                                    int nRows, int nCols, int nFilts,
                                    int nWeights) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  unsigned int half_nRows = nRows / 2;

  float4 ah;
  float2 aux, sum_Df, GfW_temp;
  float Weight;

  sum_Df = make_float2(0, 0);

  if ((Tidx < nCols) & (Tidy < half_nRows)) {

    index = Tidx + Tidy * nCols;
    GfW_temp = reinterpret_cast<float2 *>(GfW)[index];
    GfW_temp.x *= mu;
    GfW_temp.y *= mu;
    Weight = GrdWeight[0];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      ah = reinterpret_cast<float4 *>(Df)[index];

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux.x = Weight * GfW_temp.x + rho;
      aux.y = Weight * GfW_temp.y + rho;

      sum_Df =
          cuCaddf(sum_Df, make_float2((ah.x * ah.x + ah.y * ah.y) / aux.x,
                                      (ah.z * ah.z + ah.w * ah.w) / aux.y));
    }

    sum_Df.x += 1;
    sum_Df.y += 1;

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + half_nRows * k) * nCols;
      ah = reinterpret_cast<float4 *>(Df)[index];

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux.x = sum_Df.x * (Weight * GfW_temp.x + rho);
      aux.y = sum_Df.y * (Weight * GfW_temp.y + rho);

      reinterpret_cast<float4 *>(C)[index] =
          make_float4(ah.x / aux.x, ah.y / aux.x, ah.z / aux.y, ah.w / aux.y);
    }
  }
}

// This kernel does not work with vectorized data

__global__ void cuda_Cal_grd_C(float2 *C, float2 *Df, float *GfW,
                               float *GrdWeight, float rho, float mu, int nRows,
                               int nCols, int nFilts, int nWeights) {
  unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int index, k;

  float Weight, GfW_temp, aux, sum_Df = 0;
  float2 ah;

  if ((Tidx < nCols) & (Tidy < nRows)) {

    index = Tidx + Tidy * nCols;
    GfW_temp = mu * GfW[index];
    Weight = GrdWeight[0];

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      ah = Df[index];

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux = Weight * GfW_temp + rho;

      sum_Df += ((ah.x * ah.x + ah.y * ah.y) / aux);
    }

    sum_Df += 1;

    for (k = 0; k < nFilts; k += 1) {
      index = Tidx + (Tidy + nRows * k) * nCols;
      ah = Df[index];

      if (nWeights > 1)
        Weight = GrdWeight[k];
      aux = sum_Df * (Weight * GfW_temp + rho);

      C[index] = make_cuFloatComplex(ah.x / aux, ah.y / aux);
    }
  }
}

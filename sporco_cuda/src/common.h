//  Author: Gustavo Silva <gustavo.silva@pucp.edu.pe>

#include <stdio.h>

/* *********************************** */
/* *********************************** */
/*          CUDA error checking        */
/* *********************************** */
/* *********************************** */

#define checkCudaErrors(val)                                                   \
  __checkCudaErrors__((val), #val, __FILE__, __LINE__)

template <typename T>
inline void __checkCudaErrors__(T code, const char *func, const char *file,
                                int line) {
  if (code) {
    fprintf(stderr, " \n CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            (unsigned int)code, func);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

/* *********************************** */
/* *********************************** */
/*     Auxiliary Device Functions      */
/* *********************************** */
/* *********************************** */

#define warpSize 32

// Calculates the conjugate of 2 complex number vectorized as float4
__device__ inline float4 cuConjf_vec4(float4 a) {
  return make_float4(a.x, -a.y, a.z, -a.w);
}

// Calculates the sum of 2 pairs of complex number vectorized as float4
__device__ inline float4 cuCaddf_vec4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// Calculates the subtraction of 2 pairs of complex number vectorized as float4
__device__ inline float4 cuCsubf_vec4(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

// Calculates the multiplication of 2 pairs of complex number vectorized
// as float4
__device__ inline float4 cuCmulf_vec4(float4 a, float4 b) {
  return make_float4(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x,
                     a.z * b.z - a.w * b.w, a.z * b.w + a.w * b.z);
}

// Calculates the division of 2 pairs of 2 complex number vectorized as float4
__device__ inline float4 cuCdivf_vec4(float4 a, float4 b) {
  float2 c;

  c.x = b.x * b.x + b.y * b.y;
  c.y = b.z * b.z + b.w * b.w;

  return make_float4(
      (a.x * b.x + a.y * b.y) / c.x, (a.y * b.x - a.x * b.y) / c.x,
      (a.z * b.z + a.w * b.w) / c.y, (a.w * b.z - a.z * b.w) / c.y);
}

// Calculates the multiplication by a constant of 2 complex number
// vectorized as float4
__device__ inline float4 cuCmulf_by_const_vec4(float4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

// Calculates the multiplication by a constant of 2 complex number
// vectorized as float4
__device__ inline float4 cuCdivf_by_const_vec4(float4 a, float b) {
  return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

// Calculates fourth times absolute value of data minus a constant
// through vectorized array (float4)
__device__ inline float4 fabs_minus_b_vec4(float4 a, float b) {
  return make_float4(fabs(a.x) - b, fabs(a.y) - b, fabs(a.z) - b,
                     fabs(a.w) - b);
}

// Calculates fourth times absolute value of data minus float4 constant
// through vectorized array (float4)
__device__ inline float4 fabs_minus_bf4_vec4(float4 a, float4 b) {
  return make_float4(fabs(a.x) - b.x, fabs(a.y) - b.y, fabs(a.z) - b.z,
                     fabs(a.w) - b.w);
}

// Calculates fourth times x/b + u, where b is a constant and the
// others variables are vectorized data (float4)
__device__ inline float4 operator_vec4(float4 x, float4 u, float b) {
  return make_float4((x.x / b) + u.x, (x.y / b) + u.y, (x.z / b) + u.z,
                     (x.w / b) + u.w);
}

// Calculates fourth times  u + x/b - y, where b is a constant and
// the others variables are vectorized data (float4)
__device__ inline float4 operator2_vec4(float4 x, float4 y, float4 u, float b) {
  return make_float4(u.x + (x.x / b) - y.x, u.y + (x.y / b) - y.y,
                     u.z + (x.z / b) - y.z, u.w + (x.w / b) - y.w);
}

// Calculates Four simultaneous subtractions as a vectorized data (float4)
__device__ inline float4 sub_vec4(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

// Calculates Four simultaneous OverRelaxation operations (float4)
__device__ inline float4 OverRelax_vec4(float4 a, float4 b, float R) {
  return make_float4(R * a.x + (1 - R) * b.x, R * a.y + (1 - R) * b.y,
                     R * a.z + (1 - R) * b.z, R * a.w + (1 - R) * b.w);
}

// Operator used during Fidelity term calculation
__device__ inline float sum_sqrt_cuCabsf_vec4(float4 a) {
  float a1, a2;
  a1 = sqrtf(a.x * a.x + a.y * a.y);
  a2 = sqrtf(a.z * a.z + a.w * a.w);

  return a1 * a1 + a2 * a2;
}

// Well-known CUDA reduction function
// Reduction done through warps
__forceinline__ __device__ float warpReduceSum(float val, int Size) {

  for (unsigned int offset = Size; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);

  return val;
}

// Reduction done through blocks
__inline__ __device__ float blockReduceSum(float val) {

  // Shared mem for 16 partial sums due to we used 512 threads per block
  static __shared__ float shared[16];
  int lane = (threadIdx.x + threadIdx.y * blockDim.x) % warpSize;
  int wid = (threadIdx.x + threadIdx.y * blockDim.x) / warpSize;

  val =
      warpReduceSum(val, warpSize / 2); // Each warp performs partial reduction

  if (lane == 0)
    shared[wid] = val; // Write reduced value to shared memory

  __syncthreads(); // Wait for all partial reductions

  // Read from shared memory only if that warp existed
  val = ((threadIdx.x + threadIdx.y * blockDim.x) <
         ((blockDim.x * blockDim.y) / warpSize))
            ? shared[lane]
            : 0;

  // Final reduce within first warp
  if (wid == 0)
    val = warpReduceSum(val, warpSize / 4);

  return val;
}

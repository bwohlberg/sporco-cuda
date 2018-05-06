__global__ void cuda_solvedbi_sm_vec4(float2 *Out, float2 *ah, float2 *Dsf,
                                      float rho, float2 *YUf, float2 *c,
                                      int nRows, int nCols, int nFilts);

__global__ void cuda_solvedbi_sm(float2 *Out, float2 *ah, float2 *Dsf,
                                 float rho, float2 *YUf, float2 *c, int nRows,
                                 int nCols, int nFilts);

__global__ void cuda_Shrink_CalU_vec4_Scalar(float *Y, float *U, float *X,
                                             float lambda, float *L1Weight,
                                             int nRows, int nCols, int nFilts);

__global__ void cuda_Shrink_CalU_vec4_Array(float *Y, float *U, float *X,
                                            float lambda, float *L1Weight,
                                            int nRows, int nCols, int nFilts);

__global__ void cuda_Shrink_CalU_vec4_Vector(float *Y, float *U, float *X,
                                             float lambda, float *L1Weight,
                                             int nRows, int nCols, int nFilts);

__global__ void cuda_Shrink_CalU_Vector(float *Y, float *U, float *X,
                                        float lambda, float *L1Weight,
                                        int nRows, int nCols, int nFilts);

__global__ void cuda_Shrink_CalU_vec4(float *Y, float *U, float *X,
                                      float lambda, int nRows, int nCols,
                                      int nFilts);

__global__ void cuda_Shrink_vec4(float *Y, float *X, float *U, float lambda,
                                 int nRows, int nCols, int nFilts);

__global__ void cuda_CalU_vec4(float *U, float *X, float *Y, int nRows,
                               int nCols, int nFilts);

__global__ void cuda_Cal_Dsf_C_vec4(float2 *Dsf, float2 *C, float2 *Sf,
                                    float2 *Df, float rho, int nRows, int nCols,
                                    int nFilts);

__global__ void cuda_Cal_Dsf_C(float2 *Dsf, float2 *C, float2 *Sf, float2 *Df,
                               float rho, int nRows, int nCols, int nFilts);

__global__ void cuda_Cal_C_vec4(float2 *C, float2 *Df, float rho, int nRows,
                                int nCols, int nFilts);

__global__ void cuda_Cal_C(float2 *C, float2 *Df, float rho, int nRows,
                           int nCols, int nFilts);

__global__ void cuda_Pad_Dict(float *PadD, float *D, int nCols_D, int nRows_D,
                              int nFilts, int nRows, int nCols);

__global__ void cuda_CalYU_vec4(float *YU, float *Y, float *U, int nRows,
                                int nCols, int nFilts);

__global__ void cuda_OverRelax_vec4(float *Xr, float *X, float *Y,
                                    float RelaxParam, int nRows, int nCols,
                                    int nFilts);

__global__ void cuda_Cal_residuals_norms_vec4(float *s, float *r, float *nX,
                                              float *nY, float *nU, float *X,
                                              float *Y, float *U, float *Yprv,
                                              float rho, int nRows, int nCols,
                                              int nFilts);

__global__ void cuda_norm_vec4(float *nX, float *nY, float *nU, float *X,
                               float *Y, float *U, int nRows, int nCols,
                               int nFilts);

__global__ void cuda_Cal_residuals_vec4(float *s, float *r, float *X, float *Y,
                                        float *Yprv, float rho, int nRows,
                                        int nCols, int nFilts);

__global__ void cuda_Fidelity_Term_vec4(float *Jdf, float2 *Df, float2 *Xf,
                                        float2 *Sf, int nRows, int nCols,
                                        int nFilts);

__global__ void cuda_Fidelity_Term(float *Jdf, float2 *Df, float2 *Xf,
                                   float2 *Sf, int nRows, int nCols,
                                   int nFilts);

__global__ void cuda_L1_Term_vec4(float *d_JL1, float *X, float L1Weight,
                                  int factor, int nRows, int nCols, int nFilts);

__global__ void cuda_L1_Term_vec4_Scalar_Array(float *d_JL1, float *X,
                                               float *L1Weight, int factor,
                                               int nRows, int nCols, int nFilts,
                                               int nL1Weight);

__global__ void cuda_L1_Term_vec4_Vector(float *d_JL1, float *X,
                                         float *L1Weight, int factor, int nRows,
                                         int nCols, int nFilts);

__global__ void cuda_L1_Term_Vector(float *d_JL1, float *X, float *L1Weight,
                                    int factor, int nRows, int nCols,
                                    int nFilts);

__global__ void cuda_Cal_X_minus_U_W(float *Y, float *U, float *X, int *Weight,
                                     int nRows, int nCols);

__global__ void cuda_Cal_Gfw(float *GfW, float2 *Grf, float2 *Gcf, int nRows,
                             int nCols);

__global__ void cuda_solvedbd_sm_vec4(float2 *Out, float2 *ah, float2 *Dsf,
                                      float *GfW, float *GrdWeight, float rho,
                                      float mu, float2 *YUf, float2 *c,
                                      int nRows, int nCols, int nFilts,
                                      int nWeights);

__global__ void cuda_solvedbd_sm(float2 *Out, float2 *ah, float2 *Dsf,
                                 float *GfW, float *GrdWeight, float rho,
                                 float mu, float2 *YUf, float2 *c, int nRows,
                                 int nCols, int nFilts, int nWeights);

__global__ void cuda_Fidelity_Gr_Term_vec4(float *Jdf, float *Jgr, float2 *Df,
                                           float2 *Xf, float2 *Sf, float *GfW,
                                           float *GrdWeight, int nRows,
                                           int nCols, int nFilts, int nWeights);

__global__ void cuda_Fidelity_Gr_Term(float *Jdf, float *Jgr, float2 *Df,
                                      float2 *Xf, float2 *Sf, float *GfW,
                                      float *GrdWeight, int nRows, int nCols,
                                      int nFilts, int nWeights);

__global__ void cuda_Cal_Dsf_grd_C_vec4(float2 *Dsf, float2 *C, float2 *Sf,
                                        float2 *Df, float *GfW,
                                        float *GrdWeight, float rho, float mu,
                                        int nRows, int nCols, int nFilts,
                                        int nWeights);

__global__ void cuda_Cal_Dsf_grd_C(float2 *Dsf, float2 *C, float2 *Sf,
                                   float2 *Df, float *GfW, float *GrdWeight,
                                   float rho, float mu, int nRows, int nCols,
                                   int nFilts, int nWeights);

__global__ void cuda_Cal_grd_C_vec4(float2 *C, float2 *Df, float *GfW,
                                    float *GrdWeight, float rho, float mu,
                                    int nRows, int nCols, int nFilts,
                                    int nWeights);

__global__ void cuda_Cal_grd_C(float2 *C, float2 *Df, float *GfW,
                               float *GrdWeight, float rho, float mu, int nRows,
                               int nCols, int nFilts, int nWeights);

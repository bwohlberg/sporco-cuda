// Define this to turn on error checking
#define CUDA_ERROR_CHECK

/* *********************************** */
/* *********************************** */
/*          CUDA error checking        */
/* *********************************** */
/* *********************************** */

#define checkCudaErrors(val) __checkCudaErrors__((val), #val, __FILE__, __LINE__)
#define check_LastCudaError()    __check_LastCudaError( __FILE__, __LINE__ )

template <typename T>
inline void __checkCudaErrors__(T code, const char *func, const char *file,
                                int line) {
  if (code) {
    fprintf(stderr, " \n CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            (unsigned int)code, func);
//    cudaDeviceReset();
    exit(-1);
  }
}


inline void __check_LastCudaError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();

    if ( cudaSuccess != err )
    {
        fprintf( stderr, "Last CUDA error() at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
  //      exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

#endif

    return;
}

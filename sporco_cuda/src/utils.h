//  Author: Gustavo Silva <gustavo.silva@pucp.edu.pe>

#ifndef UTILS_H
#define UTILS_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

/*******************************************/
/****        def. functions             ****/
/*******************************************/

#ifdef __cplusplus
extern "C" {
#endif

void default_opts(void *data);
void clear_opts(void *data);
int check_cuda_req(int device);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // #ifndef UTILS_H

#ifndef RECLIB_COMMON_H
#define RECLIB_COMMON_H

#ifdef WITH_CUDA
    #define RECLIB_HOST __host__
    #define RECLIB_DEVICE __device__
    #define RECLIB_HD __host__ __device__
#else
    #define RECLIB_HOST 
    #define RECLIB_DEVICE
    #define RECLIB_HD
#endif


#endif
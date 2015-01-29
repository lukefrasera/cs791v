
#include "add.h"

/*
  This is the function that each thread will execute on the GPU. The
  fact that it executes on the device is indicated by the __global__
  modifier in front of the return type of the function. After that,
  the signature of the function isn't special - in particular, the
  pointers we pass in should point to memory on the device, but this
  is not indicated by the function's signature.
 */
__global__ void add_no_stride(int *a, int *b, int *c, int elements_per_thread) {
  int offset = blockIdx.x * blockDim.x * elements_per_thread + threadIdx.x * elements_per_thread;
  for (int i = 0; i < elements_per_thread; ++i) {
    c[offset + i] = a[offset + i] + b[offset + i];
  }
}

__global__ void add_stride(int *a, int *b, int *c, int elements_per_thread) {
  int offset = blockIdx.x * blockDim.x * elements_per_thread + threadIdx.x;
  for (int i = 0; i < elements_per_thread; ++i) {
    c[offset + i * blockDim.x] = a[offset + i * blockDim.x] + b[offset + i * blockDim.x];
  }
}
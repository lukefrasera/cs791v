
#include "sum.h"

__global__ void prefix_sum(float* g_odata, float* g_idata, int n) {
  extern __shared__ float temp[];
  int thid = threadIdx.x;
  int offset = 1;
  temp[2*thid] = g_idata[2*thid]; // load input into shared memory
  temp[2*thid+1] = g_idata[2*thid+1];

  for (int d = n>>1; d > 0; d >>=1) {
    __syncthreads();
    if (thid < d) {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (thid ==0) {temp[n - 1] = 0; }

  for (int d = 1; d < n; d*=2) {
    offset >>=1;
    __syncthreads();
    if (thid < d) {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  g_odata[2*thid] = temp[2*thid];
  g_odata[2*thid+1] = temp[2*thid+1];
}

__global__ void reduce(float *g_idata, float *g_odata, unsigned int n) {
  // Pointer to shared memory
  extern __shared__ float share_mem[];
  unsigned int thread_id = threadIdx.x;
  unsigned int offset = blockIdx.x*blockDim.x*2 + threadIdx.x;

  // Temp result float
  float result = (offset < n) ? g_idata[offset] : 0;

  // Perform summation
  if (offset + blockDim.x < n)
    result += g_idata[offset+blockDim.x];
  share_mem[thread_id] = result;
  // Sync Threads in a single Block
  __syncthreads();
  
  // store result to shared memory
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (thread_id < s) {
      share_mem[thread_id] = result = result + share_mem[thread_id + s];
    }
    __syncthreads();
  }

  // Store result to output data pointer
  if (thread_id == 0) g_odata[blockIdx.x] = result;
}
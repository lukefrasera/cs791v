// "vector_sum.cc"
//
//  Copyright (c) Luke Fraser 2015
//
//  This file is part of cs791vlass.
//
//    cs791cClass is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    cs791vClass is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with cs791vClass.  If not, see <http://www.gnu.org/licenses/>.
#include <stdio.h>

__global__ void reduce(float *g_idata, float *g_odata, unsigned int n) {
  // Pointer to shared memory
  extern __shared__ float share_mem[];
  unsigned int thread_id = threadIdx.x;
  unsigned int block_id = blockIdx.x;
  unsigned int block_dim = blockDim.x;
  unsigned int offset = block_id*block_dim*2 + thread_id;

  // Temp result float
  float result = (offset < n) ? g_idata[offset] : 0;

  // Perform summation
  if (offset + block_dim < n)
    result += g_idata[offset+block_dim];
  share_mem[thread_id] = result;
  // Sync Threads in a single Block
  __syncthreads();
  
  // store result to shared memory
  for (unsigned int s=block_dim/2; s>0; s>>=1) {
    if (thread_id < s) {
      share_mem[thread_id] = result = result + share_mem[thread_id + s];
    }
    __syncthreads();
  }

  // Store result to output data pointer
  if (thread_id == 0) g_odata[block_id] = result;
}

__global__ void reduce_fix(float *g_idata, float *g_odata, unsigned int n, unsigned int s_size, unsigned int loops) {
  // Pointer to shared memory
  extern __shared__ float share_mem[];
  unsigned int thread_id = threadIdx.x;
  for (int i = 0; i < loops; ++i) {
    unsigned int offset = blockIdx.x*blockDim.x*2 + threadIdx.x + blockDim.x * 2 * gridDim.x * i;

    // Temp result float
    float result = (offset < n) ? g_idata[offset] : 0;

    // Perform summation
    if (offset + blockDim.x < n)
      result += g_idata[offset+blockDim.x];
    share_mem[thread_id] = result;
    // Sync Threads in a single Block
    int delta = s_size - blockDim.x;
    if (thread_id + delta > blockDim.x-1) {
      share_mem[thread_id+delta] = 0;
    }
    __syncthreads();
    
    // store result to shared memory
    for (unsigned int s=s_size/2; s>0; s>>=1) {
      if (thread_id < s) {
        share_mem[thread_id] = result = result + share_mem[thread_id + s];
      }
      __syncthreads();
    }

    // Store result to output data pointer
    if (thread_id == 0) {
      g_odata[blockIdx.x+ gridDim.x*i] = result;
      printf("result:%f\n", result);
    }
  }
}

// GPU RECALL

__global__ void reduce_recall(float *g_idata, float *g_odata, unsigned int n, unsigned int thresh) {
  // Pointer to shared memory
  extern __shared__ float share_mem[];
  unsigned int thread_id = threadIdx.x;
  unsigned int block_id = blockIdx.x;
  unsigned int block_dim = blockDim.x;
  unsigned int offset = block_id*block_dim*2 + thread_id;

  // Temp result float
  float result = (offset < n) ? g_idata[offset] : 0;

  // Perform summation
  if (offset + block_dim < n)
    result += g_idata[offset+block_dim];
  share_mem[thread_id] = result;
  // Sync Threads in a single Block
  __syncthreads();
  
  // store result to shared memory
  for (unsigned int s=block_dim/2; s>0; s>>=1) {
    if (thread_id < s) {
      share_mem[thread_id] = result = result + share_mem[thread_id + s];
    }
    __syncthreads();
  }

  // Store result to output data pointer
  if (thread_id == 0) g_odata[block_id] = result;
}

__global__ void reduce_fix_recall(float *g_idata, float *g_odata,
    unsigned int n, unsigned int s_size, unsigned int loops,
    unsigned int thresh) {
  // Pointer to shared memory
  extern __shared__ float share_mem[];
  unsigned int thread_id = threadIdx.x;
  for (int i = 0; i < loops; ++i) {
    unsigned int offset = blockIdx.x*blockDim.x*2 + threadIdx.x + blockDim.x * 2 * gridDim.x * i;

    // Temp result float
    float result = (offset < n) ? g_idata[offset] : 0;

    // Perform summation
    if (offset + blockDim.x < n)
      result += g_idata[offset+blockDim.x];
    share_mem[thread_id] = result;
    // Sync Threads in a single Block
    int delta = s_size - blockDim.x;
    if (thread_id + delta > blockDim.x-1) {
      share_mem[thread_id+delta] = 0;
    }
    __syncthreads();
    
    // store result to shared memory
    for (unsigned int s=s_size/2; s>0; s>>=1) {
      if (thread_id < s) {
        share_mem[thread_id] = result = result + share_mem[thread_id + s];
      }
      __syncthreads();
    }

    // Store result to output data pointer
    if (thread_id == 0) {
      g_odata[blockIdx.x+ gridDim.x*i] = result;
      if (blockIdx.x == 0 && (gridDim.x > thresh)) {
        unsigned int dev_dest_size = gridDim.x;
        unsigned int threads, blocks;
        if (dev_dest_size > 1024) {
          threads = 512;
          blocks = dev_dest_size / threads / 2;
        } else {
          threads = dev_dest_size / 2;
          blocks = 1;
        }
        float * temp = g_odata;
        g_odata = g_idata;
        g_idata = temp;
        reduce_recall<<<blocks, threads, threads*sizeof(float)>>>(g_idata, g_odata, dev_dest_size, thresh);
      }
    }
  }
}
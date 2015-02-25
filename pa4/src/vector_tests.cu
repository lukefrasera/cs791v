// "vector_tests.cc"
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
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctime>

#include "../include/vector_sum.h"

typedef unsigned int uint32;

void CudaMallocErrorCheck(void** ptr, int size);
uint32 NearestPowerTwo(uint32 N);
uint32 NearestPowerBase(uint32 N, uint32 base, uint32 &power);
void GetOptParam(int argc, char *const argv[], uint32 &threads, uint32 &blocks,
    uint32 &thresh, uint32 &N, bool &block_thread_dependent, uint32 &type);

void VectorSumCPU(float * src, uint32 N);
void VectorSumGPU(float * src, uint32 threads, uint32 blocks,
    uint32 thresh, uint32 N, uint32 loops);
void VectorSumGPURecall(float * src, uint32 threads, uint32 blocks,
    uint32 thresh, uint32 N, uint32 loops);
void SetupVector(float *&src, uint32 elements, uint32 size);
void SetupVectorZero(float *&dest, uint32 size);

////////////////////////////////////////////////////////////////////////////////
// Main Loop
////////////////////////////////////////////////////////////////////////////////
int main (int argc, char *const argv[]) {
  unsigned int threads, blocks, type = 0, thresh = 1, N;
  bool block_thread_dependent = false;
  float *src;
  // Get Operating Paramters
  GetOptParam(argc, argv, threads, blocks, thresh, N, block_thread_dependent,
    type);

  // Allocate Vector
  uint32 loops;
  uint32 n_multiple = NearestPowerBase(N, threads * blocks * 2, loops);
  SetupVector(src, N, n_multiple);
  // Perform Vector addition
  if (type == 0) { // CPU Vector Addition
    std::clock_t c_start = std::clock();
    VectorSumCPU(src, N);
    std::clock_t c_end = std::clock();
    printf("%f", 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC);
  } else if (type == 1) { // GPU Vector Addition: CPU kernel recall
    VectorSumGPU(src, threads, blocks, thresh, n_multiple, loops);
  } else if (type == 2) { // GPU Vector Addition: GPU kernel recall
    VectorSumGPURecall(src, threads, blocks, thresh, n_multiple, loops);
  } else {
    printf("Error: Not a valid type\n");
    delete [] src;
    exit(1);
  }
  delete [] src;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Other Utility and Base Functions
////////////////////////////////////////////////////////////////////////////////

void VectorSumCPU(float * src, uint32 N) {
  for (int i = 1; i < N; ++i) {
     src[i] += src[i-1];
  }
}

void VectorSumGPU(float * src, uint32 threads, uint32 blocks,
    uint32 thresh, uint32 N, uint32 loops) {
  // Setup Timing
  cudaEvent_t start, end;
  float elapsedTime=0;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Allocate Memory
  float *dev_src, *dev_dest;
  uint32 dev_dest_size = NearestPowerTwo(blocks*loops);
  CudaMallocErrorCheck((void**) &dev_src, N*sizeof(float));
  float * dest;
  SetupVectorZero(dest, dev_dest_size);
  CudaMallocErrorCheck((void**) &dev_dest, dev_dest_size*sizeof(float));

  cudaEventRecord(start, 0);
  cudaMemcpy(dev_src, src, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dest, dest, dev_dest_size*sizeof(float),
      cudaMemcpyHostToDevice);

  // GPU function Call
  uint32 share = NearestPowerTwo(threads);
  reduce_fix<<<blocks, threads, share*sizeof(float)>>>(dev_src, dev_dest, N,
      share, loops);
  // Recall GPU function: Assumption Destination is power of 2. calculate block
  //                      and threads for each call.
  // GPU Call loop until Threshold
  if (dev_dest_size > 1024) {
    threads = 512;
    blocks = dev_dest_size / threads / 2;
  } else {
    threads = dev_dest_size / 2;
    blocks = 1;
  }

  while (dev_dest_size > thresh) {
    float * temp = dev_dest;
    dev_dest = dev_src;
    dev_src = temp;
    reduce<<<blocks, threads, threads*sizeof(float)>>>(dev_src, dev_dest,
      dev_dest_size);
    dev_dest_size = blocks;
    if (dev_dest_size > 1024) {
      threads = 512;
      blocks = dev_dest_size / threads / 2;
    } else {
      threads = dev_dest_size / 2;
      blocks = 1;
    }
  }

  cudaMemcpy(dest, dev_dest, dev_dest_size*sizeof(float),
    cudaMemcpyDeviceToHost);
  // Finish on CPU or Done
  if (thresh > 1) {
    VectorSumCPU(dest, dev_dest_size);
  }
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);

  printf("%f", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(dev_src);
  cudaFree(dev_dest);
  delete [] dest;
}

void VectorSumGPURecall(float * src, uint32 threads, uint32 blocks,
    uint32 thresh, uint32 N, uint32 loops) {
  // Setup Timing
  cudaEvent_t start, end;
  float elapsedTime=0;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Allocate Memory
  float *dev_src, *dev_dest;
  uint32 dev_dest_size = NearestPowerTwo(blocks*loops);

  CudaMallocErrorCheck((void**) &dev_src, N*sizeof(float));
  float * dest;
  SetupVectorZero(dest, dev_dest_size);
  CudaMallocErrorCheck((void**) &dev_dest, dev_dest_size*sizeof(float));

  cudaEventRecord(start, 0);
  cudaMemcpy(dev_src, src, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dest, dest, dev_dest_size*sizeof(float),
      cudaMemcpyHostToDevice);

  // GPU function Call
  uint32 share = NearestPowerTwo(threads);
  reduce_fix_recall<<<blocks, threads, share*sizeof(float)>>>(dev_src, dev_dest, N,
      share, loops, thresh);

  dev_dest_size = NearestPowerTwo(thresh);
  if (dev_dest_size > thresh) {
    dev_dest_size >>= 1;
  }
  cudaMemcpy(dest, dev_dest, dev_dest_size*sizeof(float),
    cudaMemcpyDeviceToHost);
  // Finish on CPU or Done
  if (thresh > 1) {
    VectorSumCPU(dest, dev_dest_size);
  }
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  printf("%f", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(dev_src);
  cudaFree(dev_dest);
  delete [] dest;
}

void GetOptParam(int argc, char *const argv[], uint32 &threads, uint32 &blocks,
    uint32 &thresh, uint32 &N, bool &block_thread_dependent, uint32 &type) {
  char c_;
  while ((c_ = getopt(argc, argv, "t:b:n:d:v:")) != -1) {
    switch (c_) {
      case 't':  // thread number
        threads = atoi(optarg);
        break;
      case 'b':  // block number
        blocks = atoi(optarg);
        break;
      case 'n':  // CPU comute
        N = atoi(optarg);
        break;
      case 'd':
        thresh = atoi(optarg);
        break;
      case 'v':
        type = atoi(optarg);
        break;
      default:
        printf("?? getopt returned character code 0%o ??\n", c_);
    }
  }
  if (optind < argc) {
    printf("non-option ARGV-elements: ");
    while (optind < argc)
      printf("%s ", argv[optind++]);
    printf("\n");
  }
}

void SetupVector(float *&src, uint32 elements, uint32 size) {
  src = new float[size];
  for (int i = 0; i < size; ++i) {
    if (i < elements) {
      src[i] = i;
    } else {
      src[i] = 0;
    }
  }
}

void SetupVectorZero(float *&dest, uint32 size) {
  dest = new float[size];
  memset(dest, 0, size*sizeof(float));
}

void CudaMallocErrorCheck(void** ptr, int size) {
  cudaError_t err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    printf("Error: %s", cudaGetErrorString(err));
    exit(1);
  }
}

uint32 NearestPowerTwo(uint32 N) {
  uint32 result = 1;
  while (result < N) {
    result <<= 1;
  }
  return result;
}

uint32 NearestPowerBase(uint32 N, uint32 base, uint32 &power) {
  uint32 result = base;
  power = 1;
  while (result < N) {
    result += base;
    power++;
  }
  return result;
}

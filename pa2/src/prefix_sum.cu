// "prefix_sum.cu"
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
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "sum.h"

typedef unsigned int uint32;

void CudaMallocErrorCheck(void** ptr, int size);
void SequentialRecord(int N, float* a);
uint32 NearestPowerTwo(uint32 N);


int main(int argc, char *const argv[]) {
  // geopt variables
  int c_;
  bool cpu = false;
  int threads = 1, blocks = 1, N;

  // Process input parameters
  while ((c_ = getopt(argc, argv, "t:b:cn:")) != -1) {
    switch (c_) {
      case 't':  // thread number
        threads = atoi(optarg);
        break;
      case 'b':  // block number
        blocks = atoi(optarg);
        break;
      case 'c':  // CPU comute
        cpu = true;
        break;
      case 'n':
        N = atoi(optarg);
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

  // Allocate adding vectors
  float *data, *result;
  float *dev_in, *dev_out;

  data = new float[N];
  result = new float[blocks];

  // Fill host arrays
  for (int i = 0; i < N; ++i) {
    data[i] = i;
  }

  // Check if CPU computation requested
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsedTime=0;

  if (cpu) {
    // SequentialRecord(N, data);
    SequentialRecord(N, data);
    return 0;
  }

  // Allocate Cuda memory
  CudaMallocErrorCheck( (void**) &dev_in, N * sizeof(float));
  CudaMallocErrorCheck( (void**) &dev_out, blocks * sizeof(float));

  // Create event timers


  cudaMemcpy(dev_in, data, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_out, result, blocks * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);
  // Perform Gpu Computation
  // prefix_sum<<<blocks, threads, N*sizeof(float)>>>(dev_b, dev_a, N);
  int share = NearestPowerTwo(threads);
  reduce_fix<<<blocks, threads, share*sizeof(float)>>>(dev_in, dev_out, N, share);

  cudaEventRecord(end, 0);
  cudaMemcpy(result, dev_out, blocks * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);

  for (int i = 1; i < blocks; ++i) {
    result[i] += result[i-1];
  }
  // Check GPU values
  printf("%f: Sol: %f\n", elapsedTime, result[blocks-1]);
  // for (int i = 0; i < blocks; ++i){
  //   printf("elem[%d]: %f\n", i, result[i]);
  // }

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(dev_in);
  cudaFree(dev_out);

  return 0;
}

uint32 NearestPowerTwo(uint32 N) {
  uint32 result = 1;
  while (result < N) {
    result <<= 1;
  }
  return result;
}

void CudaMallocErrorCheck(void** ptr, int size) {
  cudaError_t err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    printf("Error: %s", cudaGetErrorString(err));
    exit(1);
  }
}

void SequentialRecord(int N, float* a) {
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsedTime =1;

  cudaEventRecord(start, 0);
  for (int i = 1; i < N; ++i) {
    a[i] += a[i-1];
  }
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  printf("%f", elapsedTime);
}
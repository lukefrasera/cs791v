// "addexperiment.cu"
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

#include "add.h"

void CudaMallocErrorCheck(void** ptr, int size);

int main(int argc, char *const argv[]) {
  // geopt variables
  int c_;
  bool stride = false, cpu = false;
  int threads = 1, blocks = 1, N;

  // Process input parameters
  while ((c_ = getopt(argc, argv, "st:b:cn:")) != -1) {
    switch (c_) {
      case 's':  // Striding option
        stride = true;
        break;
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
  int *a, *b, *c;
  int *dev_a, *dev_b, *dev_c;
  int element_per_thread = ceil((float)N/(float)blocks/(float)threads);
  int size = blocks * threads * element_per_thread;

  a = new int[size];
  b = new int[size];
  c = new int[size];

  // Fill host arrays
  for (int i = 0; i < size; ++i) {
    a[i] = i;
    b[i] = i;
  }

  // Check if CPU computation requested
  // time_t cstart, cend;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsedTime;

  if (cpu) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < N; ++i) {
      c[i] = a[i] + b[i];
    }
    cudaEventRecord(end, 0);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("%f\n", elapsedTime);
    return 0;
  }

  // Allocate Cuda memory
  CudaMallocErrorCheck( (void**) &dev_a, size * sizeof(int));
  CudaMallocErrorCheck( (void**) &dev_b, size * sizeof(int));
  CudaMallocErrorCheck( (void**) &dev_c, size * sizeof(int));

  // Create event timers

  cudaEventRecord(start, 0);

  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  // Perform Gpu Computation
  if (stride) {
    add_stride<<<blocks, threads>>>(dev_a, dev_b, dev_c, element_per_thread);
  } else {
    add_no_stride<<<blocks, threads>>>(dev_a, dev_b, dev_c, element_per_thread);
  }

  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&elapsedTime, start, end);

  // Check GPU values
  for (int i = 0; i < N; ++i) {
    if (c[i] != a[i] + b[i]) {
      printf("0\n");

      // clean up events - we should check for error codes here.
      cudaEventDestroy(start);
      cudaEventDestroy(end);

      cudaFree(dev_a);
      cudaFree(dev_b);
      cudaFree(dev_c);
      exit(1);
    }
  }

  printf("%f\n", elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}

void CudaMallocErrorCheck(void** ptr, int size) {
  cudaError_t err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    printf("Error: %s", cudaGetErrorString(err));
    exit(1);
  }
}

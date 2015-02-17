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

#include "vector_sum.h"

typedef unsigned int uint32;

void CudaMallocErrorCheck(void** ptr, int size);
void GetOptParam(int argc, char *const argv[], uint32 &threads, uint32 &blocks,
    uint32 thresh, uint32 N, bool &block_thread_dependent);
void VectorSumCPU(float * src, uint32 N);
void VectorSumGPU(float * src, uint32 threads, unint32 blocks,
    uint32 thresh, uint32 N);
uint32 NearestPowerTwo(uint32 N);
////////////////////////////////////////////////////////////////////////////////
// Main Loop
////////////////////////////////////////////////////////////////////////////////
int main (int argc, char *const argv[]) {
  unsigned int threads, blocks, type, thresh, N;
  bool block_thread_dependent = false;
  float *src, *dest;
  // Get Operating Paramters
  GetOptParam(argc, argv, threads, blocks, thresh, N, block_thread_dependent);

  // Allocate Vector
  uint32 size = N;
  if (block_thread_dependent) {
    size = 1;
    while (size < N) {size <<= 1;}
    if (size <= 1024) {
      blocks = 1;
      threads = size;
    } else {
      blocks = size / 1024;
      threads = 1024;
    }
  }
  SetupVector(&src, N, size);
  // Perform Vector addition
  if (type == 0) { // CPU Vector Addition
    std::clock_t c_start = std::clock();
    VectorSumCPU(src, N);
    std::clock_t c_end = std::clock();
    printf("CPU Time(ms): %f", 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC);
  } else if (type == 1) { // GPU Vector Addition: CPU kernel recall
    VectorSumGPU(src, threads, blocks, thresh, N);
  } else if (type == 2) { // GPU Vector Addition: GPU kernel recall
    VectorSumGPURecall(src, threads, blocks, thresh, N);
  } else {
    printf("Error: Not a valid type\n");
    delete [] src;
    exit(1);
  }
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

void VectorSumGPU(float * src, uint32 threads, unint32 blocks,
    uint32 thresh, uint32 N) {
  // Setup Timing
  cudaEvent_t start, end;
  float elapsedTime=0;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Allocate Memory
  float *dev_src, *dev_dest;
  CudaMallocErrorCheck((void**) &dev_src, N*sizeof(float));
  uint32 result_size = round((float) N / (float) threads);
  uint32 result_size_pad = NearestPowerTwo(result_size);
  float * dest;
  SetupVectorZero(&dest, result_size_pad);
  CudaMallocErrorCheck((void**) &dev_dest, result_size_pad);

  cudaEventRecord(start, 0);
  cudaMemcpy(dev_src, src, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dest, dest, result_size_pad*sizeof(float),
      cudaMemcpyHostToDevice);

  // GPU function Call
  Reduce<<<blocks, threads, /*Shared memory*/>>>(dev_src, dev_dest, N, result_size_pad);
  // Recall GPU function: Assumption Destination is power of 2. calculate block
  //                      and threads for each call.
  uint32 src_size = result_size_pad;
  // GPU Call loop until Threshold
  while (src_size > thresh) {
    float * temp = dev_src;
    dev_src = dev_dest;
    dev_dest = temp;
    
    if (src_size > 2048) {
      blocks = src_size / 2048;
      threads = 1024;
    } else {
      blocks = 1;
      threads = src_size / 2;
    }
    ReducePowerTwo<<<blocks, threads, /*shared mem size*/>>>(dev_src, dev_dest, N);
    src_size = blocks;
  }
  cudaMemcpy(dest, dev_dest, src_size*sizeof(float), cudaMemcpyDeviceToHost);
  // Finish on CPU or Done
  if (thresh > 1) {
    VectorSumCPU(dest, src_size);
  }
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(dev_src);
  cudaFree(dev_dest);
}

void ParallelMandelbrot(unsigned char* image, unsigned short int* iter_image, int threads, int blocks) {
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsedTime=0;

  unsigned char * dev_image;
  ushort * dev_iter;

  CudaMallocErrorCheck((void**) &dev_image, ROWS*COLS*sizeof(char));
  CudaMallocErrorCheck((void**) &dev_iter, ROWS*COLS*sizeof(ushort));

  cudaMemcpy(dev_image, image, ROWS*COLS*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_iter, iter_image, ROWS*COLS*sizeof(ushort), cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);
  Mandelbrot<<<blocks, threads>>>(dev_image, dev_iter, MaxIm, Im_factor, MinRe, Re_factor);
  cudaEventRecord(end, 0);
  cudaMemcpy(image, dev_image, ROWS*COLS*sizeof(char), cudaMemcpyDeviceToHost);
  cudaMemcpy(iter_image, dev_iter, ROWS*COLS*sizeof(ushort), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);

  printf("%f", elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(dev_iter);
  cudaFree(dev_image);
}

void SequentialMadelbrot(unsigned char * image, unsigned short int* iter_image) {
  float MinRe = -2.0;
  float MaxRe = 1.0;
  float MinIm = -1.2;
  float MaxIm = MinIm+(MaxRe-MinRe)*COLS/ROWS;
  float Re_factor = (MaxRe-MinRe)/(ROWS-1);
  float Im_factor = (MaxIm-MinIm)/(COLS-1);
  unsigned MaxIterations = 1024;

  for(unsigned y=0; y<COLS; ++y) {
    float c_im = MaxIm - y*Im_factor;
    for(unsigned x=0; x<ROWS; ++x) {
      float c_re = MinRe + x*Re_factor;
      float Z_re = c_re, Z_im = c_im;
      bool isInside = true;
      for(unsigned n=0; n<MaxIterations; ++n) {
        float Z_re2 = Z_re*Z_re, Z_im2 = Z_im*Z_im;
        if(Z_re2 + Z_im2 > 4) {
          isInside = false;
          iter_image[x * COLS + y] = n;
          break;
        }
        Z_im = 2*Z_re*Z_im + c_im;
        Z_re = Z_re2 - Z_im2 + c_re;
      }
      if(isInside) { image[x * COLS + y] = 255; }
    }
  }
}

void GetOptParam(int argc, char *const argv[], int &threads, int &blocks, bool &cpu) {
  char c_;
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

void CudaMallocErrorCheck(void** ptr, int size) {
  cudaError_t err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    printf("Error: %s", cudaGetErrorString(err));
    exit(1);
  }
}
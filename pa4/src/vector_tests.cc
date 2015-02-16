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
#include "opencv2/opencv.hpp"

#include "vector_sum.h"

#define ROWS 2000
#define COLS 2000

void CudaMallocErrorCheck(void** ptr, int size);
void GetOptParam(int argc, char *const argv[], int &threads, int &blocks, bool &cpu);
void SequentialMadelbrot(unsigned char* image, unsigned short int* iter_image);
void ParallelMandelbrot(unsigned char* image, unsigned short int* iter_image, int threads, int blocks);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main (int argc, char *const argv[]) {
  unsigned int threads, blocks, type, thresh;
  // Get Operating Paramters

  if (type == 0) {
    // CPU Vector Addition
    std::clock_t c_start = std::clock();
    SequentialMadelbrot(f_image, iter_image);
    std::clock_t c_end = std::clock();
    printf("%f", 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC);
  } else if (type == 1) {
    // GPU Vector Addition
    ParallelMandelbrot(f_image, iter_image, threads, blocks);
  } else if (type == 2) {
    // GPU 
  }

  // copy image to opencv type
  cv::Mat image(cv::Size(2000,2000), CV_8UC1, f_image);
  // save image
  cv::imwrite("mandelbrot.png", image);
  delete [] f_image;
  delete [] iter_image;
  return 0;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void ParallelMandelbrot(unsigned char* image, unsigned short int* iter_image, int threads, int blocks) {
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsedTime=0;

  float MinRe = -2.0;
  float MaxRe = 1.0;
  float MinIm = -1.2;
  float MaxIm = MinIm+(MaxRe-MinRe)*COLS/ROWS;
  float Re_factor = (MaxRe-MinRe)/(ROWS-1);
  float Im_factor = (MaxIm-MinIm)/(COLS-1);

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
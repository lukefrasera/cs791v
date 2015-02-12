// "mandelbrot_gpu.cu"
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
#include "mandelbrot_gpu.h"
#define COLS 2000
#define ITER 1024
__global__ void Mandelbrot(unsigned char * image, ushort * iter) {
  unsigned int blockid = blockIdx.x;
  unsigned int threadid = threadIdx.x;
  unsigned int blockdim = blockDim.x;
  unsigned int index = blockid*blockdim + threadid;
  // Caluclate Cartiseian coordinates
  unsigned int x = index / COLS;
  unsigned int y = index - x * COLS;

  float c_im = MaxIm - y*Im_factor;
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
// "mandelbrot.cu"
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

void GetOptParam(int &threads, int &blocks, bool &cpu);
void SequentialMadelbrot(/*image parameters*/);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main (int argc, char *const argv[]) {

  // Get Operating Paramters
  GetOptParam(argc, argv, threads, blocks, cpu);

  // Allocate image

  // Determine IF CPU or GPU
  if (cpu) {
    SequentialMadelbrot(/*image parameters*/);
  }

  // GPU mandelbrot
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void SequentialMadelbrot(/*image parameters*/) {
  double MinRe = -2.0;
  double MaxRe = 1.0;
  double MinIm = -1.2;
  double MaxIm = MinIm+(MaxRe-MinRe)*ImageHeight/ImageWidth;
  double Re_factor = (MaxRe-MinRe)/(ImageWidth-1);
  double Im_factor = (MaxIm-MinIm)/(ImageHeight-1);
  unsigned MaxIterations = 30;

  for(unsigned y=0; y<ImageHeight; ++y) {
    double c_im = MaxIm - y*Im_factor;
    for(unsigned x=0; x<ImageWidth; ++x) {
      double c_re = MinRe + x*Re_factor;
      double Z_re = c_re, Z_im = c_im;
      bool isInside = true;
      for(unsigned n=0; n<MaxIterations; ++n) {
        double Z_re2 = Z_re*Z_re, Z_im2 = Z_im*Z_im;
        if(Z_re2 + Z_im2 > 4) {
          isInside = false;
          break;
        }
        Z_im = 2*Z_re*Z_im + c_im;
        Z_re = Z_re2 - Z_im2 + c_re;
      }
      if(isInside) { putpixel(x, y); }
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
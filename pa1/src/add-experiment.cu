//"addexperiment.cu"
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

#include "add.h"

int main(int argc, char const *argv[]) {
  // geopt variables
  int c;
  bool stride = false, cpu = false;
  int threads = 1, blocks = 1;

  // Process input parameters
  while ((c = getopt(argc, argv, "st:b:c")) != -1) {
    switch (c) {
      case 's': // Striding option
        stride = true;
        break;
      case 't': // thread number
        threads = atoi(optarg);
        break;
      case 'b': // block number
        blocks = atoi(optarg);
        break;
      case 'c': // CPU comute
        cpu = true;
        break;
      default:
        printf ("?? getopt returned character code 0%o ??\n", c);
    }
  }
  if (optind < argc) {
    printf ("non-option ARGV-elements: ");
    while (optind < argc)
      printf ("%s ", argv[optind++]);
    printf ("\n");
  }

  // Allocate adding vectors
  int *a, *b, *c;
  return 0;
}
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
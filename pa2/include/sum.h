#ifndef ADD_H_
#define ADD_H_

__global__ void prefix_sum(float*, float*, int);
__global__ void reduce(float *g_idata, float *g_odata, unsigned int n);
__global__ void reduce_fix(float *g_idata, float *g_odata, unsigned int n, unsigned int s_size);

#endif // ADD_H_
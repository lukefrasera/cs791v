cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(
  include
)

#cuda_add_executable( add-experimet
#  src/add-experiment.cu
#  src/add.cu
#)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS -arch=compute_35 -lcudadevrt -rdc=true)
cuda_add_executable(sum
  src/vector_tests.cu
  src/vector_sum.cu
)

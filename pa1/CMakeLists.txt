cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(
  include
)

cuda_add_executable( hellocuda
  src/hellocuda.cu
  src/add.h
  src/add.cu
)

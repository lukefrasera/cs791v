cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(
  include
)

cuda_add_executable( add-experimet
  src/add-experiment.cu
  src/add.cu
)

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
find_package( OpenCV REQUIRED )

include_directories(
  include
)

#cuda_add_executable( add-experimet
#  src/add-experiment.cu
#  src/add.cu
#)

cuda_add_executable(mandelbrot
  src/mandelbrot.cu
  src/mandelbrot_gpu.cu
)
target_link_libraries( mandelbrot ${OpenCV_LIBS} )
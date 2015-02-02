cmake_minimum_required(VERSION 2.8)

project(cs7901v)
find_package(CUDA QUIET REQUIRED)

add_subdirectory(
  pa0
  pa1
  pa2
)

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

set(FLAGRAND_CPP_BACKEND "IX" CACHE STRING "Target backend for the FlagRAND C++ wrapper")
set_property(CACHE FLAGRAND_CPP_BACKEND PROPERTY STRINGS "CUDA" "IX")
if(NOT FLAGRAND_CPP_BACKEND MATCHES "^(CUDA|IX)$")
  message(FATAL_ERROR "Unsupported FlagRAND C++ backend: ${FLAGRAND_CPP_BACKEND}")
endif()

project(FlagRANDCpp LANGUAGES CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(FLAGRAND_BUILD_CPP_WRAPPER "Build the FlagRAND C++ wrapper" ON)
if(NOT FLAGRAND_BUILD_CPP_WRAPPER)
  return()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch MODULE REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cudart cuda_driver)

execute_process(
  COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)
set(BACKEND ${FLAGRAND_CPP_BACKEND} CACHE STRING "libtriton_jit backend" FORCE)
set(TRITON_JIT_INSTALL OFF CACHE BOOL "Do not install libtriton_jit separately" FORCE)
set(TRITON_JIT_BUILD_EXAMPLES OFF CACHE BOOL "Do not build libtriton_jit examples" FORCE)
FetchContent_Declare(
  TritonJIT
  GIT_REPOSITORY https://github.com/flagos-ai/libtriton_jit.git
  GIT_TAG v0.2.0-rc0.1
  GIT_SHALLOW TRUE
  PATCH_COMMAND
    ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR> -P
    ${CMAKE_CURRENT_LIST_DIR}/cmake/PatchTritonJITIX.cmake)
FetchContent_MakeAvailable(TritonJIT)

add_library(flagrand_operators SHARED lib/operators.cpp lib/utils.cpp)
target_include_directories(
  flagrand_operators
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(flagrand_operators PRIVATE BACKEND_${FLAGRAND_CPP_BACKEND})
target_link_libraries(
  flagrand_operators
  PUBLIC Torch::Torch
  PRIVATE TritonJIT::triton_jit CUDA::cudart CUDA::cuda_driver)

pybind11_add_module(c_operators csrc/bindings.cpp)
target_compile_definitions(c_operators PRIVATE BACKEND_${FLAGRAND_CPP_BACKEND})
target_link_libraries(c_operators PUBLIC Torch::Torch_Python PRIVATE flagrand_operators)

install(TARGETS flagrand_operators c_operators LIBRARY DESTINATION ".")
install(DIRECTORY triton_src/ DESTINATION "triton_src")
install(DIRECTORY include/flagrand DESTINATION "include")

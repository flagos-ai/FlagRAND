if(NOT DEFINED Torch_ROOT)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE Torch_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_package(Torch CONFIG REQUIRED)

if(NOT TARGET Torch::Torch)
  add_library(Torch::Torch INTERFACE IMPORTED)
  target_include_directories(Torch::Torch INTERFACE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(Torch::Torch INTERFACE ${TORCH_LIBRARIES})
  target_compile_options(Torch::Torch INTERFACE ${TORCH_CXX_FLAGS})

  find_library(
    torch_python_lib
    NAMES torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    REQUIRED)
  add_library(Torch::Torch_Python INTERFACE IMPORTED)
  target_include_directories(Torch::Torch_Python INTERFACE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(Torch::Torch_Python INTERFACE ${torch_python_lib})
  target_compile_options(Torch::Torch_Python INTERFACE ${TORCH_CXX_FLAGS})
endif()

add_compile_options(${TORCH_CXX_FLAGS})

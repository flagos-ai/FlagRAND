if(TARGET Torch::Torch)
  set(Torch_FOUND TRUE)
  return()
endif()

execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "import torch; print(torch.__path__[0]); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_PYTHON_INFO
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" TORCH_PYTHON_INFO "${TORCH_PYTHON_INFO}")
list(GET TORCH_PYTHON_INFO 0 TORCH_INSTALL_PREFIX)
list(GET TORCH_PYTHON_INFO 1 TORCH_CXX11_ABI)

set(TORCH_INCLUDE_DIRS
    "${TORCH_INSTALL_PREFIX}/include"
    "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include")
set(TORCH_LIBRARY_DIR "${TORCH_INSTALL_PREFIX}/lib")
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")

foreach(TORCH_COMPONENT c10 torch_cpu torch c10_cuda torch_cuda torch_python)
  find_library(
    TORCH_${TORCH_COMPONENT}_LIBRARY
    NAMES ${TORCH_COMPONENT}
    PATHS "${TORCH_LIBRARY_DIR}"
    NO_DEFAULT_PATH
    REQUIRED)
endforeach()

add_library(Torch::Torch INTERFACE IMPORTED)
target_include_directories(Torch::Torch INTERFACE ${TORCH_INCLUDE_DIRS})
target_link_libraries(
  Torch::Torch
  INTERFACE
    ${TORCH_c10_LIBRARY}
    ${TORCH_torch_cpu_LIBRARY}
    ${TORCH_torch_LIBRARY}
    ${TORCH_c10_cuda_LIBRARY}
    ${TORCH_torch_cuda_LIBRARY})
target_compile_options(Torch::Torch INTERFACE ${TORCH_CXX_FLAGS})

add_library(Torch::Torch_Python INTERFACE IMPORTED)
target_include_directories(Torch::Torch_Python INTERFACE ${TORCH_INCLUDE_DIRS})
target_link_libraries(
  Torch::Torch_Python
  INTERFACE Torch::Torch ${TORCH_torch_python_LIBRARY})
target_compile_options(Torch::Torch_Python INTERFACE ${TORCH_CXX_FLAGS})

add_compile_options(${TORCH_CXX_FLAGS})
set(TORCH_LIBRARIES
    ${TORCH_c10_LIBRARY}
    ${TORCH_torch_cpu_LIBRARY}
    ${TORCH_torch_LIBRARY}
    ${TORCH_c10_cuda_LIBRARY}
    ${TORCH_torch_cuda_LIBRARY})
set(Torch_FOUND TRUE)

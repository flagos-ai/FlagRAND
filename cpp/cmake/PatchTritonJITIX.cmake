file(READ "${SOURCE_DIR}/CMakeLists.txt" TRITON_JIT_CMAKE)
string(
  REPLACE
    "else()\n    project(TritonJIT LANGUAGES CUDA CXX VERSION 0.1.0)"
    "elseif(BACKEND STREQUAL \"IX\")\n    project(TritonJIT LANGUAGES CXX VERSION 0.1.0)\nelse()\n    project(TritonJIT LANGUAGES CUDA CXX VERSION 0.1.0)"
    TRITON_JIT_CMAKE
    "${TRITON_JIT_CMAKE}")
file(WRITE "${SOURCE_DIR}/CMakeLists.txt" "${TRITON_JIT_CMAKE}")

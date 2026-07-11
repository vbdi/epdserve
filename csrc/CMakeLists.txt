cmake_minimum_required(VERSION 3.10)
project(cuda_ipc_utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -D_GLIBCXX_USE_CXX11_ABI=0")

# Locate environment variables
if (NOT DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME "/usr/local/cuda")
    message(STATUS "CUDA_HOME is not set. Using default: ${CUDA_HOME}")
else()
    set(CUDA_HOME $ENV{CUDA_HOME})
    message(STATUS "CUDA_HOME is set to: ${CUDA_HOME}")
endif()

if (NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "CONDA_PREFIX environment variable not set. Please activate your Conda environment.")
endif()

# Paths to PyTorch and Python headers in the Conda environment
set(CONDA_ENV_PATH $ENV{CONDA_PREFIX})
set(TORCH_HOME "${CONDA_ENV_PATH}/lib/python3.10/site-packages/torch")
set(PYTHON_INCLUDE "${CONDA_ENV_PATH}/include/python3.10")

# Include and link directories
include_directories(
    ${PYTHON_INCLUDE}  # Python headers
    ${CUDA_HOME}/include
    ${TORCH_HOME}/include
    ${TORCH_HOME}/include/torch/csrc/api/include
    ${TORCH_HOME}/include/TH
    ${TORCH_HOME}/include/THC
)

link_directories(${CUDA_HOME}/lib64)
link_directories(${TORCH_HOME}/lib)

# Source files
set(SOURCES
    block_migration.cpp
    zero_copy.cpp
    py_nccl.cc
)

# Output shared library
add_library(cuda_ipc_utils SHARED ${SOURCES})

# Link libraries
target_link_libraries(cuda_ipc_utils
    c10
    torch
    torch_cpu
    torch_cuda
    c10_cuda
    nccl
)

# Print information for debugging
message(STATUS "CUDA_HOME: ${CUDA_HOME}")
message(STATUS "CONDA_ENV_PATH: ${CONDA_ENV_PATH}")
message(STATUS "TORCH_HOME: ${TORCH_HOME}")
message(STATUS "PYTHON_INCLUDE: ${PYTHON_INCLUDE}")

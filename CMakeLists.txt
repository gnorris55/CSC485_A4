
cmake_minimum_required(VERSION 3.10)

project(cuda LANGUAGES CUDA CXX)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)


set(SOURCES "./src/main.cu")
list(APPEND SOURCES "./src/cpu_baseline.cu" "./src/gpu_solution.cu")

# Create an executable with all appended source files

add_executable("${CMAKE_PROJECT_NAME}" ${SOURCES})
#include_directories("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include")  # On Windows
target_include_directories("${CMAKE_PROJECT_NAME}" PUBLIC "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include" "./include/")


set_target_properties("${CMAKE_PROJECT_NAME}" PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


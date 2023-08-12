cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(compnal DESCRIPTION "Condensed Matter Physics Numerical Analytics Libirary")

enable_language(CXX)

set(COMPNAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

# Set OpenMP
if(APPLE)
  message(STATUS "Try to enable OpenMP from Homebrew on Apple")
  execute_process(COMMAND brew --prefix libomp COMMAND tr -d '\n' OUTPUT_VARIABLE OMP_DIR)
  include_directories("${OMP_DIR}/include")
  link_directories("${OMP_DIR}/lib")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
else()
  find_package(OpenMP)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP Found!")
  else()
    message(STATUS "OpenMP NOT Found!")
  endif()
endif()
  
# Set Eigen3
find_package(Eigen3 CONFIG)
add_subdirectory(include)

# Build Python Extension 
message(STATUS "Build Python Extension.")
find_package(Python3 COMPONENTS Interpreter Development NumPy)
# Scikit-Build does not add your site-packages to the search path
# automatically, so we need to add it _or_ the pybind11 specific directory
# here.
if(NOT DEFINED PYTHON_EXECUTABLE) 
  message(STATUS "NOT DEFINED PYTHON_EXECUTABLE")
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(compnal)
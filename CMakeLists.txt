# Copyright 2023 Kohei Suzuki

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(compnal DESCRIPTION "Condensed Matter Physics Numerical Analytics Libirary")

set(COMPNAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

# Eigen3 
find_package(Eigen3 CONFIG)
if(EIGEN3_FOUND)
  message(STATUS "Skip Download eigen3")
else()
  include(dependencies/FetchEigen.cmake)
endif()

# pybind11
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
if(PYBIND11_FOUND)
  message(STATUS "Skip Download pybind11")
else()
  include(dependencies/FetchPybind11.cmake)
endif()

# OpenMP
if(APPLE)
  message(STATUS "Enable OpenMP forcely from Homebrew on Apple")
  execute_process(COMMAND brew --prefix libomp COMMAND tr -d '\n' OUTPUT_VARIABLE OMP_DIR)
  include_directories("${OMP_DIR}/include")
  link_directories("${OMP_DIR}/lib")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
else()
  find_package(OpenMP REQUIRED)
endif()

add_subdirectory(include)
add_subdirectory(compnal)


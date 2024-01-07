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

# Standard includes
INCLUDE (CheckLanguage)

set(COMPNAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# IF (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
#     IF (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
#         SET (lines "You are building in-place.")
#         MESSAGE (AUTHOR_WARNING ${lines})
#     ENDIF ()
#     SET (compnal_MAIN_PROJECT ON)
#     MESSAGE (STATUS "CMake ${CMAKE_VERSION}")
#     IF (CMAKE_CXX_STANDARD)
#         SET (CMAKE_CXX_EXTENSIONS OFF)
#         SET (CMAKE_CXX_STANDARD_REQUIRED ON)
#     ENDIF ()
# ELSE ()
#     SET (compnal_MAIN_PROJECT OFF)
# ENDIF ()

# Options
# OPTION (compnal_INSTALL "Install compnal header files?" ${compnal_MAIN_PROJECT})
# OPTION (compnal_TEST "Build compnal test suite?" ${compnal_MAIN_PROJECT})
# OPTION (compnal_DOCS "Build compnal docs?" ${compnal_MAIN_PROJECT})
# OPTION (BUILD_DOCS "Enable Doxygen support." OFF)
# OPTION (CMAKE_REQUIRE_FAILE "If CMake could not find dependencies, build will faile." OFF)

# LIST (APPEND CMAKE_MODULE_PATH dependencies)

# CHECK_LANGUAGE (C)
# CHECK_LANGUAGE (CXX)

# IF (CMAKE_C_COMPILER)
#     ENABLE_LANGUAGE (C)
# ENDIF ()
# IF (CMAKE_CXX_COMPILER)
#     ENABLE_LANGUAGE (CXX)
# ENDIF ()

# IF (UNIX AND NOT APPLE)
#     CHECK_LANGUAGE (Fortran)
#     IF (CMAKE_Fortran_COMPILER)
#         ENABLE_LANGUAGE (Fortran)
#     ENDIF ()
# ENDIF ()

# IF (CMAKE_REQUIRE_FAILE)
#     SET (CMAKE_REQUIRE_FIND_PACKAGE_Eigen3 ON)
#     SET (CMAKE_REQUIRE_FIND_PACKAGE_pybind11 ON)
# ENDIF ()

# detect operating system and host processor
# MESSAGE (STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
# MESSAGE (STATUS "The host processor is ${CMAKE_HOST_SYSTEM_PROCESSOR}")
# MESSAGE (STATUS "CMAKE_GENERATOR = ${CMAKE_GENERATOR}")
# MESSAGE (STATUS "CIBUILDWHEEL = $ENV{CIBUILDWHEEL}")
# MESSAGE (STATUS "CMAKE_REQUIRE_FIND_PACKAGE_Eigen3 = ${CMAKE_REQUIRE_FIND_PACKAGE_Eigen3}")
# MESSAGE (STATUS "CMAKE_REQUIRE_FIND_PACKAGE_pybind11 = ${CMAKE_REQUIRE_FIND_PACKAGE_pybind11}")
# MESSAGE (STATUS "Started CMake for ${PROJECT_NAME} \n")

# IF (APPLE)
#     SET (CMAKE_MACOSX_RPATH TRUE)
#     MESSAGE (STATUS "CMAKE_OSX_DEPLOYMENT_TARGET = ${CMAKE_OSX_DEPLOYMENT_TARGET}")
#     ADD_COMPILE_OPTIONS (-Wno-unused-command-line-argument)
# ENDIF ()

SET (CMAKE_CXX_STANDARD 17)
SET (CXX_STANDARD_REQUIRED TRUE)
SET (CMAKE_VERBOSE_MAKEFILE TRUE)

# IF (NOT APPLE)
#     IF ($ENV{CIBUILDWHEEL})
#         MESSAGE (STATUS "ENABLE BLAS STATIC")
#         SET (BLA_STATIC ON CACHE BOOL "" FORCE)
#     ENDIF ()
# ENDIF ()

# FIND_PACKAGE (BLAS)
# FIND_PACKAGE (LAPACK)

##### Set default behavior #####
# SET (DEFAULT_USE_OMP ON)

# IF (NOT APPLE)
#     IF (NOT DEFINED USE_OMP)
#         SET (USE_OMP ${DEFAULT_USE_OMP})
#     ENDIF ()
# ELSEIF ($ENV{CIBUILDWHEEL})
#     MESSAGE (STATUS "OpenMP is disabled on macos for distribution binaries.")
# ELSE ()
#     SET (USE_OMP ${DEFAULT_USE_OMP})
# ENDIF ()

# IF (USE_OMP)
#     FIND_PACKAGE (OpenMP)
#     IF (OpenMP_FOUND)
#         MESSAGE (STATUS "OMP_FOUND = YES")
#         MESSAGE (STATUS "USE_OMP = ON")
#         IF (WIN32)
#             SET (CMAKE_INSTALL_OPENMP_LIBRARIES ON)
#         ENDIF ()
#         ADD_DEFINITIONS (-DUSE_OMP)
#     ELSE ()
#         MESSAGE (STATUS "OMP_FOUND = NO")
#         MESSAGE (STATUS "USE_OMP = OFF")
#         SET (USE_OMP OFF)
#     ENDIF ()
# ENDIF ()

FIND_PACKAGE (Eigen3 CONFIG)

IF (EIGEN3_FOUND)
    MESSAGE (STATUS "Skip Download eigen3")
ELSE ()
    MESSAGE (STATUS "Downlod eigen3")
    INCLUDE (dependencies/FetchEigen.cmake)
ENDIF ()

if(APPLE)
  message(STATUS "Enable OpenMP forcely from Homebrew on Apple")
  execute_process(COMMAND brew --prefix libomp COMMAND tr -d '\n' OUTPUT_VARIABLE OMP_DIR)
  include_directories("${OMP_DIR}/include")
  link_directories("${OMP_DIR}/lib")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
endif()

ADD_SUBDIRECTORY (include)

# Build Python Extension 
MESSAGE (STATUS "Build Python Extension.")
# OPTION (WITH_THREAD "Compile in rudimentary thread support" ON)
# OPTION (BUILD_TESTING "Enable CTest support." OFF)
FIND_PACKAGE (Python COMPONENTS Interpreter Development)
# Scikit-Build does not add your site-packages to the search path
# automatically, so we need to add it _or_ the pybind11 specific directory
# here.
EXECUTE_PROCESS (
        COMMAND "${PYTHON_EXECUTABLE}" -c
        "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
LIST (APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
FIND_PACKAGE (pybind11 CONFIG REQUIRED)
    
ADD_SUBDIRECTORY (compnal)


cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(compnal DESCRIPTION "Condensed Matter Physics Numerical Analytics Libirary")

include(CheckLanguage)

set(COMPNAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(COMPNAL_MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(COMPNAL_MAIN_PROJECT ON)
endif()

check_language(CXX)

list(APPEND CMAKE_MODULE_PATH external)

if(CMAKE_CXX_COMPILER)
  enable_language(CXX)
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED TRUE)

find_package(OpenMP)
if(OpenMP_FOUND) 
  message(STATUS "OMP_FOUND = YES")
  message(STATUS "USE_OMP = ON")
  add_definitions(-DUSE_OMP)
  add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

add_subdirectory(include)

# Build Python Extension 
if(SKBUILD AND COMPNAL_MAIN_PROJECT)
  message(STATUS "Build Python Extension.")
  option(WITH_THREAD "Compile in rudimentary thread support" ON)
  option(BUILD_TESTING "Enable CTest support." OFF)
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
  
elseif(COMPNAL_MAIN_PROJECT AND ((${CMAKE_BUILD_TYPE} MATCHES Debug) OR COMPNAL_TEST))
  include(CTest)
  if(ENABLE_COVERAGE AND (NOT MSVC))
    message(STATUS "Use Codecov")
    find_package(codecov)
  endif()
  
  message(STATUS "Use Googletest") 
  message(STATUS "Downlod Googletest")
  include(external/googletest.cmake)
  
  enable_testing()
  
  add_subdirectory(tests)
  
  if(ENABLE_COVERAGE AND (NOT MSVC))
     coverage_evaluate()
  endif()
endif()
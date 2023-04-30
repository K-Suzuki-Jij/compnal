message(CHECK_START "compnal")

message(STATUS "Build compnal")

set(python_module_name base_compnal)

add_library(${python_module_name} MODULE base_compnal/pybind_main.cpp)

target_include_directories(${python_module_name} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

pybind11_extension(${python_module_name})

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
  # Strip unnecessary sections of the binary on Linux/macOS
  pybind11_strip(${python_module_name})
endif()

target_compile_definitions(${python_module_name} PRIVATE 
    PYTHON_VERSION_INFO=${PYTHON_VERSION_STRING}
)

target_link_libraries(${python_module_name}  PRIVATE
  cxx_compnal
  pybind11::module
  pybind11::thin_lto
  pybind11::windows_extras
)

install(TARGETS ${python_module_name} LIBRARY DESTINATION .)

message(CHECK_PASS "Finished")

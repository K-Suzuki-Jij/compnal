message(CHECK_START "base_compnal")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

message(STATUS "Build base_compnal")

set(python_module_name base_compnal)

add_library(${python_module_name} MODULE base_compnal/pybind11_main.cpp)

target_include_directories(${python_module_name} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

pybind11_extension(${python_module_name})

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(${python_module_name})
endif()

if(CUDAToolkit_FOUND)
    set_target_properties(${python_module_name} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET "hidden"
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    set_target_properties(${python_module_name} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET "hidden"
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

target_compile_definitions(${python_module_name} PRIVATE 
    PYTHON_VERSION_INFO=${PYTHON_VERSION_STRING}
)

target_link_libraries(${python_module_name}  PRIVATE
      base_compnal_header_only
      pybind11::module
      pybind11::thin_lto
      pybind11::windows_extras
)

install(TARGETS ${python_module_name} LIBRARY DESTINATION .)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "Finished")
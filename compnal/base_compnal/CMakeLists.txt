add_library(base_compnal MODULE pybind11_main.cpp)

set_target_properties(base_compnal PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBLITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    VERSION_INFO "${COMPNAL_VERSION_INFO}"
)

target_link_libraries(base_compnal
    pybind11::module
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
)

add_custom_target(python
    DEPENDS base_compnal
)
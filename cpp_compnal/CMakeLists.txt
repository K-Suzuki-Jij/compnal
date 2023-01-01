message(CHECK_START "base_compnal_header_only")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

add_library(base_compnal_header_only INTERFACE)

target_include_directories(base_compnal_header_only INTERFACE 
  $<BUILD_INTERFACE:${COMPNAL_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(base_compnal_header_only INTERFACE 
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
    $<$<TARGET_EXISTS:BLAS::BLAS>:BLAS::BLAS>
    $<$<TARGET_EXISTS:LAPACK::LAPACK>:LAPACK::LAPACK>
)

target_compile_definitions(base_compnal_header_only INTERFACE 
    EIGEN_MPL2_ONLY
    BUILD_TESTING=OFF
    TEST_LIB=OFF
    EIGEN_BUILD_PKGCONFIG=OFF
    EIGEN_BUILD_DOC=OFF
    EIGEN_DOC_USE_MATHJAX=OFF 
    EIGEN_BUILD_TESTING=OFF 
    EIGEN_TEST_NOQT=OFF 
    EIGEN_LEAVE_TEST_IN_ALL_TARGET=OFF 
    $<$<TARGET_EXISTS:BLAS::BLAS>:EIGEN_USE_BLAS>
    $<$<TARGET_EXISTS:LAPACK::LAPACK>:EIGEN_USE_LAPACKE>
    $<$<CXX_COMPILER_ID:MSVC>:EIGEN_Fortran_COMPILER_WORKS=OFF>   
)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "Finished")

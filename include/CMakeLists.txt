message(CHECK_START "cxx_compnal")

add_library(cxx_compnal INTERFACE)

target_include_directories(cxx_compnal INTERFACE 
  $<BUILD_INTERFACE:${COMPNAL_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(cxx_compnal INTERFACE 
  $<$<TARGET_EXISTS:Eigen3::Eigen>:Eigen3::Eigen>
  $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
  $<$<TARGET_EXISTS:BLAS::BLAS>:BLAS::BLAS>
  $<$<TARGET_EXISTS:LAPACK::LAPACK>:LAPACK::LAPACK>
)

target_compile_definitions(cxx_compnal INTERFACE 
  BUILD_TESTING=OFF
  TEST_LIB=OFF
)

message(CHECK_PASS "Finished")

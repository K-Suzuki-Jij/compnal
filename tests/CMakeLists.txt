add_executable(cxx_compnal_test
    cpp_tests/test_main.cpp
)

if(ENABLE_COVERAGE AND (NOT MSVC))
  add_coverage(cxx_compnal_test)
endif()

target_include_directories(cxx_compnal_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(cxx_compnal_test PRIVATE
    BUILD_TESTING=OFF
)

target_link_libraries(cxx_compnal_test PRIVATE
    cxx_compnal
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,gtest>
    $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,gtest_main>
    $<IF:$<TARGET_EXISTS:GTest::gmock>,GTest::gmock,gmock>
    $<IF:$<TARGET_EXISTS:GTest::gmock_main>,GTest::gmock_main,gmock_main>
)

gtest_discover_tests(cxx_compnal_test)

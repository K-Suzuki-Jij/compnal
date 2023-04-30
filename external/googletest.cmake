include(FetchContent)

message(CHECK_START "Fetching GoogleTest")
set(CMAKE_CXX_STANDARD 17)

#### Google test ####
FetchContent_Declare(
    googletest
    GIT_REPOSITORY  https://github.com/google/googletest
    GIT_TAG         release-1.12.1
    GIT_SHALLOW     TRUE
)

FetchContent_MakeAvailable(googletest)

find_package(GTest)

message(CHECK_PASS "fetched")

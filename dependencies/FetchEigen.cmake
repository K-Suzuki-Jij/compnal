include(FetchContent)

message(CHECK_START "Downloading Eigen3...")
FetchContent_Declare(
    eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
    GIT_TAG         3.4.0
    GIT_SHALLOW     TRUE
    )

FetchContent_MakeAvailable(eigen)
add_library(compnal INTERFACE)
target_include_directories(compnal INTERFACE ${eigen_SOURCE_DIR})
message(CHECK_PASS "Eigen3 download complete.")
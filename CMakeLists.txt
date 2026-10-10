cmake_minimum_required(VERSION 3.10)
project(mini_dl_compiler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# No -ffast-math on purpose: fused and unfused kernels must give identical results.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_library(dlc STATIC src/graph.cpp src/parser.cpp src/executor.cpp src/passes.cpp)
target_include_directories(dlc PUBLIC include)
target_compile_options(dlc PRIVATE -Wall -Wextra)

add_executable(dlc_cli src/main.cpp)
target_link_libraries(dlc_cli dlc)
set_target_properties(dlc_cli PROPERTIES OUTPUT_NAME dlc)

enable_testing()
add_executable(test_dlc tests/test_dlc.cpp)
target_link_libraries(test_dlc dlc)
add_test(NAME unit_tests COMMAND test_dlc)

cmake_minimum_required(VERSION 3.13)

project(neural-networks-from-scratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# For code visibility
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${EIGEN_INCLUDE_DIR})

set(EIGEN3_DIR "${CMAKE_SOURCE_DIR}/pkg/eigen-3.4.0")
set(ANYMOVABLE_DIR "${CMAKE_SOURCE_DIR}/pkg/AnyMovable")

file(GLOB SOURCES "src/*.cpp")

add_executable(neural-net ${SOURCES})

target_include_directories(neural-net PUBLIC ${EIGEN3_DIR})
target_include_directories(neural-net PUBLIC ${ANYMOVABLE_DIR})

add_subdirectory(src)
add_subdirectory(pkg)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

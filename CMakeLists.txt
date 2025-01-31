cmake_minimum_required(VERSION 3.13)

project(neural-networks-from-scratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# For code visibility in VSCode
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)
add_subdirectory(pkg)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

file(GLOB SOURCES "src/*.cpp")

add_executable(neural-net ${SOURCES})

target_link_libraries(neural-net Eigen3::Eigen)
target_link_libraries(neural-net AnyMovable)

# For all local libraries
target_include_directories(AnyMovable INTERFACE "${PROJECT_SOURCE_DIR}/pkg/AnyObject")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

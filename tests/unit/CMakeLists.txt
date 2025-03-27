add_executable(test_loss test_loss.cpp)

target_link_libraries(test_loss PRIVATE neural-net Catch2::Catch2WithMain)

# Register tests
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/pkg/Catch2/extras)
include(CTest)
include(Catch)

catch_discover_tests(test_loss)
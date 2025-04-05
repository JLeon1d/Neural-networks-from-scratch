add_executable(test_loss test_loss.cpp utils.cpp)
add_executable(test_optimizer test_optimizer.cpp utils.cpp)
add_executable(test_layer test_layer.cpp utils.cpp)
add_executable(test_data_loader test_data_loader.cpp utils.cpp)

add_executable(test_all test_all.cpp
               test_loss.cpp
               test_optimizer.cpp
               test_layer.cpp
               test_data_loader.cpp
               test_network.cpp
               utils.cpp)

target_link_libraries(test_loss PRIVATE neural-net Catch2::Catch2WithMain)
target_link_libraries(test_optimizer PRIVATE neural-net Catch2::Catch2WithMain)
target_link_libraries(test_layer PRIVATE neural-net Catch2::Catch2WithMain)
target_link_libraries(test_data_loader PRIVATE neural-net Catch2::Catch2WithMain)

target_link_libraries(test_all PRIVATE neural-net Catch2::Catch2WithMain)

# Register tests
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/pkg/Catch2/extras)
include(CTest)
include(Catch)

catch_discover_tests(test_loss)
catch_discover_tests(test_optimizer)
catch_discover_tests(test_layer)
catch_discover_tests(test_data_loader)
catch_discover_tests(test_all)
add_executable(test_mnist test_mnist.cpp mnist_data_loader.cpp)

target_link_libraries(test_mnist PRIVATE neural-net)

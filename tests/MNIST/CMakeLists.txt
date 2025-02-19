add_executable(test_mnist test_mnist.cpp mnist_data_loader.cpp)

# target_link_libraries(tests Catch2::Catch2)
target_link_libraries(test_mnist PRIVATE neural-net)

# target_compile_features(tests PUBLIC cxx_std_11)
# set_target_properties(tests PROPERTIES CXX_EXTENSIONS OFF)

# add_test(NAME tests COMMAND tests)
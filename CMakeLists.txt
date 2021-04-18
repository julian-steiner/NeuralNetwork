cmake_minimum_required(VERSION 3.0.0)
project(NeuralNetwork VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(googletest)

include_directories(Neuron/head Network/head)

set(Sources
    Neuron/src/Neuron.cpp
    Network/src/NetworkBuffer.cpp
    Network/src/NeuralNetwork.cpp
)

set(NeuronTestSources
    UnitTesting/NeuronTest.cpp
    UnitTesting/NetworkTest.cpp
)

add_library(NeuralNetwork ${Sources})

add_executable(Test ${NeuronTestSources})
target_link_libraries(Test PUBLIC gtest_main)
target_link_libraries(Test PUBLIC NeuralNetwork)

add_test(Test Test)
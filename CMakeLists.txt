cmake_minimum_required(VERSION 3.0.0)
project(NeuralNetwork VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(googletest)

include_directories(Neuron/head Network/head Timer/head)

set(TimerSources
    Timer/src/Timer.cpp
)

set(NetworkSources
    Neuron/src/Neuron.cpp
    Network/src/NetworkBuffer.cpp
    Network/src/NeuralNetwork.cpp
)

set(NeuronTestSources
    UnitTesting/NeuronTest.cpp
    UnitTesting/NetworkTest.cpp
)

#add_definitions(-DPROFILING)

add_library(Timer ${TimerSources})

add_library(NeuralNetwork ${NetworkSources})

add_executable(NeuralNetworkTest ${NeuronTestSources})
target_link_libraries(NeuralNetworkTest PUBLIC gtest_main)
target_link_libraries(NeuralNetworkTest PUBLIC NeuralNetwork)
target_link_libraries(NeuralNetworkTest PUBLIC Timer)

add_test(Test Test)
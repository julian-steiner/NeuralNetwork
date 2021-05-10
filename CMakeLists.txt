cmake_minimum_required(VERSION 3.0.0)
project(NeuralNetwork VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(googletest)

include_directories(Neuron/head Network/head Timer/head)

add_definitions(-DPROFILING)

set(TimerSources
    Timer/src/Timer.cpp
)

set(NetworkSources
    Neuron/src/Neuron.cpp
    Network/src/NetworkBuffer.cpp
    Network/src/NeuralNetwork.cpp
)

set(TestingSources
    UnitTesting/NeuronTest.cpp
    UnitTesting/NetworkTest.cpp
)

set(BenchmarkingSources
    Benchmarking/NetworkBenchmark.cpp
)

add_library(Timer ${TimerSources})

add_library(NeuralNetwork ${NetworkSources})

add_executable(NeuralNetworkTest ${TestingSources})
target_link_libraries(NeuralNetworkTest PUBLIC gtest_main)
target_link_libraries(NeuralNetworkTest PUBLIC NeuralNetwork)
target_link_libraries(NeuralNetworkTest PUBLIC Timer)

add_executable(NetworkBenchmark ${BenchmarkingSources})
target_link_libraries(NetworkBenchmark PUBLIC NeuralNetwork)
target_link_libraries(NetworkBenchmark PUBLIC Timer)

add_test(Test Test)
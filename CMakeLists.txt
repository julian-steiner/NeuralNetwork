cmake_minimum_required(VERSION 3.0.0)
project(NeuralNetwork VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(googletest)

include_directories(Neuron/head Network/head Population/head Profiler/head googletest/include)

add_definitions(-DPROFILING)

set(ProfilerSources
    Profiler/src/Profiler.cpp
    Profiler/src/Timer.cpp
)

set(NetworkSources
    Neuron/src/Neuron.cpp
    Network/src/NetworkBuffer.cpp
    Network/src/NeuralNetwork.cpp
    Network/src/Layer.cpp
    Population/src/Population.cpp
)

set(TestingSources
    UnitTesting/NeuronTest.cpp
    UnitTesting/NetworkTest.cpp
    UnitTesting/PopulationTest.cpp
)

set(BenchmarkingSources
    Benchmarking/NetworkBenchmark.cpp
)

add_library(Profiler ${ProfilerSources})

add_library(NeuralNetwork ${NetworkSources})

add_executable(NeuralNetworkTest ${TestingSources})
target_link_libraries(NeuralNetworkTest PUBLIC gtest_main)
target_link_libraries(NeuralNetworkTest PUBLIC NeuralNetwork)
target_link_libraries(NeuralNetworkTest PUBLIC Profiler)

add_executable(PopulationDemo ${BenchmarkingSources})
target_link_libraries(PopulationDemo PUBLIC NeuralNetwork)
target_link_libraries(PopulationDemo PUBLIC Profiler)

add_test(Test Test)
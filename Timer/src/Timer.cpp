#include "Timer.h"

using namespace timer;

Timer::Timer(std::string name)
{
    this->name = name;
    this->startTime = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
    endTime = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(startTime).time_since_epoch().count();
    auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endTime).time_since_epoch().count();

    auto duration = end - start;
    //std::cout << "Time needed for " << this->name << ":  " << (double)duration.count()/1000000 << "ms" << std::endl; 
    std::cout << "Time needed for " << this->name << ":  " << duration << "ns" << std::endl; 
}
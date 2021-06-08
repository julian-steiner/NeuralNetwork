#include "Timer.h"

using namespace timing;

ScopeTimer::ScopeTimer(const char* name) 
{
    this->start_time = std::chrono::high_resolution_clock::now();
    this->name = name;
}

ScopeTimer::~ScopeTimer()
{
    this->end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << this->name << " took " << duration.count() / 1000000.0 << " ms" << std::endl;
}
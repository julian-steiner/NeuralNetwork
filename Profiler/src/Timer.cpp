#include "Timer.h"

using namespace profiling;

Scope_Timer::Scope_Timer(const char* name) 
{
    this->start_time = std::chrono::high_resolution_clock::now();
    this->name = name;
}

Scope_Timer::~Scope_Timer()
{
    this->end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << this->name << " took " << duration.count() / 1000000.0 << " ms" << std::endl;
}
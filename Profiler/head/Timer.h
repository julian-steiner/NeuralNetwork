#ifndef TIMER
#define TIMER

#include <chrono>
#include <iostream>

#define TIME_SCOPE(name) profiling::Scope_Timer(name);

namespace profiling
{
    class Scope_Timer
    {
        private:
        const char* name;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
        std::chrono::time_point<std::chrono::high_resolution_clock> end_time;

        public:
        Scope_Timer(const char* name);
        ~Scope_Timer();
    };
}

#endif
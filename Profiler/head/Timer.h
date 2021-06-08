#ifndef TIMER
#define TIMER

#include <chrono>
#include <iostream>

#define TIME_SCOPE(name) profiling::Scope_Timer(name);

namespace timing
{
    class ScopeTimer
    {
        private:
        const char* name;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
        std::chrono::time_point<std::chrono::high_resolution_clock> end_time;

        public:
        ScopeTimer(const char* name);
        ~ScopeTimer();
    };
}

#endif
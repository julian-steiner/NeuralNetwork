#include <chrono>
#include <iostream>
#include <string.h>

namespace timer 
{
    class Timer
    {
        private:
        std::string name;
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
        std::chrono::time_point<std::chrono::high_resolution_clock> endTime;

        public:
        Timer(std::string name);
        ~Timer();
    };
}
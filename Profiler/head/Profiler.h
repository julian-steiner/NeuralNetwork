#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <chrono> 

#if PROFILING
#define PROFILE_SCOPE(name) profiling::InstrumentationTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__function__)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#endif


namespace profiling 
{
    struct ProfileResult
    {
        std::string Name;
        long long Start, End;
    };
    
    struct InstrumentationSession
    {
        std::string Name;
    };  

    class Instrumentor
    {
        private:
        InstrumentationSession* m_currentSession;
        std::ofstream m_OutputStream;
        int m_ProfileCount;

        public:
        Instrumentor();
        void BeginSession(const std::string& name, const std::string& filepath = "results.json");
        void EndSession();
        void WriteProfile(const ProfileResult& result);
        void WriteHeader();
        void WriteFooter();
        static Instrumentor& Get()
        {
            static Instrumentor* instance = new Instrumentor();
            return *instance;
        };
    };

    class InstrumentationTimer
    {
        private:
        const char* m_Name;
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
        std::chrono::time_point<std::chrono::high_resolution_clock> endTime;

        public:
        InstrumentationTimer(const char* name);
        ~InstrumentationTimer();
    };
}
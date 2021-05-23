#include "Profiler.h"

using namespace profiling;

InstrumentationTimer::InstrumentationTimer(const char* name)
{
    m_Name = name;
    startTime = std::chrono::high_resolution_clock::now();
}

InstrumentationTimer::~InstrumentationTimer()
{
    endTime = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startTime).time_since_epoch().count();
    auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTime).time_since_epoch().count();

    Instrumentor::Get().WriteProfile({m_Name, start, end});
}

Instrumentor::Instrumentor() : m_currentSession(nullptr), m_ProfileCount(0)
{
}

void Instrumentor::BeginSession(const std::string& name, const std::string& filepath)
{
    m_OutputStream.open(filepath);
    WriteHeader();
    m_currentSession = new InstrumentationSession { name };
}

void Instrumentor::EndSession()
{
    WriteFooter();
    m_OutputStream.close();
    delete m_currentSession;
    m_currentSession = nullptr;
    m_ProfileCount = 0;
}

void Instrumentor::WriteProfile(const ProfileResult& result)
{
    if (m_ProfileCount++ > 0)
    {
        m_OutputStream << ",";
    }
    
    std::string name = result.Name;
    std::replace(name.begin(), name.end(), '"', '\'');

    m_OutputStream << "{";
    m_OutputStream << "\"cat\":\"function\",";
    m_OutputStream << "\"dur\":" << (result.End - result.Start) << ',';
    m_OutputStream << "\"name\":\"" << name << "\",";
    m_OutputStream << "\"ph\":\"X\",";
    m_OutputStream << "\"pid\":0,";
    m_OutputStream << "\"tid\":0,";
    m_OutputStream << "\"ts\":" << result.Start;
    m_OutputStream << "}";

    m_OutputStream.flush();
}

void Instrumentor::WriteHeader()
{
    m_OutputStream << "{\"otherData\": {},\"traceEvents\":[";
    m_OutputStream.flush();
}

void Instrumentor::WriteFooter()
{
    m_OutputStream << "]}";
    m_OutputStream.flush();
}
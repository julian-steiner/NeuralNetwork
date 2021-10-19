#pragma once

#include "NeuralNetwork.h"

namespace logging
{

class Logger
{
private:
    std::ofstream document;
    const std::string& filename;
    int networksCounter;
public:
    Logger(const std::string& filename);
    ~Logger();

    void addNetwork(nn::NeuralNetwork& network, const int& generation);
};

}
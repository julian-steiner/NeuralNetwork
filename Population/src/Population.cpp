#include "Population.h"

using namespace population;

Population::Population(const int& size, nn::NeuralNetwork* templateNetwork)
{
    this->networks.reserve(size);
    for (int i = 0; i < size; i++)
    {
        this->networks.push_back(std::move(templateNetwork->getCopy<nn::NeuralNetwork>()));
    }
}

Population::getNetwork(int number)
{
    return this->networks.at(number);
}
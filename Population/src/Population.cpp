#include "Population.h"

using namespace population;

Population::Population(const int& size, const nn::NeuralNetwork& templateNetwork)
{
    Population::networks.reserve(size);

    for (int i = 0; i < size; i++)
    {
        Population::networks.push_back(templateNetwork.getCopy());
    }
}
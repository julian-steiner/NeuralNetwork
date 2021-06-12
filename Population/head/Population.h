#ifndef POPULATION
#define POPULATION

#include "NeuralNetwork.h"

namespace population
{
    class Population
    {
        public:
        Population(const int& size, const nn::NeuralNetwork& templateNetwork);

        private:
        std::vector<nn::NeuralNetwork> networks;
        std::vector<connection::ConnectionDummy> connectionDatabase;

    };
}

#endif
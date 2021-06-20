#ifndef POPULATION
#define POPULATION

#include "NeuralNetwork.h"

namespace population
{
    class Population
    {
        public:
        Population(const int& size, nn::NeuralNetwork* templateNetwork);

        nn::NeuralNetwork* getNetwork(int number);

        private:
        std::vector<nn::NeuralNetwork> networks;
        std::vector<connection::ConnectionDummy> connectionDatabase;

    };
}

#endif
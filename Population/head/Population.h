#ifndef POPULATION
#define POPULATION

#include "NeuralNetwork.h"

namespace population
{
    class Population
    {
        public:
        double mutationRate = 1;
        double structuralMutationRate = 1;

        Population(const int& size, nn::NeuralNetwork* templateNetwork);

        nn::NeuralNetwork* getNetwork(int number);
        int getCurrentInnovationNumber();

        void mutate();

        private:
        void weightMutate(connection::Connection* target);
        void addConnection(nn::NeuralNetwork* targetNetwork, connection::NeuronLocation neuron1, connection::NeuronLocation neuron2);
        void addNeuron(nn::NeuralNetwork* targetNetwork, connection::Connection* target);
        std::vector<nn::NeuralNetwork> networks;
        int currentInnovationNumber;
    };
}

#endif
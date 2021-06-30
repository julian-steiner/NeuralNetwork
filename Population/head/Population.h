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
        double learningRate = 1;

        Population(const int& size, nn::NeuralNetwork* templateNetwork);
        ~Population();

        nn::NeuralNetwork* getNetwork(int number);
        int getCurrentInnovationNumber();

        void crossover();
        void mutate();

        private:
        nn::NeuralNetwork getChild(nn::NeuralNetwork* first, nn::NeuralNetwork* second);
        int getSize();
        double getTotalFitness();
        void weightMutate(connection::Connection* target);
        void addConnection(nn::NeuralNetwork* targetNetwork, connection::NeuronLocation neuron1, connection::NeuronLocation neuron2);
        void addNeuron(nn::NeuralNetwork* targetNetwork, connection::Connection* target);
        std::vector<nn::NeuralNetwork>* networks;
        int currentInnovationNumber;
    };
}

#endif
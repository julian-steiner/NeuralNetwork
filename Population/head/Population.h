#ifndef POPULATION
#define POPULATION

#include "NeuralNetwork.h"

namespace population
{
    struct NetworkComparison
    {
        int matchingGenes;
        int nonMatchingGenes;
        double weightDifferences;
        double differenceRatio;
    };

    class Population
    {
        public:
        double weightChangingRate = 0;
        double neuronAddingRate = 0;
        double connectionAddingRate = 0;
        double learningRate = 1;
        double nonMatchingGenesWeight = 1;
        double weightDifferenceWeight = 0.5;
        double numberOfSpecies = 1;

        Population(const int& size, nn::NeuralNetwork* templateNetwork);
        ~Population();

        nn::NeuralNetwork* getNetwork(int number);
        int getCurrentInnovationNumber();

        void mutate();
        population::NetworkComparison compareNetworks(nn::NeuralNetwork* first, nn::NeuralNetwork* second);
        void speciate();
        void crossover();

        private:
        nn::NeuralNetwork getChild(nn::NeuralNetwork* first, nn::NeuralNetwork* second);
        int getSize();
        double getTotalFitness();
        double getMaxDifference(nn::NeuralNetwork* reference);
        void weightMutate(connection::Connection* target);
        void biasMutate(neuron::Neuron* target);
        void addConnection(nn::NeuralNetwork* targetNetwork, connection::NeuronLocation neuron1, connection::NeuronLocation neuron2);
        void addNeuron(nn::NeuralNetwork* targetNetwork, connection::Connection* target);
        std::vector<nn::NeuralNetwork>* networks;
        std::vector<std::vector<nn::NeuralNetwork*>> species;
        int currentInnovationNumber;
    };
}

#endif
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
        nn::NeuralNetwork& getNetwork(const int& number);
        const int& getCurrentInnovationNumber();

        double compareNetworks(nn::NeuralNetwork* network1, nn::NeuralNetwork* network2);

        private:
        std::vector<nn::NeuralNetwork>* networks;
        int currentInnovationNumber;
    };
}

#endif
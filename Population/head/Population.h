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

    struct Species
    {
        std::vector<nn::NeuralNetwork*> networks;
        double totalFitness;
        int numChildrenAllowed;
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
        
        double targetNumberOfSpecies = 1;
        double speciationThreshold;

        Population(const int& size, nn::NeuralNetwork* templateNetwork);
        ~Population();
        nn::NeuralNetwork& getNetwork(const int& number);
        const int& getCurrentInnovationNumber();

        int getNumberOfSpecies();

        double compareNetworks(nn::NeuralNetwork* network1, nn::NeuralNetwork* network2);
        void speciate();

        private:
        std::vector<nn::NeuralNetwork>* networks;
        std::vector<Species> species;

        int currentInnovationNumber;
    };
}

#endif
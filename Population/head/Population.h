#ifndef POPULATION
#define POPULATION

#include "NeuralNetwork.h"
#include "Neuron.h"

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
        double weightChangingRate = 0.8;
        double neuronAddingRate = 0;
        double connectionAddingRate = 0;
        double learningRate = 0.2;

        double nonMatchingGenesWeight = 1;
        double weightDifferenceWeight = 0.5;
        
        double targetNumberOfOrganisms; 
        double targetNumberOfSpecies = 1;
        double speciationThreshold;

        std::vector<nn::NeuralNetwork>* networks;

        Population(const int& size, nn::NeuralNetwork* templateNetwork);
        ~Population();
        nn::NeuralNetwork& getNetwork(const int& number);
        const int& getCurrentInnovationNumber();

        int getNumberOfSpecies();

        double compareNetworks(nn::NeuralNetwork* network1, nn::NeuralNetwork* network2);
        void speciate();
        void mutate();
        void crossover();

        nn::NeuralNetwork* getFittest();

        private:
        void assignInnovationNumber(const connection::ConnectionDummy& dummy, bool& found, int& innovationNumber);
        void addConnection(const connection::ConnectionDummy& dummy, nn::NeuralNetwork& currentNetwork, const bool& found, const int& innovationNumber);
        void findMatchingConnection(std::vector<connection::Connection*>* connections, const int& innovationNumber, bool& found, double& weight);
        void computeChildrenAllowed();
        nn::NeuralNetwork getChild(nn::NeuralNetwork* first, nn::NeuralNetwork* second);
        std::vector<Species> species;

        int currentInnovationNumber;

        std::vector<connection::ConnectionDummy> connectionDatabase;
    };
}

#endif
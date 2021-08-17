#ifndef POPULATION
#define POPULATION

#include "NeuralNetwork.h"
#include <random>

namespace population
{
    class RandomGenerator
    {
    private:
        unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine engine = std::default_random_engine(seed);
        std::uniform_real_distribution<double> distr = std::uniform_real_distribution<double>(0.0, 1.0);

    public:
        double getRandomNumber()
        {
            return distr(engine);
        }

    };

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
        double targetNumberOfOrganisms = 0;
        double speciationThreshold = 0;

        std::vector<connection::ConnectionDummy> connectionDatabase;

        Population(const int& size, nn::NeuralNetwork* templateNetwork);
        ~Population();

        nn::NeuralNetwork* getNetwork(int number);
        int getCurrentInnovationNumber();

        void mutate();
        population::NetworkComparison compareNetworks(nn::NeuralNetwork* first, nn::NeuralNetwork* second);
        void speciate();
        void crossover();

        int getNumberOfSpecies();

        private:
        RandomGenerator randomGenerator;

        nn::NeuralNetwork getChild(nn::NeuralNetwork* first, nn::NeuralNetwork* second);
        int getSize();
        double getTotalFitness();
        double getMaxDifference(nn::NeuralNetwork* reference);
        void weightMutate(connection::Connection* target);
        void biasMutate(neuron::Neuron* target);
        void addConnection(nn::NeuralNetwork* targetNetwork, connection::NeuronLocation neuron1, connection::NeuronLocation neuron2);
        void addNeuron(nn::NeuralNetwork* targetNetwork, connection::Connection* target);
        void computeChildrenAllowed();
        std::vector<nn::NeuralNetwork>* networks;
        std::vector<Species> species;
        int currentInnovationNumber;

        void assignInnovationNumber(const connection::ConnectionDummy& dummy, int& innovationNumber);
    };
}

#endif
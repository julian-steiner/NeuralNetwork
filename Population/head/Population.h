#ifndef POPULATION
#define POPULATION

#include "NeuralNetwork.h"

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
        // Changing dials for mutation
        double weightChangingRate = 0;
        double neuronAddingRate = 0;
        double connectionAddingRate = 0;
        double learningRate = 1;

        // Weights for network speciation
        double nonMatchingGenesWeight = 1;
        double weightDifferenceWeight = 0.5;
        double targetNumberOfSpecies = 1;
        double targetNumberOfOrganisms = 0;
        double speciationThreshold = 0;

        // Constructor
        Population(const int& size, nn::NeuralNetwork* templateNetwork);
        ~Population();

        // Getters
        int getSize();
        int getNumberOfSpecies();
        nn::NeuralNetwork* getNetwork(int number);

        // Core Methods
        void mutate();
        void speciate();
        void crossover();

        private:
        RandomGenerator randomGenerator;
        std::vector<connection::ConnectionDummy> connectionDatabase;
        std::vector<nn::NeuralNetwork>* networks;
        std::vector<Species> species;


        // Methods for mutation
        connection::NeuronLocation generateRandomNeuronLocation(nn::NeuralNetwork* currentNeuralNetwork);
        bool validateConnection(nn::NeuralNetwork* currentNeuralNetwork, const connection::ConnectionDummy& connectionDummy);

        void handleNumberMutations(nn::NeuralNetwork* currentNeuralNetwork);
        void handleWeightMutations(nn::NeuralNetwork* currentNeuralNetwork);
        void handleBiasMutations(nn::NeuralNetwork* currentNeuralNetwork);

        void handleConnectionMutations(nn::NeuralNetwork* currentNeuralNetwork);
        void handleNeuronMutations(nn::NeuralNetwork* currentNeuralNetwork);
        void handleStructuralMutations(nn::NeuralNetwork* currentNeuralNetwork);

        void weightMutate(connection::Connection* target);
        void biasMutate(neuron::Neuron* target);
        void addConnection(nn::NeuralNetwork* targetNetwork, connection::NeuronLocation neuron1, connection::NeuronLocation neuron2);
        void addNeuron(nn::NeuralNetwork* targetNetwork, connection::Connection* target);

        // Methods relevant for speciation
        population::NetworkComparison compareNetworks(nn::NeuralNetwork* first, nn::NeuralNetwork* second);

        // Methods for computing children
        double getTotalFitness();
        double getMaxDifference(nn::NeuralNetwork* reference);
        void computeChildrenAllowed();

        // Methods for crossover
        nn::NeuralNetwork getChild(nn::NeuralNetwork* first, nn::NeuralNetwork* second);

        // Innovation Number management
        void assignInnovationNumber(const connection::ConnectionDummy& dummy, int& innovationNumber);
    };
}

#endif
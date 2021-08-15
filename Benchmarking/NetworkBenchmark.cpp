#include "NeuralNetwork.h"
#include "Population.h"

int main()
{
    nn::NeuralNetwork testNetwork;
    int numberOfGenerations = 100;

    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    //testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    population::Population testPopulation(1000, &testNetwork);

    testPopulation.connectionAddingRate = 0.3;
    testPopulation.neuronAddingRate = 0.1;
    testPopulation.weightChangingRate = 1;
    testPopulation.targetNumberOfSpecies = 4;
    testPopulation.speciationThreshold = 2;
    testPopulation.learningRate = 0.5;

    for (int i = 0; i < numberOfGenerations; i++)
    {
        for (int a = 0; a < testPopulation.networks->size(); a++)
        {
            nn::NeuralNetwork& currentNetwork = testPopulation.networks->at(a);

            currentNetwork.fitness = 4;
            currentNetwork.fitness -= currentNetwork.predict({0, 0}).back();
            currentNetwork.fitness -= currentNetwork.predict({1, 1}).back();
            //currentNetwork.fitness -= 1 - currentNetwork.predict({0, 1}).back();
            //currentNetwork.fitness -= 1 - currentNetwork.predict({1, 0}).back();
        }

        std::cout << "-------------------------Generation: " << i+1 << "---------------------------" <<std::endl;
        std::cout << "Fittest: " << testPopulation.getFittest()->fitness << std::endl;
        std::cout << "SpeciationThreshold: " << testPopulation.speciationThreshold << std::endl;
        std::cout << "Number Of Species: " << testPopulation.getNumberOfSpecies() << std::endl;
        std::cout << "Population Size: " << testPopulation.networks->size() << std::endl;

        std::cout << std::endl << std::endl;

        if (i < numberOfGenerations-1)
        {
            testPopulation.speciate();
            testPopulation.crossover();
            testPopulation.mutate();
        }
    }

    testPopulation.getFittest()->saveConnectionScheme("Schemes/NetworkBefore.tex");
}
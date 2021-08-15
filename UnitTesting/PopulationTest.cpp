#include <gtest/gtest.h>
#include "Population.h"
#include <time.h>

TEST(Population, PopulationCreatedCorrectly)
{
    nn::NeuralNetwork testNetwork;

    testNetwork.addLayer(4, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    testNetwork.addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1, 69);
    testNetwork.connect({0, 0}, {1, 0}, 0);
    testNetwork.connect({1, 0}, {2, 1}, 420);

    population::Population testPopulation(10, &testNetwork);

    ASSERT_EQ(testPopulation.getNetwork(8).layers->at(1)->neurons.at(0)->bias, 69);
    ASSERT_EQ(testPopulation.getNetwork(8).layers->at(1)->neurons.at(0)->connections_forward.at(0)->innovationNumber, 420);
    ASSERT_EQ(testPopulation.getCurrentInnovationNumber(), 420);
}

TEST(Population, NetworksComparedCorrectly)
{
    nn::NeuralNetwork testNetwork;

    testNetwork.addLayer(4, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    population::Population testPopulation(2, &testNetwork);

    double difference = testPopulation.compareNetworks(&testPopulation.getNetwork(0), &testPopulation.getNetwork(1));

    ASSERT_EQ(difference, 0);
    
    testPopulation.getNetwork(0).addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1, 69);
    testPopulation.getNetwork(0).connect({0, 0}, {1, 0}, 0);
    testPopulation.getNetwork(0).connections->at(0)->weight = 0;

    testPopulation.getNetwork(1).addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1, 69);
    testPopulation.getNetwork(1).connect({0, 0}, {1, 0}, 0);
    testPopulation.getNetwork(1).connect({1, 0}, {2, 1}, 420);
    testPopulation.getNetwork(1).connections->at(0)->weight = 320;
    testPopulation.getNetwork(1).connections->at(1)->weight = 320;

    difference = testPopulation.compareNetworks(&testPopulation.getNetwork(0), &testPopulation.getNetwork(1));

    ASSERT_EQ(difference, 321);
}

TEST(Population, SpeciatingCorrectly)
{
    nn::NeuralNetwork testNetwork;

    testNetwork.addLayer(4, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    population::Population testPopulation(10, &testNetwork);
    testPopulation.targetNumberOfSpecies = 3;
    testPopulation.speciationThreshold = 0;

    testPopulation.getNetwork(1).addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1, 69);
    testPopulation.getNetwork(1).connect({0, 0}, {1, 0}, 0);
    testPopulation.getNetwork(1).connect({1, 0}, {2, 1}, 420);
    testPopulation.getNetwork(1).connections->at(0)->weight = 320;
    testPopulation.getNetwork(1).connections->at(1)->weight = 320;

    testPopulation.getNetwork(2).addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1, 69);
    testPopulation.getNetwork(2).connect({0, 0}, {1, 0}, 0);
    testPopulation.getNetwork(2).connections->at(0)->weight = 320;

    testPopulation.speciate();

    ASSERT_EQ(testPopulation.getNumberOfSpecies(), 3);

    testPopulation.speciationThreshold = 2;

    testPopulation.speciate();

    ASSERT_EQ(testPopulation.getNumberOfSpecies(), 1);
}

TEST(Population, MutationWorking)
{
    nn::NeuralNetwork testNetwork;
    testNetwork.addLayer(1, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    testNetwork.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    population::Population testPopulation(1, &testNetwork);
    testPopulation.connectionAddingRate = 1;
    testPopulation.neuronAddingRate = 0;

    testPopulation.mutate();

    testPopulation.getNetwork(0).saveConnectionScheme("Schemes/NetworkBefore.tex");

    ASSERT_EQ(testPopulation.getNetwork(0).connections->size(), 1);
    ASSERT_EQ(testPopulation.getNetwork(0).neurons->size(), 2);

    testPopulation.connectionAddingRate = 0;
    testPopulation.neuronAddingRate = 1;

    testPopulation.mutate();

    testPopulation.getNetwork(0).saveConnectionScheme("Schemes/NetworkAfter.tex");

    ASSERT_EQ(testPopulation.getNetwork(0).connections->size(), 3);
    ASSERT_EQ(testPopulation.getNetwork(0).neurons->size(), 3);
    ASSERT_EQ(testPopulation.getNetwork(0).connections->at(1)->innovationNumber, 1);
    ASSERT_EQ(testPopulation.getNetwork(0).connections->at(2)->innovationNumber, 2);
}

TEST(Population, CrossoverWorking)
{
    nn::NeuralNetwork testNetwork;
    testNetwork.addLayer(2, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    testNetwork.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    population::Population testPopulation(5, &testNetwork);
    testPopulation.connectionAddingRate = 1;
    testPopulation.mutate();

    testPopulation.speciationThreshold = 3;
    testPopulation.speciate();
    testPopulation.crossover();
    testPopulation.mutate();

    std::vector<nn::NeuralNetwork*> networks = std::vector<nn::NeuralNetwork*>();
    for (int i = 0; i < testPopulation.networks->size(); i++)
    {
        nn::NeuralNetwork* currentNetwork = &testPopulation.getNetwork(i);

        std::vector<std::string> connectionScheme = currentNetwork->getConnectionScheme();
        for (const std::string& string : connectionScheme)
        {
            std::cout << string << std::endl;
        }

        networks.push_back(currentNetwork);
    }

    std::cout << "Hello World" << std::endl;
}
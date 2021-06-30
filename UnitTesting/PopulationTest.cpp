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

    ASSERT_EQ(testPopulation.getNetwork(8)->layers->at(1)->neurons.at(0)->bias, 69);
    ASSERT_EQ(testPopulation.getNetwork(8)->layers->at(1)->neurons.at(0)->connections_forward.at(0)->innovationNumber, 420);
    ASSERT_EQ(testPopulation.getCurrentInnovationNumber(), 420);
}

TEST(Population, PopulationAddingConnectionCorrectly)
{
    nn::NeuralNetwork testNetwork;

    std::srand(time(NULL));

    testNetwork.addLayer(1, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::None, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    testNetwork.addLayer(1, neuron::Activation::None, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    population::Population testPopulation(1, &testNetwork);
    testPopulation.mutationRate = 0;
    testPopulation.structuralMutationRate = 1;

    ASSERT_EQ(testPopulation.getNetwork(0)->connections->size(), 0);

    testPopulation.mutate();

    ASSERT_EQ(testPopulation.getNetwork(0)->connections->size(), 1);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(0)->inNeuronLocation.layer, 0);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(0)->inNeuronLocation.number, 0);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(0)->outNeuronLocation.layer, 2);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(0)->outNeuronLocation.number, 0);

    testPopulation.mutate();

    ASSERT_TRUE(testPopulation.getNetwork(0)->connections->size() == 3 || testPopulation.getNetwork(0)->connections->size() == 5);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(0)->enabled, false);
    ASSERT_TRUE(testPopulation.getNetwork(0)->layers->at(1)->getSize() == 1 || testPopulation.getNetwork(0)->layers->at(1)->getSize() == 2);

    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(1)->inNeuronLocation.layer, 0);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(1)->inNeuronLocation.number, 0);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(1)->outNeuronLocation.layer, 1);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(1)->outNeuronLocation.number, 0);

    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(2)->inNeuronLocation.layer, 1);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(2)->inNeuronLocation.number, 0);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(2)->outNeuronLocation.layer, 2);
    ASSERT_EQ(testPopulation.getNetwork(0)->connections->at(2)->outNeuronLocation.number, 0);
}

TEST(Population, CrossingOverworkingCorrectly)
{
    nn::NeuralNetwork testNetwork;
    testNetwork.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    testNetwork.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    population::Population testPopulation(2, &testNetwork);

    testPopulation.getNetwork(0)->addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1);
    testPopulation.getNetwork(0)->connect({0, 0}, {1, 0}, 1);
    testPopulation.getNetwork(0)->connect({1, 0}, {2, 0}, 2);
    testPopulation.getNetwork(0)->fitness = 2;
    testPopulation.getNetwork(0)->connections->at(0)->weight = 420;
    testPopulation.getNetwork(0)->connections->at(1)->weight = 69;
    testPopulation.getNetwork(1)->fitness = 1;

    testPopulation.crossover();

    // Figure out why there are 4 connection dummys per network instead of 2 

    ASSERT_TRUE(testPopulation.getNetwork(1)->connections->size() == 2 || testPopulation.getNetwork(1)->connections->size() == 0);
    if (testPopulation.getNetwork(1)->connections->size() == 2)
    {
        ASSERT_EQ(testPopulation.getNetwork(1)->connections->at(1)->weight, 69);
    }
}
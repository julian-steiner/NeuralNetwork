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
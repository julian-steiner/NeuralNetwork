#include <gtest/gtest.h>
#include "Population.h"

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
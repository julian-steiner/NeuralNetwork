#include <gtest/gtest.h>
#include "NetworkBuffer.h"
#include "NeuralNetwork.h"
#include <cstdlib>

TEST(NetworkBuffer, BufferInitializedCorrectly)
{
    nn::NetworkBuffer testNetwork;
    ASSERT_EQ(testNetwork.connections.size(), 0);
    ASSERT_EQ(testNetwork.neurons.size(), 0);
}

TEST(NetworkBuffer, NeuronCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addNeuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid);

    ASSERT_EQ(testNetwork.neurons.size(), 1);
}

TEST(NetworkBuffer, ConnectionCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;
    neuron::Neuron testNeuron = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid);
    neuron::Neuron testNeuron2 = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid);

    testNetwork.addNeuron(std::move(testNeuron));
    testNetwork.addNeuron(std::move(testNeuron2));
    testNetwork.addConnection(&testNeuron, &testNeuron2);

    ASSERT_EQ(testNetwork.neurons.size(), 2);
    ASSERT_EQ(testNetwork.connections.size(), 1);
}

TEST(NetworkBuffer, ConnectingWorkingCorrectly)
{
    nn::NetworkBuffer testNetwork;
    testNetwork.addNeuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid);
    testNetwork.addNeuron(neuron::NeuronType::Input, neuron::Activation::None);
    
    testNetwork.connect(1, 2);

    ASSERT_EQ(testNetwork.neurons.size(), 2);
    ASSERT_EQ(testNetwork.connections.size(), 1);
    ASSERT_EQ(testNetwork.connections.at(0).in->activation, neuron::Activation::Sigmoid);
    ASSERT_EQ(testNetwork.connections.at(0).out->activation, neuron::Activation::None);
}

TEST(NetworkBuffer, LayersCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);

    ASSERT_EQ(testNetwork.inputLayerSize, 10);
    ASSERT_EQ(testNetwork.neurons.size(), 32);
    ASSERT_EQ(testNetwork.previousLayerSize, 2);
    ASSERT_EQ(testNetwork.neurons.at(5).type, neuron::NeuronType::Input);
    ASSERT_EQ(testNetwork.neurons.at(15).type, neuron::NeuronType::Hidden);
}

TEST(NetworkBuffer, ConnectionsCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);

    ASSERT_EQ(testNetwork.neurons.at(0).connections_forward.size(), 20);
    ASSERT_EQ(testNetwork.neurons.at(15).connections_back.size(), 10);
    ASSERT_EQ(testNetwork.neurons.at(31).connections_forward.size(), 0);
    ASSERT_EQ(testNetwork.neurons.at(31).connections_back.size(), 20);
}

TEST(NeuralNetwork, NetworkInheritanceWorkingCorrectly)
{
    nn::NeuralNetwork testNetwork; 
    testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);

    ASSERT_EQ(testNetwork.neurons.at(0).connections_forward.size(), 20);
    ASSERT_EQ(testNetwork.neurons.at(15).connections_back.size(), 10);
    ASSERT_EQ(testNetwork.neurons.at(31).connections_forward.size(), 0);
    ASSERT_EQ(testNetwork.neurons.at(31).connections_back.size(), 20);
}

TEST(NeuralNetwork, FeedforwardWorkingCorrectly)
{
    nn::NeuralNetwork testNetwork;
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Binary, nn::LayerType::FullyConnected);
    testNetwork.addLayer(1, neuron::Activation::Binary, nn::LayerType::FullyConnected);

    testNetwork.neurons.at(2).bias = 0.5;
    testNetwork.neurons.at(3).bias = -1.5;
    testNetwork.neurons.at(4).bias = 1.5;
    
    testNetwork.connections.at(0).weight = 1;
    testNetwork.connections.at(1).weight = 1;
    testNetwork.connections.at(2).weight = -1;
    testNetwork.connections.at(3).weight = -1;
    testNetwork.connections.at(4).weight = 1;
    testNetwork.connections.at(5).weight = 1;

    ASSERT_EQ(testNetwork.predict({0, 0}).at(0), 0);
    ASSERT_EQ(testNetwork.predict({0, 1}).at(0), 1);
}
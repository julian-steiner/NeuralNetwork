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
    std::shared_ptr<neuron::Neuron> testNeuronPointer = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid));

    testNetwork.addNeuron(testNeuronPointer);

    ASSERT_EQ(testNetwork.neurons.size(), 1);
    ASSERT_EQ(testNetwork.neurons.at(0), testNeuronPointer);
}

TEST(NetworkBuffer, ConnectionCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;
    std::shared_ptr<neuron::Neuron> testNeuronPointer = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid));
    std::shared_ptr<neuron::Neuron> testNeuronPointer2 = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid));

    testNetwork.addNeuron(testNeuronPointer);
    testNetwork.addNeuron(testNeuronPointer2);
    testNetwork.addConnection(testNeuronPointer, testNeuronPointer2);

    ASSERT_EQ(testNetwork.neurons.size(), 2);
    ASSERT_EQ(testNetwork.neurons.at(0), testNeuronPointer);
    ASSERT_EQ(testNetwork.connections.size(), 1);
    ASSERT_EQ(testNetwork.connections.at(0)->in, testNeuronPointer);
    ASSERT_EQ(testNetwork.connections.at(0)->out, testNeuronPointer2);
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
    ASSERT_EQ(testNetwork.neurons.at(5)->type, neuron::NeuronType::Input);
    ASSERT_EQ(testNetwork.neurons.at(15)->type, neuron::NeuronType::Hidden);
}

TEST(NetworkBuffer, ConnectionsCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);

    testNetwork.connectNetwork();

    ASSERT_EQ(testNetwork.neurons.at(0)->connections_forward.size(), 20);
    ASSERT_EQ(testNetwork.neurons.at(15)->connections_back.size(), 10);
    ASSERT_EQ(testNetwork.neurons.at(31)->connections_forward.size(), 0);
    ASSERT_EQ(testNetwork.neurons.at(31)->connections_back.size(), 20);
}

TEST(NeuralNetwork, NetworkInheritanceWorkingCorrectly)
{
    nn::NeuralNetwork testNetwork; 
    testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);

    testNetwork.connectNetwork();

    ASSERT_EQ(testNetwork.neurons.at(0)->connections_forward.size(), 20);
    ASSERT_EQ(testNetwork.neurons.at(15)->connections_back.size(), 10);
    ASSERT_EQ(testNetwork.neurons.at(31)->connections_forward.size(), 0);
    ASSERT_EQ(testNetwork.neurons.at(31)->connections_back.size(), 20);
}
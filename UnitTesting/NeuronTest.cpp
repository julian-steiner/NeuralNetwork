#include <gtest/gtest.h>
#include "Neuron.h"

TEST(Neuron, NeuronInitializedCorrectly)
{
    neuron::Neuron testNeuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid);
    ASSERT_EQ(testNeuron.activation, neuron::Activation::Sigmoid);
    ASSERT_EQ(testNeuron.value, 0);
    ASSERT_EQ(testNeuron.type, neuron::NeuronType::Input);
    ASSERT_EQ(testNeuron.bias, 0);
}

TEST(Connection, ConnectionInitializedCorrectly)
{
    neuron::Neuron testNeuron(neuron::NeuronType::Output, neuron::Activation::Sigmoid);
    testNeuron.value = 54;
    ASSERT_EQ(testNeuron.value, 54);
    connection::Connection testConnection(std::make_shared<neuron::Neuron>(testNeuron), nullptr);
    
    ASSERT_NE(testConnection.in, nullptr);
    ASSERT_EQ(testConnection.in->value, 54);
    ASSERT_EQ(testConnection.out, nullptr);
}
#include <gtest/gtest.h>
#include "Neuron.h"
#include <iostream>

TEST(Neuron, NeuronInitializedCorrectly)
{
    std::shared_ptr<neuron::Neuron> testNeuron = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid));
    ASSERT_EQ(testNeuron->activation, neuron::Activation::Sigmoid);
    ASSERT_EQ(testNeuron->value, 0);
    ASSERT_EQ(testNeuron->type, neuron::NeuronType::Input);
    ASSERT_EQ(testNeuron->bias, 0);
}

TEST(Neuron, InputNeuronComputingCorrectly)
{
    std::shared_ptr<neuron::Neuron> testNeuron = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::None));
    testNeuron->value = 53;
    testNeuron->bias = 23103910;

    double result = testNeuron->calculate();

    ASSERT_EQ(result, 53);
}

TEST(Neuron, ComputingCorrectly)
{
    std::shared_ptr<neuron::Neuron> testNeuron = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid));
    testNeuron->value = 54;

    std::shared_ptr<neuron::Neuron> testNeuron2 = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid));
    testNeuron2->value = 100;

    bool cacheValue = NULL;
    std::shared_ptr<neuron::Neuron> testNeuron3 = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, &cacheValue));
    testNeuron3->bias = 1;

    testNeuron3->connections_back.push_back(std::make_shared<connection::Connection>(connection::Connection(testNeuron, testNeuron3)));
    testNeuron3->connections_back.push_back(std::make_shared<connection::Connection>(connection::Connection(testNeuron2, testNeuron3)));

    double result = testNeuron3->calculate();
    std::cout << testNeuron3->weightedSumCache << std::endl;
    std::cout << result << std::endl;

    ASSERT_EQ(true, false);
}

TEST(Connection, ConnectionInitializedCorrectly)
{
    std::shared_ptr<neuron::Neuron> testNeuron = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Output, neuron::Activation::Sigmoid));
    testNeuron->value = 54;

    std::shared_ptr<neuron::Neuron> testNeuron2 = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Output, neuron::Activation::Sigmoid));

    ASSERT_EQ(testNeuron->value, 54);
    connection::Connection testConnection(testNeuron, testNeuron2);
    
    ASSERT_EQ(testConnection.in, testNeuron);
    ASSERT_EQ(testConnection.out, testNeuron2);
    
    testNeuron->value = 10;
    
    ASSERT_EQ(testConnection.in->value, 10);
}

TEST(Connection, ConnectionAssignedCorrectlyToNeurons)
{
    std::shared_ptr<neuron::Neuron> testNeuron1 = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid));
    std::shared_ptr<neuron::Neuron> testNeuron2 = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid));
    connection::Connection testConnection(testNeuron1, testNeuron2);

    testConnection.configureConnectedNeurons();
    
    ASSERT_EQ(testNeuron1->connections_forward.size(), 1);
    ASSERT_EQ(testNeuron2->connections_back.size(), 1);
}
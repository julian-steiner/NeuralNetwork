#include <gtest/gtest.h>
#include "Neuron.h"
#include <iostream>

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

TEST(Neuron, NeuronInitializedCorrectly)
{
    std::shared_ptr<neuron::Neuron> testNeuron = std::make_shared<neuron::Neuron>(neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false));
    ASSERT_EQ(testNeuron->activation, neuron::Activation::Sigmoid);
    ASSERT_EQ(testNeuron->value, 0);
    ASSERT_EQ(testNeuron->type, neuron::NeuronType::Input);
    ASSERT_EQ(testNeuron->bias, 0);
}

TEST(Neuron, BinaryActivationWorkingCorrectly)
{
    neuron::Neuron testNeuron = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Binary, false);
    testNeuron.value = -0.5;

    neuron::Neuron testNeuron2 = neuron::Neuron(neuron::NeuronType::Hidden, neuron::Activation::Binary, false);
    testNeuron2.bias = 0;

    connection::Connection connection = connection::Connection(&testNeuron, &testNeuron2);
    connection.weight = 1;

    ASSERT_EQ(testNeuron2.recursiveCalculate(), 0);

    testNeuron.value = 1;

    ASSERT_EQ(testNeuron2.recursiveCalculate(), 1);
}

TEST(Neuron, InputNeuronComputingCorrectly)
{
    neuron::Neuron testNeuron = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Without, false);
    testNeuron.value = 53;
    testNeuron.bias = 23103910;

    double result = testNeuron.calculate();

    ASSERT_EQ(result, 53);
}

TEST(Neuron, ComputingCorrectlyWithoutCache)
{
    neuron::Neuron testNeuron = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    testNeuron.value = 54;

    neuron::Neuron testNeuron2 = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    testNeuron2.value = 100;

    neuron::Neuron testNeuron3 = neuron::Neuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, false, false);
    testNeuron3.bias = 1;

    connection::Connection connection1 = connection::Connection(&testNeuron, &testNeuron3);
    connection::Connection connection2 = connection::Connection(&testNeuron2, &testNeuron3);
    
    double weight1 = testNeuron3.connections_back.at(0)->weight;
    double weight2 = testNeuron3.connections_back.at(1)->weight;

    double result = testNeuron3.calculate();
    
    ASSERT_EQ(result, sigmoid(54 * weight1 + 100 * weight2 + 1));
}

TEST(Neuron, ComputingCorrectlyWithoutRefreshingCache)
{
    neuron::Neuron testNeuron = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    testNeuron.value = 54;

    neuron::Neuron testNeuron2 = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    testNeuron2.value = 100;

    neuron::Neuron testNeuron3 = neuron::Neuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, true);
    testNeuron3.bias = 1;

    testNeuron3.rewriteCache = false;
    testNeuron3.value = 12039;

    connection::Connection connection1 = connection::Connection(&testNeuron, &testNeuron3);
    connection::Connection connection2 = connection::Connection(&testNeuron2, &testNeuron3);

    double result = testNeuron3.calculate();
    
    ASSERT_EQ(result, 12039);
}

TEST(Neuron, ComputingCorrectlyWithRefreshingCache)
{
    neuron::Neuron testNeuron = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    testNeuron.value = 54;

    neuron::Neuron testNeuron2 = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    testNeuron2.value = 100;

    neuron::Neuron testNeuron3 = neuron::Neuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, true);
    testNeuron3.bias = 1;

    testNeuron3.rewriteCache = true;
    testNeuron3.value = 12039;

    connection::Connection connection1 = connection::Connection(&testNeuron, &testNeuron3);
    connection::Connection connection2 = connection::Connection(&testNeuron2, &testNeuron3);
    
    double weight1 = testNeuron3.connections_back.at(0)->weight;
    double weight2 = testNeuron3.connections_back.at(1)->weight;

    double result = testNeuron3.calculate();
    
    ASSERT_EQ(testNeuron3.weightedSumCache, 54 * weight1 + 100 * weight2 + 1);
    ASSERT_EQ(result, sigmoid(54 * weight1 + 100 * weight2 + 1));
}

TEST(Neuron, ReverseCalculateWorkingCorrectly)
{
    neuron::Neuron testNeuron = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    testNeuron.value = 54;

    neuron::Neuron testNeuron2 = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    testNeuron2.value = 100;

    neuron::Neuron testNeuron3 = neuron::Neuron(neuron::NeuronType::Hidden, neuron::Activation::Without, false);
    testNeuron3.bias = 3;

    neuron::Neuron testNeuron4 = neuron::Neuron(neuron::NeuronType::Hidden, neuron::Activation::Without, false);
    testNeuron4.bias = 4;

    connection::Connection connection1 = connection::Connection(&testNeuron, &testNeuron3);
    connection::Connection connection2 = connection::Connection(&testNeuron2, &testNeuron3);
    connection::Connection connection3 = connection::Connection(&testNeuron3, &testNeuron4);

    connection1.weight = 1;
    connection2.weight = 2;
    connection3.weight = 3;
    
    double result = testNeuron4.recursiveCalculate();
    double result_check = (54 + 100*2 + 3) * 3 + 4;

    ASSERT_EQ(result, result_check);
}

TEST(Connection, ConnectionInitializedCorrectly)
{
    neuron::Neuron testNeuron = neuron::Neuron(neuron::NeuronType::Output, neuron::Activation::Sigmoid, false);
    testNeuron.value = 54;
    neuron::Neuron testNeuron2 = neuron::Neuron(neuron::NeuronType::Output, neuron::Activation::Sigmoid, false);

    connection::Connection testConnection(&testNeuron, &testNeuron2);
    
    ASSERT_EQ(testConnection.in, &testNeuron);
    ASSERT_EQ(testConnection.out, &testNeuron2);
    
    testNeuron.value = 10;
    
    ASSERT_EQ(testConnection.in->value, 10);
}

TEST(Connection, ConnectionAssignedCorrectlyToNeurons)
{
    neuron::Neuron testNeuron1 = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    neuron::Neuron testNeuron2 = neuron::Neuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, false);
    connection::Connection testConnection(&testNeuron1, &testNeuron2);

    testConnection.configureConnectedNeurons();
    
    ASSERT_EQ(testNeuron1.connections_forward.size(), 1);
    ASSERT_EQ(testNeuron2.connections_back.size(), 1);
}
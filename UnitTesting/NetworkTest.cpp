#include <gtest/gtest.h>
#include "NetworkBuffer.h"
#include "NeuralNetwork.h"
#include <cstdlib>

TEST(NetworkBuffer, BufferInitializedCorrectly)
{
    nn::NetworkBuffer testNetwork;
    ASSERT_EQ(testNetwork.connections->size(), 0);
    ASSERT_EQ(testNetwork.neurons->size(), 0);
}

TEST(NetworkBuffer, NeuronCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(0, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addNeuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, 0);

    ASSERT_EQ(testNetwork.neurons->size(), 1);
}

TEST(NetworkBuffer, ConnectionCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(0, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addNeuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, 0);
    testNetwork.addNeuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, 0);
    testNetwork.connect({0, 0}, {0, 1});

    ASSERT_EQ(testNetwork.neurons->size(), 2);
    ASSERT_EQ(testNetwork.connections->size(), 1);
}

TEST(NetworkBuffer, ConnectingWorkingCorrectly)
{
    nn::NetworkBuffer testNetwork;
    testNetwork.addLayer(0, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addNeuron(neuron::NeuronType::Input, neuron::Activation::Sigmoid, 0);
    testNetwork.addNeuron(neuron::NeuronType::Input, neuron::Activation::None, 0);
    
    testNetwork.connect({0, 0}, {0, 1});

    ASSERT_EQ(testNetwork.neurons->size(), 2);
    ASSERT_EQ(testNetwork.connections->size(), 1);
    ASSERT_EQ(testNetwork.connections->at(0)->in->activation, neuron::Activation::Sigmoid);
    ASSERT_EQ(testNetwork.connections->at(0)->out->activation, neuron::Activation::None);
}

TEST(NetworkBuffer, LayersCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    ASSERT_EQ(testNetwork.inputLayer->size, 10);
    ASSERT_EQ(testNetwork.neurons->size(), 32);
    ASSERT_EQ(testNetwork.previousLayerSize, 2);
    ASSERT_EQ(testNetwork.neurons->at(5)->type, neuron::NeuronType::Input);
    ASSERT_EQ(testNetwork.neurons->at(15)->type, neuron::NeuronType::Hidden);
}

TEST(NetworkBuffer, ConnectionsCorrectlyAdded)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    ASSERT_EQ(testNetwork.neurons->at(0)->connections_forward.size(), 20);
    ASSERT_EQ(testNetwork.neurons->at(15)->connections_back.size(), 10);
    ASSERT_EQ(testNetwork.neurons->at(31)->connections_forward.size(), 0);
    ASSERT_EQ(testNetwork.neurons->at(31)->connections_back.size(), 20);
}

TEST(NetworkBuffer, BufferCopiedCorrectly)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    testNetwork.addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 2);
    testNetwork.addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 2);

    testNetwork.connect({2, 0}, {3, 1});
    testNetwork.connect({2, 0}, {2, 1});

    testNetwork.layers->at(2)->neurons.at(0)->bias = 69;

    nn::NetworkBuffer testNetwork2 = testNetwork.getCopy<nn::NetworkBuffer>();

    ASSERT_EQ(testNetwork2.connections->size(), testNetwork.connections->size());
    ASSERT_EQ(testNetwork2.layers->at(0)->neurons.size(), 10);
    ASSERT_EQ(testNetwork2.neurons->at(0)->connections_forward.size(), testNetwork.neurons->at(0)->connections_forward.size());
    ASSERT_EQ(testNetwork2.neurons->at(15)->connections_back.size(), testNetwork.neurons->at(15)->connections_back.size());
    ASSERT_EQ(testNetwork2.neurons->at(31)->connections_forward.size(), testNetwork.neurons->at(31)->connections_forward.size());
    ASSERT_EQ(testNetwork2.neurons->at(31)->connections_back.size(), testNetwork.neurons->at(31)->connections_back.size());
    connection::ConnectionDummy connection = testNetwork2.layers->at(3)->connectionDummys.at(0);
    ASSERT_EQ(testNetwork2.layers->at(connection.inNeuronLocation.layer)->neurons.at(connection.inNeuronLocation.number)->bias, 69);
}

TEST(NetworkBuffer, LayerNumberAddedCorrectly)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::FullyConnected);

    ASSERT_EQ(testNetwork.neurons->at(2)->layerNumber, 1);
}

TEST(NetworkBuffer, ConnectionDummiesAddedCorrectly)
{
    nn::NetworkBuffer testNetwork;

    testNetwork.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::FullyConnected);

    testNetwork.addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1);
    testNetwork.addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1);

    testNetwork.connect({0, 0}, {1, 0});
    testNetwork.connect({1, 0}, {1, 1});

    ASSERT_EQ(testNetwork.layers->at(0)->connectionDummys.size(), 0);
    ASSERT_EQ(testNetwork.layers->at(1)->connectionDummys.size(), 2);

    ASSERT_EQ(testNetwork.layers->at(1)->connectionDummys.at(1).inNeuronLocation.number, 0);
    ASSERT_EQ(testNetwork.layers->at(1)->connectionDummys.at(1).outNeuronLocation.layer, 1);
}

TEST(NetworkBuffer, RecursionTestCorrectly)
{
    nn::NetworkBuffer testNetworkBuffer1;

    testNetworkBuffer1.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetworkBuffer1.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetworkBuffer1.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);

    ASSERT_EQ(testNetworkBuffer1.connections->size(), 2);
    ASSERT_EQ(testNetworkBuffer1.checkForRecursion({1, 0}, {0, 0}), true);
    ASSERT_EQ(testNetworkBuffer1.checkForRecursion({2, 0}, {0, 0}), true);

    testNetworkBuffer1.addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1);

    ASSERT_EQ(false, testNetworkBuffer1.checkForRecursion({0, 0}, {1, 1}));
}

TEST(NetworkBuffer, PrintingCorrectly)
{
    nn::NetworkBuffer testNetworkBuffer;

    testNetworkBuffer.addLayer(2, neuron::Activation::None, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetworkBuffer.addLayer(3, neuron::Activation::None, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    testNetworkBuffer.addLayer(2, neuron::Activation::None, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    testNetworkBuffer.addLayer(2, neuron::Activation::None, nn::LayerType::Output, nn::LayerConnectionType::CustomConnected);

    testNetworkBuffer.connect({0, 0}, {1, 0});
    testNetworkBuffer.connect({0, 0}, {1, 1});
    testNetworkBuffer.connect({0, 0}, {1, 2});
    testNetworkBuffer.connect({0, 1}, {1, 0});
    testNetworkBuffer.connect({0, 1}, {1, 2});

    testNetworkBuffer.connect({1, 0}, {2, 0});
    testNetworkBuffer.connect({1, 1}, {2, 1});
    testNetworkBuffer.connect({1, 2}, {2, 1});

    testNetworkBuffer.connect({2, 0}, {3, 0});
    testNetworkBuffer.connect({2, 0}, {3, 1});
    testNetworkBuffer.connect({2, 1}, {3, 1});

    std::vector<std::string> scheme = testNetworkBuffer.getConnectionScheme();
}

TEST(NeuralNetwork, NetworkInheritanceWorkingCorrectly)
{
    nn::NeuralNetwork testNetwork; 
    testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

    ASSERT_EQ(testNetwork.neurons->at(0)->connections_forward.size(), 20);
    ASSERT_EQ(testNetwork.neurons->at(15)->connections_back.size(), 10);
    ASSERT_EQ(testNetwork.neurons->at(31)->connections_forward.size(), 0);
    ASSERT_EQ(testNetwork.neurons->at(31)->connections_back.size(), 20);
}

TEST(NeuralNetwork, FeedforwardWorkingCorrectly)
{
    nn::NeuralNetwork testNetwork;
    testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(2, neuron::Activation::Binary, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
    testNetwork.addLayer(1, neuron::Activation::Binary, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);
    
    testNetwork.connections->at(0)->weight = 1;
    testNetwork.connections->at(1)->weight = 1;
    testNetwork.connections->at(2)->weight = -1;
    testNetwork.connections->at(3)->weight = -1;
    testNetwork.connections->at(4)->weight = 1;
    testNetwork.connections->at(5)->weight = 1;

    ASSERT_EQ(testNetwork.predict({0, 0}).at(0), 0);
    ASSERT_EQ(testNetwork.predict({0, 1}).at(0), 1);
    ASSERT_EQ(testNetwork.predict({1, 0}).at(0), 1);
    ASSERT_EQ(testNetwork.predict({1, 1}).at(0), 0);
}

TEST(NeuralNetwork, NetworkCopiedCorrectly)
{
    {
        nn::NeuralNetwork testNetwork;

        testNetwork.addLayer(10, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
        testNetwork.addLayer(20, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
        testNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::FullyConnected);
        testNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

        testNetwork.addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 2);
        testNetwork.addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 2);

        testNetwork.connect({2, 0}, {3, 1});
        testNetwork.connect({2, 0}, {2, 1});

        testNetwork.layers->at(2)->neurons.at(0)->bias = 69;

        nn::NeuralNetwork testNetwork2 = testNetwork.getCopy<nn::NeuralNetwork>();

        ASSERT_EQ(testNetwork2.connections->size(), testNetwork.connections->size());
        ASSERT_EQ(testNetwork2.neurons->at(0)->connections_forward.size(), testNetwork.neurons->at(0)->connections_forward.size());
        ASSERT_EQ(testNetwork2.neurons->at(15)->connections_back.size(), testNetwork.neurons->at(15)->connections_back.size());
        ASSERT_EQ(testNetwork2.neurons->at(31)->connections_forward.size(), testNetwork.neurons->at(31)->connections_forward.size());
        ASSERT_EQ(testNetwork2.neurons->at(31)->connections_back.size(), testNetwork.neurons->at(31)->connections_back.size());
        connection::ConnectionDummy connection = testNetwork2.layers->at(3)->connectionDummys.at(0);
        ASSERT_EQ(testNetwork2.layers->at(connection.inNeuronLocation.layer)->neurons.at(connection.inNeuronLocation.number)->bias, 69);
    }
}

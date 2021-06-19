#include "NetworkBuffer.h"
#include "NeuralNetwork.h"

using namespace nn;

NetworkBuffer::NetworkBuffer()
{
    previousLayerSize = 0;
    currentLayerNumber = 0;
    this->connections = std::vector<connection::Connection*>();
    this->neurons = std::vector<neuron::Neuron*>();
    this->layers = std::vector<nn::Layer*>();
}

NetworkBuffer::~NetworkBuffer()
{
    for (connection::Connection* c_connection : this->connections)
    {
        delete c_connection;
    }

    for (neuron::Neuron* c_neuron : this->neurons)
    {
        delete c_neuron;
    }

    for (nn::Layer* c_layer : this->layers)
    {
        delete c_layer;
    }
}

void NetworkBuffer::addConnection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber)
{
    this->connections.push_back(new connection::Connection(in, out, innovationNumber));
}

void NetworkBuffer::connect(connection::NeuronLocation inNeuronLocation, connection::NeuronLocation outNeuronLocation, int innovationNumber)
{
    neuron::Neuron* in = this->layers.at(inNeuronLocation.layer)->neurons.at(inNeuronLocation.number);
    neuron::Neuron* out = this->layers.at(outNeuronLocation.layer)->neurons.at(outNeuronLocation.number);
    addConnection(in, out, innovationNumber);

    // Add the connection to the connectionDummies in the layer (only to the out to prevent redundancy)
    this->layers.at(out->layerNumber)->connectionDummys.emplace_back(connection::ConnectionDummy(inNeuronLocation, outNeuronLocation));
}

void NetworkBuffer::addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber)
{
    this->neurons.push_back(new neuron::Neuron(type, activation, true, layerNumber));
    this->layers.at(layerNumber)->neurons.push_back(this->neurons.back());
}

void NetworkBuffer::addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber, double bias)
{
    this->neurons.push_back(new neuron::Neuron(type, activation, bias, true, layerNumber));
    this->layers.at(layerNumber)->neurons.push_back(this->neurons.back());
}

void NetworkBuffer::addNeuron(neuron::Neuron&& neuron, int layerNumber)
{
    neuron.layerNumber = layerNumber;
    this->neurons.push_back(&neuron);
    this->layers.at(layerNumber)->neurons.push_back(this->neurons.back());
}


void NetworkBuffer::addLayer(int numNeurons, neuron::Activation activation, LayerType layerType, LayerConnectionType layerConnectionType)
{
    //Determining the layerType out of the network size
    //If the Network size is 0 then the layer has to be input

    int currentNetworkSize = this->neurons.size();
    neuron::NeuronType neuronType = neuron::NeuronType::Hidden;

    //Reserve the space in the Neurons Vector to minimize copying
    this->neurons.reserve(this->neurons.size() + numNeurons);

    if (currentNetworkSize == 0)
    {
        neuronType = neuron::NeuronType::Input;
        layerType = LayerType::Input;
    }

    //Add and reserve space in the layers vector
    switch (layerType)
    {
        case LayerType::Input:
            this->layers.emplace_back(new InputLayer(numNeurons, activation, layerConnectionType));
            this->inputLayer = (nn::InputLayer*) this->layers.at(currentLayerNumber);
            break;
        case LayerType::Output:
            this->layers.emplace_back(new OutputLayer(numNeurons, activation, layerConnectionType));
            this->outputLayer = (nn::OutputLayer*) this->layers.at(currentLayerNumber);
            break;
        case LayerType::Hidden:
            this->layers.emplace_back(new HiddenLayer(numNeurons, activation, layerConnectionType));
            break;
        case LayerType::CustomConnectedHidden:
            this->layers.emplace_back(new CustomConnectedHiddenLayer(numNeurons, activation, layerConnectionType));
            break;
    }

    this->layers.at(currentLayerNumber)->neurons.reserve(numNeurons);

    //Adding the neurons to the network

    for(int i = 0; i < numNeurons; i++)
    {
        addNeuron(neuronType, activation, currentLayerNumber);
    }

    //Connecting the layer if it should be
    switch (layerConnectionType)
    {
        case LayerConnectionType::FullyConnected:
            if(neuronType != neuron::NeuronType::Input)
            {
                //Reserving the space in the connections Vector
                this->connections.reserve(this->connections.size() + (numNeurons * previousLayerSize));

                for(int i = 0; i < this->layers.at(currentLayerNumber)->getSize(); i++)
                {
                    for(int a = 0; a < this->layers.at(currentLayerNumber-1)->getSize(); a++)
                    {
                        connect({currentLayerNumber-1, a}, {currentLayerNumber, i});
                    }
                }
            }
    }
    this->previousLayerSize = numNeurons;
    this->currentLayerNumber ++;
}

template<typename T>
T nn::NetworkBuffer::getCopy()
{
    T tempNetworkBuffer;

    // Add every layer
    for (nn::Layer* currentLayer : this->layers)
    {
        // Add the layers
        switch(currentLayer->layerType)
        {
            case(nn::LayerType::Input):
                tempNetworkBuffer.layers.emplace_back(new nn::InputLayer(currentLayer->getSize(), currentLayer->activation, currentLayer->layerConnectionType));
                tempNetworkBuffer.inputLayer = (nn::InputLayer*)tempNetworkBuffer.layers.back();
                break;

            case(nn::LayerType::Output):
                tempNetworkBuffer.layers.emplace_back(new nn::OutputLayer(currentLayer->getSize(), currentLayer->activation, currentLayer->layerConnectionType));
                tempNetworkBuffer.outputLayer = (nn::OutputLayer*)tempNetworkBuffer.layers.back();
                break;

            case(nn::LayerType::Hidden):
                tempNetworkBuffer.layers.emplace_back(new nn::HiddenLayer(currentLayer->getSize(), currentLayer->activation, currentLayer->layerConnectionType));
                break;

            case(nn::LayerType::CustomConnectedHidden):
                tempNetworkBuffer.layers.emplace_back(new nn::CustomConnectedHiddenLayer(currentLayer->getSize(), currentLayer->activation, currentLayer->layerConnectionType));
                break;
        }

        // Copy all the layer parameters to the new layer
        tempNetworkBuffer.layers.back()->connectionDummys = currentLayer->connectionDummys;

        // Add the neurons
        for (neuron::Neuron* currentNeuron : currentLayer->neurons)
        {
            tempNetworkBuffer.addNeuron(currentNeuron->type, currentNeuron->activation, currentNeuron->layerNumber, currentNeuron->bias);
        }
    }

    // Add the connections (here because otherwise you get nullptrs)
    for (nn::Layer* currentLayer : this->layers)
    {
        // Add the connections
        for (connection::ConnectionDummy currentConnectionDummy : currentLayer->connectionDummys)
        {
            tempNetworkBuffer.connect(currentConnectionDummy.inNeuronLocation, currentConnectionDummy.outNeuronLocation);
        }
    }

    return tempNetworkBuffer;
}

template nn::NetworkBuffer nn::NetworkBuffer::getCopy<nn::NetworkBuffer>();
template nn::NeuralNetwork nn::NetworkBuffer::getCopy<nn::NeuralNetwork>();
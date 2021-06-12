#include "NetworkBuffer.h"

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

void NetworkBuffer::addConnection(neuron::Neuron* in, neuron::Neuron* out)
{
    this->connections.push_back(new connection::Connection(in, out));
}

void NetworkBuffer::addConnection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber)
{
    this->connections.push_back(new connection::Connection(in, out, innovationNumber));
}

void NetworkBuffer::connect(int inNeuronNumber, int outNeuronNumber)
{
    this->connections.push_back(new connection::Connection(this->neurons.at(inNeuronNumber-1), this->neurons.at(outNeuronNumber-1)));
}

void NetworkBuffer::addNeuron(neuron::NeuronType type, neuron::Activation activation)
{
    this->neurons.push_back(new neuron::Neuron(type, activation, true));
}

void NetworkBuffer::addNeuron(neuron::Neuron&& neuron)
{
    this->neurons.push_back(&neuron);
}

void NetworkBuffer::addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber)
{
    neuron::Neuron* neuronPtr = new neuron::Neuron(type, activation, true);
    this->neurons.push_back(neuronPtr);
    this->layers.at(layerNumber)->neurons.push_back(neuronPtr);
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
    std::cout << layerType << std::endl;
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

                for(int i = currentNetworkSize; i < currentNetworkSize + numNeurons; i++)
                {
                    for(int a = currentNetworkSize - this->previousLayerSize; a < currentNetworkSize; a++)
                    {
                        addConnection(this->neurons.at(a), this->neurons.at(i));
                    }
                }
            }
    }
    this->previousLayerSize = numNeurons;
    this->currentLayerNumber ++;
}

nn::NetworkBuffer nn::NetworkBuffer::getCopy()
{
    nn::NetworkBuffer tempNetworkBuffer;

    for (nn::Layer* currentLayer: this->layers)
    {
        tempNetworkBuffer.addLayer(currentLayer->size, currentLayer->activation, nn::LayerType::Hidden, currentLayer->layerType);
    }

    return tempNetworkBuffer;
}
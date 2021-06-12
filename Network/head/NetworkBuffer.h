#ifndef NEURALNETWORK
#define NEURALNETWORK

#include <vector>
#include <memory>
#include "Layer.h"
#include "../../Neuron/head/Neuron.h"

namespace nn
{

    enum LayerType : unsigned int {Input, Output, Hidden, CustomConnectedHidden};
    enum LayerConnectionType : unsigned int {FullyConnected};

    struct NetworkBuffer
    {
        int previousLayerSize;

        nn::InputLayer* inputLayer;
        nn::OutputLayer* outputLayer;

        int currentLayerNumber;
        std::vector<neuron::Neuron*> neurons;
        std::vector<connection::Connection*> connections;
        std::vector<nn::Layer*> layers;

        NetworkBuffer();
        ~NetworkBuffer();

        void addConnection(neuron::Neuron* in, neuron::Neuron* out);
        void addConnection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber);
        void addNeuron(neuron::Neuron&& neuron, int layerNumber);
        void addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber);
        void connect(int inNeuronNumber, int outNeuronNumber);

        void addLayer(int numNeurons, neuron::Activation activation, LayerType layerType, LayerConnectionType connectionType);

        nn::NetworkBuffer getCopy();
    };
}

#endif
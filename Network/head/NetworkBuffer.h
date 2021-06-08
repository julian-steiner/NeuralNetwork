#ifndef NEURALNETWORK
#define NEURALNETWORK

#include <vector>
#include <memory>
#include "../../Neuron/head/Neuron.h"

namespace nn
{

    enum LayerType {FullyConnected};

    struct NetworkBuffer
    {
        int previousLayerSize;
        int inputLayerSize;
        int currentLayerNumber;
        std::vector<neuron::Neuron*> neurons;
        std::vector<connection::Connection*> connections;
        std::vector<std::vector<neuron::Neuron*>> layers;

        NetworkBuffer();
        ~NetworkBuffer();

        void addConnection(neuron::Neuron* in, neuron::Neuron* out);
        void addConnection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber);
        void addNeuron(neuron::Neuron&& neuron);

        void addNeuron(neuron::NeuronType type, neuron::Activation activation);
        void connect(int inNeuronNumber, int outNeuronNumber);

        //TODO: Fix the bug that addLayer changes the pointers
        void addLayer(int numNeurons, neuron::Activation activation, LayerType type);

        private:
        void addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber);
    };
}

#endif
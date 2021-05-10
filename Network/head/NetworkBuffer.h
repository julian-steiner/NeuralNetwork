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
        std::vector<std::shared_ptr<neuron::Neuron>> neurons;
        std::vector<std::shared_ptr<connection::Connection>> connections;

        NetworkBuffer();
        void addConnection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out);
        void addConnection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out, int innovationNumber);
        void addNeuron(std::shared_ptr<neuron::Neuron> neuron);
        void addLayer(int numNeurons, neuron::Activation activation, LayerType type);
    };
}

#endif
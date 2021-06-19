#ifndef NEURALNETWORK
#define NEURALNETWORK

#include <vector>
#include <memory>
#include "Layer.h"

namespace nn
{
    class NeuralNetwork;

    enum LayerConnectionType : unsigned int {FullyConnected, CustomConnected};

    class NetworkBuffer
    {
    public:
        int previousLayerSize;

        nn::InputLayer* inputLayer;
        nn::OutputLayer* outputLayer;

        int currentLayerNumber;
        std::vector<neuron::Neuron*> neurons;
        std::vector<connection::Connection*> connections;
        std::vector<nn::Layer*> layers;

        NetworkBuffer();
        ~NetworkBuffer();

        void addNeuron(neuron::Neuron&& neuron, int layerNumber);
        void addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber);
        void addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber, double weight);
        void connect(connection::NeuronLocation inNeuronNumber, connection::NeuronLocation outNeuronLocation, int innovationNumber=0);

        void addLayer(int numNeurons, neuron::Activation activation, LayerType layerType, LayerConnectionType connectionType);

        template<typename T>
        T getCopy();
    
    private:
        void addConnection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber=0);
    };
}

#endif
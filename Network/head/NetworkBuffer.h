#ifndef NEURALNETWORK
#define NEURALNETWORK

#include <vector>
#include <memory>
#include "Layer.h"
#include "CopyNetworkImpl.h"

namespace nn
{
    class NeuralNetwork;

    enum LayerConnectionType : unsigned int {FullyConnected, CustomConnected};

    class NetworkBuffer
    {

    private:
        void addConnection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber=0, connection::NeuronLocation inNeuronLocation = {0, 0}, connection::NeuronLocation outNeuronLocation = {0, 0});
        bool RecursivelyCheckForRecursion(neuron::Neuron* currentNeuron, neuron::Neuron* targetNeuron);

    public:
        int previousLayerSize;

        nn::InputLayer* inputLayer;
        nn::OutputLayer* outputLayer;

        int currentLayerNumber;
        std::vector<neuron::Neuron*>* neurons;
        std::vector<connection::Connection*>* connections;
        std::vector<nn::Layer*>* layers;

        NetworkBuffer();
        ~NetworkBuffer();
        NetworkBuffer(const NetworkBuffer& other);
        NetworkBuffer(NetworkBuffer&& other);
        NetworkBuffer& operator=(const nn::NetworkBuffer& other);

        void addNeuron(neuron::Neuron&& neuron, int layerNumber);
        void addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber);
        void addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber, double weight);
        void connect(connection::NeuronLocation inNeuronNumber, connection::NeuronLocation outNeuronLocation, int innovationNumber=0);
        bool checkForRecursion(connection::NeuronLocation inNeuronNumber, connection::NeuronLocation outNeuronNumber);

        void addLayer(int numNeurons, neuron::Activation activation, LayerType layerType, LayerConnectionType connectionType);

        template<typename T>
        T getCopy();
    };
}

#endif
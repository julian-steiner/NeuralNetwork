#ifndef LAYER
#define LAYER

#include "Neuron.h"

namespace nn
{
    enum LayerConnectionType : unsigned int;

    struct Layer
    {
        Layer() {};
        Layer(int size, neuron::Activation activation, nn::LayerConnectionType layerType);

        int size;
        neuron::Activation activation;
        nn::LayerConnectionType layerType;
        std::vector<neuron::Neuron*> neurons;
    };

    struct InputLayer : public Layer
    {
        InputLayer() {};
        InputLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerType);
    };

    struct OutputLayer : public Layer
    {
        OutputLayer() {};
        OutputLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerType);
    };

    struct HiddenLayer : public Layer
    {
        HiddenLayer() {};
        HiddenLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerType);
    };

    struct CustomConnectedHiddenLayer : public Layer
    {
        CustomConnectedHiddenLayer() {};
        CustomConnectedHiddenLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerType);
    };
}

#endif
#include "Layer.h"

using namespace nn;

Layer::Layer(int size, neuron::Activation activation, nn::LayerConnectionType layerConnectionType)
{
    this->size = size;
    this->activation = activation;
    this->layerConnectionType = layerConnectionType;
    this->layerType = nn::LayerType::Input;
}

int Layer::getSize()
{
    return this->neurons.size();
}

InputLayer::InputLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerConnectionType)
{
    this->size = size;
    this->activation = activation;
    this->layerConnectionType = layerConnectionType;
    this->layerType = nn::LayerType::Input;
}

OutputLayer::OutputLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerConnectionType)
{
    this->size = size;
    this->activation = activation;
    this->layerConnectionType = layerConnectionType;
    this->layerType = nn::LayerType::Output;
}

HiddenLayer::HiddenLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerConnectionType)
{
    this->size = size;
    this->activation = activation;
    this->layerConnectionType = layerConnectionType;
    this->layerType = nn::LayerType::Hidden;
}

CustomConnectedHiddenLayer::CustomConnectedHiddenLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerConnectionType)
{
    this->size = size;
    this->activation = activation;
    this->layerConnectionType = layerConnectionType;
    this->layerType = nn::LayerType::CustomConnectedHidden;
}
#include "Layer.h"

using namespace nn;

Layer::Layer(int size, neuron::Activation activation, nn::LayerConnectionType layerType)
{
    this->size = size;
    this->activation = activation;
    this->layerType = layerType;
}

InputLayer::InputLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerType)
{
    this->size = size;
    this->activation = activation;
    this->layerType = layerType;
}

OutputLayer::OutputLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerType)
{
    this->size = size;
    this->activation = activation;
    this->layerType = layerType;
}

HiddenLayer::HiddenLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerType)
{
    this->size = size;
    this->activation = activation;
    this->layerType = layerType;
}

CustomConnectedHiddenLayer::CustomConnectedHiddenLayer(int size, neuron::Activation activation, nn::LayerConnectionType layerType)
{
    this->size = size;
    this->activation = activation;
    this->layerType = layerType;
}
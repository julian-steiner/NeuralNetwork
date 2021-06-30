#include "NeuralNetwork.h"
#include "Profiler.h"

void nn::NeuralNetwork::calculateNeuron(neuron::Neuron* neuron)
{
    neuron->calculate();
}

std::vector<double> nn::NeuralNetwork::predict(std::vector<double> inputs)
{
    // check if the inputs have the right length
    if (inputs.size() != this->inputLayer->size)
    {
        std::cout << "Input vector has to be the same size as the input Layer" << std::endl;
        throw ("Input vector has to be the same size as the input Layer");
    }

    // reset the cache on the neurons 
    for (neuron::Neuron* c_neuron: *this->neurons)
    {
        c_neuron->rewriteCache = true;
    }

    // set the values on the input neurons
    for (int i = 0; i < this->inputLayer->size; i++)
    {
        this->inputLayer->neurons.at(i)->value = inputs.at(i);
    }

    // create the result vector
    std::vector<double> result = std::vector<double>();
    result.reserve(this->previousLayerSize);

    // feedforward
    for (int i = 0; i < this->outputLayer->neurons.size(); i++)
    {
        neuron::Neuron* c_neuron = this->outputLayer->neurons.at(i);
        result.push_back(c_neuron->recursiveCalculate());
    }

    return result;
}
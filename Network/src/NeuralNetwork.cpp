#include "NeuralNetwork.h"
#include "Profiler.h"

void nn::NeuralNetwork::calculateNeuron(neuron::Neuron* neuron)
{
    neuron->calculate();
}

std::vector<double> nn::NeuralNetwork::predict(std::vector<double> inputs)
{
    // check if the inputs have the right length
    if (inputs.size() != this->inputLayerSize)
    {
        std::cout << "Input vector has to be the same size as the input Layer" << std::endl;
        throw ("Input vector has to be the same size as the input Layer");
    }

    // reset the cache on the neurons 
    for (neuron::Neuron* c_neuron: this->neurons)
    {
        c_neuron->rewriteCache = true;
    }

    // set the values on the input neurons
    for (int i = 0; i < this->inputLayerSize; i++)
    {
        this->neurons.at(i)->value = inputs.at(i);
    }

    // create the result vector
    std::vector<double> result = std::vector<double>();
    result.reserve(this->previousLayerSize);

    // feedforward
    int a = this->neurons.size() - this->previousLayerSize;
    for (int i = this->neurons.size() - this->previousLayerSize; i < this->neurons.size(); i++)
    {
        result.push_back(this->neurons.at(i)->recursiveCalculate());
    }

    return result;
}

//std::vector<double> nn::NeuralNetwork::predict(std::vector<double> inputs)
//{
    //// check if the inputs have the right length
    //if (inputs.size() != this->inputLayerSize)
    //{
        //std::cout << "Input vector has to be the same size as the input Layer" << std::endl;
        //throw ("Input vector has to be the same size as the input Layer");
    //}

    //// reset the cache on the neurons 
    //for (neuron::Neuron* c_neuron: this->neurons)
    //{
        //c_neuron->rewriteCache = std::make_shared<bool>(true);
    //}

    //// set the values on the input neurons
    //for (int i = 0; i < this->inputLayerSize; i++)
    //{
        //this->neurons.at(i)->value = inputs.at(i);
    //}

    //// create the result vector
    //std::vector<double> result = std::vector<double>();
    //result.reserve(this->previousLayerSize);

    //std::vector<std::future<void>> futures;
    //futures.reserve(this->neurons.size());

    //// asynchronously calculating one layer at a time
    //{
        //PROFILE_SCOPE("Preprocessing");
        //for (int i = 1; i < this->layers.size(); i++)
        //{
            //for (neuron::Neuron* c_neuron : this->layers.at(i))
            //{
                ////c_neuron->calculate();
                //futures.push_back(std::async(std::launch::async, calculateNeuron, c_neuron));
            //}
        //}
    //}

    //// feedforward
    //int a = this->neurons.size() - this->previousLayerSize;
    //for (neuron::Neuron* c_output_neuron : this->layers.back())
    //{
        //result.push_back(c_output_neuron->value);
    //}

    //return result;
//}
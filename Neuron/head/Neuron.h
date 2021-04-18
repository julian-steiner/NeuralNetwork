#ifndef NeuronHeader
#define NeuronHeader

#include <vector>
#include <memory>

namespace neuron
{
    enum NeuronType {Input, Hidden, Output};
    enum Activation {Sigmoid, None};

    struct Neuron
    {
        double value;
        double bias;
        NeuronType type;
        Activation activation;

        Neuron(NeuronType type, Activation activation);
    };
}

namespace connection
{
    struct Connection
    {
        std::shared_ptr<neuron::Neuron> in;
        std::shared_ptr<neuron::Neuron> out;
        double weight;
        bool enabled;
        int innovationNumber;

        Connection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out);
        Connection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out, int innovationNumber);
    };
}

#endif
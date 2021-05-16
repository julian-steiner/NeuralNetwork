#ifndef NeuronHeader
#define NeuronHeader

#include <vector>
#include <memory>
#include <iostream>
#include <cmath>

namespace connection
{
    struct Connection;
}

namespace neuron
{
    enum NeuronType {Input, Hidden, Output};
    enum Activation {Sigmoid, None};

    struct Neuron
    {
        NeuronType type;
        Activation activation;

        double value;
        double bias;

        bool* rewriteCache;
        double weightedSumCache;

        std::vector<std::shared_ptr<connection::Connection>> connections_forward;
        std::vector<std::shared_ptr<connection::Connection>> connections_back;

        Neuron(NeuronType type, Activation activation);
        Neuron(NeuronType type, Activation activation, bool* rewriteCache);

        double calculate();
        double feedForward();

        private:
        double calculateSum();
        double activate(double value);
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
        bool isConnected;

        Connection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out);
        Connection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out, int innovationNumber);

        void configureConnectedNeurons();
    };
}

#endif
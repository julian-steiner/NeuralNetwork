#ifndef NeuronHeader
#define NeuronHeader

#include <vector>
#include <memory>
#include <iostream>
#include <cmath>
#include <random>

namespace connection
{
    struct Connection;
}

namespace neuron
{
    enum NeuronType {Input, Hidden, Output};
    enum Activation {Sigmoid, Binary, None};

    struct Neuron
    {
        NeuronType type;
        Activation activation;

        double value;
        double bias;

        bool rewriteCache;
        bool hasCache;
        double weightedSumCache;

        std::vector<connection::Connection*> connections_forward;
        std::vector<connection::Connection*> connections_back;

        Neuron(NeuronType type, Activation activation, bool has_cache = false);

        double calculate();
        double recursiveCalculate();

        private:
        double calculateSum();
        double activate(double value);
    };
}

namespace connection
{
    struct Connection
    {
        neuron::Neuron* in;
        neuron::Neuron* out;
        int inNeuronNumber;
        int outNeuronNumber;
        double weight;
        bool enabled;
        int innovationNumber;
        bool isConnected;

        Connection(neuron::Neuron* in, neuron::Neuron* out);
        Connection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber);
        Connection(neuron::Neuron* in, neuron::Neuron* out, int inNeuronNumber, int outNeuronNumber);

        void configureConnectedNeurons();
    };

    struct ConnectionDummy
    {
        int inNeuronNumber;
        int outNeuronNumber;
    };
}

#endif
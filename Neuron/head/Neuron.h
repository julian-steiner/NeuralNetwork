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

        int layerNumber;
        bool rewriteCache;
        bool hasCache;
        double weightedSumCache;

        std::vector<connection::Connection*> connections_forward;
        std::vector<connection::Connection*> connections_back;

        Neuron(NeuronType type, Activation activation, bool has_cache = false, int layerNumber=0);
        Neuron(NeuronType type, Activation activation, double bias, bool rewriteCache, bool hasCache, int layerNumber=0);

        double calculate();
        double recursiveCalculate();

        private:
        double calculateSum();
        double activate(double value);
    };
}

namespace connection
{
    struct NeuronLocation
    {
       int layer;
       int number; 
    };

    struct Connection
    {
        neuron::Neuron* in;
        neuron::Neuron* out;
        connection::NeuronLocation inNeuronLocation;
        connection::NeuronLocation outNeuronLocation;
        double weight;
        bool enabled;
        int innovationNumber;
        bool isConnected;

        Connection(neuron::Neuron* in, neuron::Neuron* out);
        Connection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber);
        Connection(neuron::Neuron* in, neuron::Neuron* out, connection::NeuronLocation inNeuronLocation, connection::NeuronLocation outNeuronLocation);

        void configureConnectedNeurons();
    };

    struct ConnectionDummy
    {
        connection::NeuronLocation inNeuronLocation;
        connection::NeuronLocation outNeuronLocation;

        ConnectionDummy(connection::NeuronLocation inNeuronLocation, connection::NeuronLocation outNeuronLocation);
    };
}

#endif
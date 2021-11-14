#ifndef NeuronHeader
#define NeuronHeader

namespace connection
{
    struct Connection;
}

namespace neuron
{
    enum NeuronType {Input, Hidden, Output};
    enum Activation {Sigmoid, Binary, Without};

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

        Neuron(NeuronType type, Activation activation, bool has_cache);
        Neuron(NeuronType type, Activation activation, bool has_cache, int layerNumber);
        Neuron(NeuronType type, Activation activation, double bias, bool rewriteCache, bool hasCache, int layerNumber);

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

    struct ConnectionDummy
    {
        connection::NeuronLocation inNeuronLocation;
        connection::NeuronLocation outNeuronLocation;
        int innovationNumber;
        ConnectionDummy(connection::NeuronLocation inNeuronLocation, connection::NeuronLocation outNeuronLocation, int innovationNumber);
        ConnectionDummy() = default;
    };

    struct Connection
    {
        neuron::Neuron* in;
        neuron::Neuron* out;
        double weight;
        bool enabled;
        int innovationNumber;
        connection::NeuronLocation inNeuronLocation;
        connection::NeuronLocation outNeuronLocation;

        bool isConnected;

        Connection(neuron::Neuron* in, neuron::Neuron* out);
        Connection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber);
        Connection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber, connection::NeuronLocation inNeuronLocation, connection::NeuronLocation outNeuronLocation);

        void configureConnectedNeurons();
    };
}

bool operator==(const connection::ConnectionDummy& first, const connection::ConnectionDummy& second);
bool operator==(const connection::Connection& first, const connection::Connection& second);
bool operator!=(const connection::ConnectionDummy& first, const connection::ConnectionDummy& second);
bool operator!=(const connection::Connection& first, const connection::Connection& second);

#endif
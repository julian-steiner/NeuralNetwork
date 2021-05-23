#include "Neuron.h"

using namespace neuron;
using namespace connection;

Connection::Connection(neuron::Neuron* in, neuron::Neuron* out)
{
    this->enabled = true;
    this->in = in;
    this->out = out;
    this->innovationNumber = 0;
    this->weight = (double)std::rand() / (double)RAND_MAX * 2 - 1;
    this->isConnected = false;
    this->configureConnectedNeurons();
}

Connection::Connection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber)
{
    this->enabled = true;
    this->in = in;
    this->out = out;
    this->innovationNumber = innovationNumber;
    this->weight = (double)std::rand() / (double)RAND_MAX * 2 - 1;
    this->isConnected = false;
    this->configureConnectedNeurons();
}

Connection::Connection(neuron::Neuron* in, neuron::Neuron* out, int inNeuronNumber, int outNeuronNumber)
{
    this->enabled = true;
    this->in = in;
    this->out = out;
    this->innovationNumber = 0;
    this->weight = (double)std::rand() / (double)RAND_MAX * 2 - 1;
    this->isConnected = false;
    this->inNeuronNumber = inNeuronNumber;
    this->outNeuronNumber = outNeuronNumber;
    this->configureConnectedNeurons();
}

void Connection::configureConnectedNeurons()
{
    if (isConnected == false)
    {
        this->in->connections_forward.push_back(this);
        this->out->connections_back.push_back(this);
        this->isConnected = true;
    }
    else
    {
        std::cout << "Already Connected" << std::endl;
    }
}

Neuron::Neuron(NeuronType type, Activation activation, bool has_cache)
{
    this->value = 0;
    this->bias = 0;
    this->type = type;

    if (has_cache)
    {
        this->rewriteCache = std::make_shared<bool>(true);
    }

    else
    {
        this->rewriteCache = nullptr;
    }

    this->activation = activation;
    this->connections_forward = std::vector<connection::Connection*>();
    this->connections_back = std::vector<connection::Connection*>();
}

double Neuron::calculate()
{
    // skip computation if the neuron is input
    if (this->type != NeuronType::Input)
    {
        // compute if rewriteCache is set
        if (this->rewriteCache != nullptr)
        {
            // compute if rewriteCache is true
            if (*this->rewriteCache == true)
            {
                // reset the storages
                this->weightedSumCache = 0; 

                this->value = this->calculateSum();

                // cache the sum
                this->weightedSumCache = this->value;

                // activate the sum
                this->value = this->activate(this->value);
            }
        }

        else
        {
            this->value = this->calculateSum();

            // activate the sum
            this->value = this->activate(this->value);
        }
    }

    return this->value;
}

double Neuron::activate(double value)
{
    switch(this->activation)
    {
        case neuron::Activation::None:
            return this->value;

        case neuron::Activation::Sigmoid:
            return 1 / (1 + exp(-value));

        case neuron::Activation::Binary:
            return (double)value == 1;
        
        default:
            return 0;
    }

}

double Neuron::calculateSum()
{
    double value_cache = 0;

    // calculate the sum
    for (connection::Connection* connection : this->connections_back)
    {
        if(connection->enabled)
        {
            value_cache += connection->in->value * connection->weight;
        }
    }

    value_cache += this->bias;

    return value_cache;
}

double Neuron::recursiveCalculate()
{
    // proceed further at any other neuron
    if (this->type != neuron::NeuronType::Input)
    {
        // check if the neuron has a cache
        if (this->rewriteCache != nullptr)
        {

            // calculate if the neuron has to cache
            if (*this->rewriteCache == true)
            {
                this->value = 0;

                // calculate the sum
                for (connection::Connection* connection : this->connections_back)
                {
                    if (connection->enabled)
                    {
                        this->value += connection->in->recursiveCalculate() * connection->weight;
                    }
                }

                this->value += this->bias;

                this->value = this->activate(this->value);

                this->rewriteCache = std::make_shared<bool>(false);
            }
        }

        // calculate when there is no cache
        else
        {
            this->value = 0;

            // calculate the sum
            for (connection::Connection* connection : this->connections_back)
            {
                if (connection->enabled)
                {
                    this->value += connection->in->recursiveCalculate() * connection->weight;
                }
            }

            this->value += this->bias;

            this->value = this->activate(this->value);
        }
    }

    return this->value;
}
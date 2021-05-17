#include "Neuron.h"

using namespace neuron;
using namespace connection;

Connection::Connection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out)
{
    this->enabled = true;
    this->in = in;
    this->out = out;
    this->innovationNumber = 0;
    this->weight = std::rand() / RAND_MAX * 2 - 1;
    this->isConnected = false;
    this->configureConnectedNeurons();
}

Connection::Connection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out, int innovationNumber)
{
    this->enabled = true;
    this->in = in;
    this->out = out;
    this->innovationNumber = innovationNumber;
    this->weight = std::rand() / RAND_MAX * 2 - 1;
    this->isConnected = false;
    this->configureConnectedNeurons();
}

void Connection::configureConnectedNeurons()
{
    if (isConnected == false)
    {
        this->in->connections_forward.push_back(std::make_shared<connection::Connection>(*this));
        this->out->connections_back.push_back(std::make_shared<connection::Connection>(*this));
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
    this->connections_forward = std::vector<std::shared_ptr<connection::Connection>>();
    this->connections_back = std::vector<std::shared_ptr<connection::Connection>>();
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
        
        default:
            return 0;
    }
}

double Neuron::calculateSum()
{
    double value_cache = 0;

    // calculate the sum
    for (std::shared_ptr<connection::Connection> connection : this->connections_back)
    {
       value_cache += connection->in->value * connection->weight;
    }

    value_cache += this->bias;

    return value_cache;
}
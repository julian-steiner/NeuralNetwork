#include "Neuron.h"

using namespace neuron;
using namespace connection;

Connection::Connection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out)
{
    this->enabled = true;
    this->in = in;
    this->out = out;
    this->innovationNumber = 0;
    this->weight = 0;
}

Connection::Connection(std::shared_ptr<neuron::Neuron> in, std::shared_ptr<neuron::Neuron> out, int innovationNumber)
{
    this->enabled = true;
    this->in = in;
    this->out = out;
    this->innovationNumber = innovationNumber;
    this->weight = 0;
}

void Connection::configureConnectedNeurons()
{
    this->in->connections_forward.push_back(std::make_shared<connection::Connection>(*this));
    this->out->connections_back.push_back(std::make_shared<connection::Connection>(*this));
}

Neuron::Neuron(NeuronType type, Activation activation)
{
    this->value = 0;
    this->bias = 0;
    this->type = type;
    this->activation = activation;
    this->connections_forward = std::vector<std::shared_ptr<connection::Connection>>();
    this->connections_back = std::vector<std::shared_ptr<connection::Connection>>();
}
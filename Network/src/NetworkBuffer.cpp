#include "NetworkBuffer.h"
#include "NeuralNetwork.h"

using namespace nn;

NetworkBuffer::NetworkBuffer()
{
    previousLayerSize = 0;
    currentLayerNumber = 0;
    this->connections = new std::vector<connection::Connection*>();
    this->neurons = new std::vector<neuron::Neuron*>();
    this->layers = new std::vector<nn::Layer*>();
}

NetworkBuffer::~NetworkBuffer()
{
    if (this->connections != nullptr)
    {
        for (connection::Connection* c_connection : *this->connections)
        {
            delete c_connection;
        }
    }

    if (this->neurons != nullptr)
    {
        for (neuron::Neuron* c_neuron : *this->neurons)
        {
            delete c_neuron;
        }
    }

    if (this->connections != nullptr)
    {
        for (nn::Layer* c_layer : *this->layers)
        {
            delete c_layer;
        }
    }
}

NetworkBuffer::NetworkBuffer(const NetworkBuffer& other)
{
    nn::copyNetwork(this, &other);
}

NetworkBuffer::NetworkBuffer(NetworkBuffer&& other)
{
    this->currentLayerNumber = other.currentLayerNumber;
    this->previousLayerSize = other.previousLayerSize;

    this->inputLayer = other.inputLayer;
    this->outputLayer = other.outputLayer;

    this->connections = other.connections;
    this->layers = other.layers;
    this->neurons = other.neurons;

    other.connections = nullptr;
    other.neurons = nullptr;
    other.neurons = nullptr;
}

NetworkBuffer& NetworkBuffer::operator=(const NetworkBuffer& other)
{
    if(&other!=this)
    {
        nn::copyNetwork(this, &other);
    }

    return *this;
}

void NetworkBuffer::addConnection(neuron::Neuron* in, neuron::Neuron* out, int innovationNumber, connection::NeuronLocation inNeuronLocation, connection::NeuronLocation outNeuronLocation)
{
    this->connections->push_back(new connection::Connection(in, out, innovationNumber, inNeuronLocation, outNeuronLocation));
}

void NetworkBuffer::connect(connection::NeuronLocation inNeuronLocation, connection::NeuronLocation outNeuronLocation, int innovationNumber)
{
    neuron::Neuron* in = this->layers->at(inNeuronLocation.layer)->neurons.at(inNeuronLocation.number);
    neuron::Neuron* out = this->layers->at(outNeuronLocation.layer)->neurons.at(outNeuronLocation.number);
    addConnection(in, out, innovationNumber, inNeuronLocation, outNeuronLocation);

    // Add the connection to the connectionDummies in the layer (only to the out to prevent redundancy)
    this->layers->at(out->layerNumber)->connectionDummys.emplace_back(connection::ConnectionDummy(inNeuronLocation, outNeuronLocation, innovationNumber));
}

void NetworkBuffer::addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber)
{
    this->neurons->push_back(new neuron::Neuron(type, activation, true, layerNumber));
    this->layers->at(layerNumber)->neurons.push_back(this->neurons->back());
}

void NetworkBuffer::addNeuron(neuron::NeuronType type, neuron::Activation activation, int layerNumber, double bias)
{
    this->neurons->push_back(new neuron::Neuron(type, activation, bias, true, true, layerNumber));
    this->layers->at(layerNumber)->neurons.push_back(this->neurons->back());
}

void NetworkBuffer::addNeuron(neuron::Neuron&& neuron, int layerNumber)
{
    neuron.layerNumber = layerNumber;
    this->neurons->push_back(&neuron);
    this->layers->at(layerNumber)->neurons.push_back(this->neurons->back());
}


void NetworkBuffer::addLayer(int numNeurons, neuron::Activation activation, LayerType layerType, LayerConnectionType layerConnectionType)
{
    //Determining the layerType out of the network size
    //If the Network size is 0 then the layer has to be input

    int currentNetworkSize = this->neurons->size();
    neuron::NeuronType neuronType = neuron::NeuronType::Hidden;

    //Reserve the space in the Neurons Vector to minimize copying
    this->neurons->reserve(this->neurons->size() + numNeurons);

    if (currentNetworkSize == 0)
    {
        neuronType = neuron::NeuronType::Input;
        layerType = LayerType::Input;
    }

    //Add and reserve space in the layers vector
    switch (layerType)
    {
        case LayerType::Input:
            this->layers->emplace_back(new InputLayer(numNeurons, activation, layerConnectionType));
            this->inputLayer = (nn::InputLayer*) this->layers->at(currentLayerNumber);
            break;
        case LayerType::Output:
            this->layers->emplace_back(new OutputLayer(numNeurons, activation, layerConnectionType));
            this->outputLayer = (nn::OutputLayer*) this->layers->at(currentLayerNumber);
            break;
        case LayerType::Hidden:
            this->layers->emplace_back(new HiddenLayer(numNeurons, activation, layerConnectionType));
            break;
        case LayerType::CustomConnectedHidden:
            this->layers->emplace_back(new CustomConnectedHiddenLayer(numNeurons, activation, layerConnectionType));
            break;
    }

    this->layers->at(currentLayerNumber)->neurons.reserve(numNeurons);

    //Adding the neurons to the network

    for(int i = 0; i < numNeurons; i++)
    {
        addNeuron(neuronType, activation, currentLayerNumber);
    }

    //Connecting the layer if it should be
    switch (layerConnectionType)
    {
        case LayerConnectionType::FullyConnected:
            if(neuronType != neuron::NeuronType::Input)
            {
                //Reserving the space in the connections Vector
                this->connections->reserve(this->connections->size() + (numNeurons * previousLayerSize));

                for(int i = 0; i < this->layers->at(currentLayerNumber)->getSize(); i++)
                {
                    for(int a = 0; a < this->layers->at(currentLayerNumber-1)->getSize(); a++)
                    {
                        connect({currentLayerNumber-1, a}, {currentLayerNumber, i});
                    }
                }
            }
    }
    this->previousLayerSize = numNeurons;
    this->currentLayerNumber ++;
}

bool nn::NetworkBuffer::RecursivelyCheckForRecursion(neuron::Neuron* currentNeuron, neuron::Neuron* targetNeuron)
{
    if (currentNeuron == targetNeuron)
    {
        return true;
    }

    for (connection::Connection* currentConnection : currentNeuron->connections_back)
    {
        if (RecursivelyCheckForRecursion(currentConnection->in, targetNeuron))
        {
            return true;
        }
    }

    return false;
}

bool nn::NetworkBuffer::checkForRecursion(connection::NeuronLocation inNeuronLocation, connection::NeuronLocation outNeuronLocation)
{
    return RecursivelyCheckForRecursion(this->layers->at(inNeuronLocation.layer)->neurons.at(inNeuronLocation.number), this->layers->at(outNeuronLocation.layer)->neurons.at(outNeuronLocation.number));
}

template<typename T>
T nn::NetworkBuffer::getCopy()
{
    T tempNetworkBuffer;

    nn::copyNetwork(&tempNetworkBuffer, this);

    return std::move(tempNetworkBuffer);
}

template nn::NetworkBuffer nn::NetworkBuffer::getCopy<nn::NetworkBuffer>();
template nn::NeuralNetwork nn::NetworkBuffer::getCopy<nn::NeuralNetwork>();

std::stringstream nn::NetworkBuffer::drawScheme(neuron::Neuron* targetNeuron)
{
    std::stringstream output;

    std::vector<neuron::Neuron*>::iterator it = std::find_if(neurons->begin(), neurons->end(), [targetNeuron](const neuron::Neuron* n){return n == targetNeuron;});
    int distance = std::distance(neurons->begin(), it);

    if(targetNeuron->rewriteCache == true)
    {
        targetNeuron->rewriteCache = false;
        output << distance;

        if(targetNeuron->connections_forward.size() != 0)
        {
            if (targetNeuron->connections_forward.size() > 1) output << " -> {";
            else output << " -> ";

            for(int i = 0; i < targetNeuron->connections_forward.size(); i++)
            {
                connection::Connection* currentConnection = targetNeuron->connections_forward.at(i);
                if (currentConnection->enabled)
                {
                    output << drawScheme(currentConnection->out).str();
                    if(i != targetNeuron->connections_forward.size() - 1)
                    {
                        output << ", ";
                    }
                }
            }

            if (targetNeuron->connections_forward.size() > 1) output << "}";
        }
    }

    else
    {
        output << distance;
    }

    return output;
}

std::vector<std::string> nn::NetworkBuffer::getConnectionScheme()
{
    std::vector<std::string> output;

    // Using the rewriteCache variable to see if it has already been iterated over
    for(int i = 0; i < this->neurons->size(); i++)
    {
        this->neurons->at(i)->rewriteCache = true;
    }

    for(int i = 0; i < this->inputLayer->neurons.size(); i++)
    {
        neuron::Neuron* currentNeuron = this->inputLayer->neurons.at(i);
        output.push_back(drawScheme(currentNeuron).str());
    }

    return output;
}

void nn::NetworkBuffer::saveConnectionScheme(const std::string& filename)
{
    std::ofstream document;
    document.open(filename);
    document.clear();

    if (!document)
    {
        std::cout << "Error opening the document" << std::endl;
    }

    std::vector<std::string> connectionScheme = getConnectionScheme();

    document << "\\documentclass{standalone}" << std::endl;
    document << "\\usepackage{tikz}" << std::endl;
    document << "\\usetikzlibrary {graphs}" << std::endl;
    document << "\\begin{document}" << std::endl;
    document << "\\tikz \\graph {" << std::endl;

    for (const std::string& str : connectionScheme)
    {
        document << str << ";" << std::endl;
    }

    document << "};" << std::endl;
    document << "\\end{document}" << std::endl;

    document.close();
}
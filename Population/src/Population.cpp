#include "Population.h"

using namespace population;

Population::Population(const int& size, nn::NeuralNetwork* templateNetwork)
{
    currentInnovationNumber = 0;

    this->networks.reserve(size);
    for (int i = 0; i < size; i++)
    {
        this->networks.push_back(std::move(templateNetwork->getCopy<nn::NeuralNetwork>()));
    }

    // Setting the start of the innovation number to the end of the template network 
    for (connection::Connection* currentConnection: *this->networks.back().connections)
    {
        if (currentConnection->innovationNumber > currentInnovationNumber) 
        {
            currentInnovationNumber = currentConnection->innovationNumber;
        }
    }
}

nn::NeuralNetwork* Population::getNetwork(int number)
{
    return &this->networks.at(number);
}

int Population::getCurrentInnovationNumber()
{
    return currentInnovationNumber;
}

void Population::mutate()
{
    double randomNumber;

    for (nn::NeuralNetwork currentNeuralNetwork : this->networks)
    {
        // handle all the mutations based on a connection
        for (connection::Connection* currentConnection : *currentNeuralNetwork.connections)
        {
            randomNumber = std::rand() / RAND_MAX;

            if (randomNumber < mutationRate)
            {
                weightMutate(currentConnection);
            }
        }

        // handle the addition of a connection
        for (int i = 0; i < 2; i++)
        {
            nn::Layer* currentLayer = currentNeuralNetwork.layers->at(i);
            for (int a = 0; a < currentLayer->neurons.size(); a++)
            {
                neuron::Neuron* currentNeuron = currentNeuralNetwork.neurons->at(a);
                randomNumber = std::rand() / RAND_MAX;

                if (randomNumber < structuralMutationRate)
                {
                    int randomLayerNumber = (int) (std::rand() / RAND_MAX) * 2 + 1;
                    int randomNeuronNumber = (std::rand() / (double)RAND_MAX * currentNeuralNetwork.layers->at(randomLayerNumber)->getSize());
                    bool alreadyConnected = false;
                    
                    // check if the neuron is connected already
                    for (connection::Connection* currentConnection : currentNeuron->connections_forward)
                    {
                        if (currentConnection->inNeuronLocation.layer == randomLayerNumber && currentConnection->inNeuronLocation.number == randomNeuronNumber)
                        {
                            addNeuron(&currentNeuralNetwork, currentConnection);
                            alreadyConnected = true;
                        }
                    }

                    if (alreadyConnected == false)
                    {
                        addConnection(&currentNeuralNetwork, {i, a}, {randomLayerNumber, randomNeuronNumber});
                    }
                }
            }
        }
    }
}

void Population::weightMutate(connection::Connection* target)
{
    target->weight += std::rand() / RAND_MAX * 2 - 1;
}

void Population::addConnection(nn::NeuralNetwork* targetNetwork, connection::NeuronLocation neuron1, connection::NeuronLocation neuron2)
{
    targetNetwork->connect(neuron1, neuron2, this->currentInnovationNumber);
    currentInnovationNumber ++;
}

void Population::addNeuron(nn::NeuralNetwork* targetNetwork, connection::Connection* target)
{
    target->enabled = false;
    targetNetwork->addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1);
    targetNetwork->connect(target->inNeuronLocation, {1, (int)targetNetwork->neurons->size()}, currentInnovationNumber);
    currentInnovationNumber ++;
    targetNetwork->connect({1, (int)targetNetwork->neurons->size()}, target->outNeuronLocation, currentInnovationNumber);
    currentInnovationNumber ++;
}
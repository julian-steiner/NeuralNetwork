#include "Population.h"

using namespace population;

Population::Population(const int& size, nn::NeuralNetwork* templateNetwork)
{
    currentInnovationNumber = 0;

    this->networks = new std::vector<nn::NeuralNetwork>();
    this->networks->reserve(size);
    for (int i = 0; i < size; i++)
    {
        this->networks->push_back(templateNetwork->getCopy<nn::NeuralNetwork>());
    }

    // Setting the start of the innovation number to the end of the template network 
    for (connection::Connection* currentConnection: *this->networks->back().connections)
    {
        if (currentConnection->innovationNumber > currentInnovationNumber) 
        {
            currentInnovationNumber = currentConnection->innovationNumber;
        }
    }
}

Population::~Population()
{
    delete this->networks;
}

nn::NeuralNetwork* Population::getNetwork(int number)
{
    return &this->networks->at(number);
}

int Population::getCurrentInnovationNumber()
{
    return currentInnovationNumber;
}

void Population::mutate()
{
    double randomNumber;

    for(int i = 0; i < this->networks->size(); i++)
    {
        nn::NeuralNetwork* currentNeuralNetwork = getNetwork(i);

        // handle changing of a weight
        randomNumber = std::rand() /  (double)RAND_MAX;
        if (randomNumber < mutationRate)
        {
            randomNumber = round(std::rand() / (double)RAND_MAX * currentNeuralNetwork->connections->size());
            weightMutate(currentNeuralNetwork->connections->at(randomNumber));
        }

        // handle connecting 2 neurons or inserting a node into a connection
        randomNumber = std::rand() /  (double)RAND_MAX;
        if (randomNumber < structuralMutationRate)
        {
            // the -1 making sure that only layers 0 and 1 can be the origin of the connection
            int randomInLayerNumber = round(std::rand() / (double)RAND_MAX * (currentNeuralNetwork->layers->size()-2));
            if(currentNeuralNetwork->layers->at(randomInLayerNumber)->neurons.size() == 0)
            {
                randomInLayerNumber --;
            }
            int randomInNeuronNumber = round(std::rand() / (double)RAND_MAX * (currentNeuralNetwork->layers->at(randomInLayerNumber)->neurons.size()-1));
            neuron::Neuron* inNeuron = currentNeuralNetwork->layers->at(randomInLayerNumber)->neurons.at(randomInNeuronNumber);

            // the +1 making sure that only layers 1 and 2 can be the output of the connection
            int randomOutLayerNumber = round(std::rand() / (double)RAND_MAX * (currentNeuralNetwork->layers->size()-2) + 1);
            if(currentNeuralNetwork->layers->at(randomOutLayerNumber)->neurons.size() == 0)
            {
                randomOutLayerNumber ++;
            }
            int randomOutNeuronNumber = round(std::rand() / (double)RAND_MAX * (currentNeuralNetwork->layers->at(randomOutLayerNumber)->neurons.size()-1));
            neuron::Neuron* outNeuron = currentNeuralNetwork->layers->at(randomOutLayerNumber)->neurons.at(randomOutNeuronNumber);


            // check if the neuron is connected already
            bool alreadyConnected = false;
            for (int a = 0; a < inNeuron->connections_forward.size(); a++)
            {
                connection::Connection* currentConnection = inNeuron->connections_forward.at(a);
                if (currentConnection->outNeuronLocation.layer == randomOutLayerNumber && currentConnection->outNeuronLocation.number == randomOutNeuronNumber)
                {
                    // Add a node to the connection if it is already connected
                    addNeuron(currentNeuralNetwork, currentConnection);
                    alreadyConnected = true;
                }
            }

            // add a connection if it isn't already connected or it would cause recursion
            if (alreadyConnected == false && ((randomInLayerNumber != randomOutLayerNumber) || (randomInNeuronNumber != randomOutNeuronNumber)) && !currentNeuralNetwork->checkForRecursion({randomInLayerNumber, randomInNeuronNumber}, {randomOutLayerNumber, randomOutNeuronNumber}))
            {
                addConnection(currentNeuralNetwork, {randomInLayerNumber, randomInNeuronNumber}, {randomOutLayerNumber, randomOutNeuronNumber});
            }
        }
    }
}

void Population::crossover()
{
    std::vector<nn::NeuralNetwork*> matingPool = std::vector<nn::NeuralNetwork*>();
    std::vector<nn::NeuralNetwork>* nextGeneration = new std::vector<nn::NeuralNetwork>();
    matingPool.reserve(getSize());
    nextGeneration->reserve(getSize());

    double totalFitness = getTotalFitness();

    for (nn::NeuralNetwork& currentNetwork : *this->networks)
    {
        for (int i = 0; i < (int)round((currentNetwork.fitness / totalFitness) * getSize()); i++)
        {
            matingPool.push_back(&currentNetwork);
        }
    }

    for (int i = 0; i < getSize(); i++)
    {
        double randomNumber1 = round((std::rand() / (double)RAND_MAX) * (getSize()-1));
        double randomNumber2 = round((std::rand() / (double)RAND_MAX) * (getSize()-1));

        nextGeneration->push_back(getChild(&this->networks->at(randomNumber1), &this->networks->at(randomNumber2)));
    }

    this->networks = nextGeneration;
}

nn::NeuralNetwork Population::getChild(nn::NeuralNetwork* network1, nn::NeuralNetwork* network2)
{
    nn::NeuralNetwork* higher;
    nn::NeuralNetwork* lower;

    if (network1->fitness >= network2->fitness)
    {
        higher = network1;
        lower = network2;
    }
    
    else
    {
        higher = network2;
        lower = network1;
    }

    nn::NeuralNetwork tempNetwork = higher->getCopy<nn::NeuralNetwork>();

    for (connection::Connection* currentConnection : *higher->connections)
    {
        for (connection::Connection* currentMatching : *lower->connections)
        {
            if (currentMatching->innovationNumber = currentConnection->innovationNumber)
            {
                double randomNumber = std::rand() / (double)RAND_MAX;
                if (randomNumber >= 0.5)
                {
                    currentConnection->weight = currentMatching->weight;
                }
            }
        }
    }

    return std::move(tempNetwork);
}

int Population::getSize()
{
    return this->networks->size();
}

double Population::getTotalFitness()
{
    double totalFitness = 0;
    for (int i = 0; i < getSize(); i++)
    {
        totalFitness += getNetwork(i)->fitness;
    }
    return totalFitness;
}

void Population::weightMutate(connection::Connection* target)
{
    target->weight += (std::rand() / (double)RAND_MAX * 2 - 1) * learningRate;
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
    targetNetwork->connect(target->inNeuronLocation, {1, (int)targetNetwork->layers->at(1)->getSize()-1}, currentInnovationNumber);
    currentInnovationNumber ++;
    targetNetwork->connect({1, (int)targetNetwork->layers->at(1)->getSize()-1}, target->outNeuronLocation, currentInnovationNumber);
    currentInnovationNumber ++;
}
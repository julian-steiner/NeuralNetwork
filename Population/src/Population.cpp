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
        if (randomNumber < weightChangingRate)
        {
            randomNumber = round(std::rand() / (double)RAND_MAX * currentNeuralNetwork->connections->size()-1);
            if (randomNumber >= 0)
            {
                weightMutate(currentNeuralNetwork->connections->at(randomNumber));
            }
        }

        // handle changing of a bias
        randomNumber = std::rand() /  (double)RAND_MAX;
        if (randomNumber < weightChangingRate)
        {
            randomNumber = round(std::rand() / (double)RAND_MAX * currentNeuralNetwork->neurons->size()-1);
            if (randomNumber >= 0)
            {
                biasMutate(currentNeuralNetwork->neurons->at(randomNumber));
            }
        }

        // handle neuron insertions
        randomNumber = std::rand() /  (double)RAND_MAX;
        if (randomNumber < neuronAddingRate)
        {
            randomNumber = round(std::rand() / (double)RAND_MAX * currentNeuralNetwork->connections->size()-1);
            if (randomNumber >= 0)
            {
                addNeuron(currentNeuralNetwork, currentNeuralNetwork->connections->at(randomNumber));
            }
        }

        // handle connecting 2 neurons 
        randomNumber = std::rand() /  (double)RAND_MAX;
        if (randomNumber < connectionAddingRate)
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
    std::vector<nn::NeuralNetwork>* nextGeneration = new std::vector<nn::NeuralNetwork>();
    nextGeneration->reserve(getSize());

    double totalFitness = getTotalFitness();

    for (int i = 0; i < getSize(); i++)
    {
        double randomNumber1 = round((std::rand() / (double)RAND_MAX) * (this->networks->size()-1));
        double randomNumber2 = round((std::rand() / (double)RAND_MAX) * (this->networks->size()-1));

        nextGeneration->push_back(getChild(&this->networks->at(randomNumber1), &this->networks->at(randomNumber2)));
    }

    delete networks;
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

void Population::biasMutate(neuron::Neuron* target)
{
    target->bias += (std::rand() / (double)RAND_MAX * 2 - 1) * learningRate;
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

NetworkComparison Population::compareNetworks(nn::NeuralNetwork* first, nn::NeuralNetwork* second)
{
    NetworkComparison comp = {0, 0, 0};

    // For each connection check the matching genes
    for (int i = 0; i < first->connections->size(); i++)
    {
        connection::Connection* currentConnection = first->connections->at(i);
        for(int a = 0; a < second->connections->size(); a++)
        {
            connection::Connection* correspondingConnection = second->connections->at(a);
            if (correspondingConnection->innovationNumber == currentConnection->innovationNumber)
            {
                comp.matchingGenes++;
                comp.weightDifferences += pow((currentConnection->weight - correspondingConnection->weight), 2);
            }
        }
    }

    // calculating the number of nonMatchingGenes
    comp.nonMatchingGenes = first->connections->size() - comp.matchingGenes + (second->connections->size() - comp.matchingGenes);

    // compute the difference ratio
    comp.differenceRatio = comp.nonMatchingGenes * nonMatchingGenesWeight + comp.weightDifferences * weightDifferenceWeight;
    return std::move(comp);
}

void Population::speciate()
{
    this->species.clear();
    this->species.reserve(numberOfSpecies);

    double maxDifferenceRatio = 0; 

    // determining the highest difference ratio
    for(int i = 0; i < this->networks->size(); i++)
    {
        nn::NeuralNetwork* currentNetwork = &this->networks->at(i);
        double differenceRatio = compareNetworks(&this->networks->at(0), &this->networks->at(i)).differenceRatio;

        if (differenceRatio > maxDifferenceRatio) maxDifferenceRatio = differenceRatio;
    }

    // determining the speciation threshold
    double speciationThreshold = maxDifferenceRatio / numberOfSpecies;

    // copying all the networkPointers into a vector for easier speciation
    std::vector<nn::NeuralNetwork*> networkPointers = std::vector<nn::NeuralNetwork*>();
    networkPointers.reserve(this->networks->size());

    for(int i = 0; i < this->networks->size(); i++)
    {
        networkPointers.push_back(&this->networks->at(i));
    }

    //std::cout << maxDifferenceRatio << std::endl;

    // adding the networks to the different species
    for (int i = 0; i < numberOfSpecies; i++)
    {
        this->species.emplace_back(std::vector<nn::NeuralNetwork*>());

        if(networkPointers.size() == 0)
        {
            continue;
        }

        int randomNumber = round((std::rand() / (double)RAND_MAX) * (networkPointers.size()-1));
        nn::NeuralNetwork* populationPivot = networkPointers.at(randomNumber);
        networkPointers.erase(networkPointers.begin() + randomNumber);

        this->species.at(i).push_back(populationPivot);

        // go through every network and add them to the species if they are similar to the pivot element
        std::vector<nn::NeuralNetwork*>::iterator pointerIt = networkPointers.begin();

        while(pointerIt != networkPointers.end())
        {
            if (compareNetworks(populationPivot, *pointerIt).differenceRatio <= speciationThreshold)
            {
                this->species.at(i).push_back(*pointerIt);
                networkPointers.erase(pointerIt);
                pointerIt--;
            }
            pointerIt++;
        }
    }
}
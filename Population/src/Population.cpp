#include "Population.h"

using namespace population;

Population::Population(const int& size, nn::NeuralNetwork* templateNetwork)
{
    currentInnovationNumber = 0;
    targetNumberOfOrganisms = size;

    this->networks = new std::vector<nn::NeuralNetwork>();
    this->networks->reserve(size);

    // Setting the start of the innovation number to the end of the template network 
    for (connection::Connection* currentConnection: *templateNetwork->connections)
    {
        int innovationNumber;
        assignInnovationNumber({currentConnection->inNeuronLocation, currentConnection->outNeuronLocation, 0}, innovationNumber);
        currentConnection->innovationNumber = innovationNumber;
    }

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

void Population::assignInnovationNumber(const connection::ConnectionDummy& dummy, int& innovationNumber)
{
    bool found = false;
    for(int j = 0; j > connectionDatabase.size(); j++)
    {
        const connection::ConnectionDummy& currentDummy = connectionDatabase.at(j);
        if (currentDummy == dummy)
        {
            found = true;
            innovationNumber = j;
            break;
        }
    }
    if (found == false)
    {
        connectionDatabase.push_back(dummy);
        innovationNumber = connectionDatabase.size() - 1;
    }
}

void Population::mutate()
{
    double randomNumber;

    for(int i = 0; i < this->networks->size(); i++)
    {
        nn::NeuralNetwork* currentNeuralNetwork = getNetwork(i);

        // handle changing of a weight
        randomNumber = randomGenerator.getRandomNumber();
        if (randomNumber < weightChangingRate)
        {
            randomNumber = round(randomGenerator.getRandomNumber() * currentNeuralNetwork->connections->size()-1);
            if (randomNumber >= 0)
            {
                weightMutate(currentNeuralNetwork->connections->at(randomNumber));
            }
        }

        // handle changing of a bias
        randomNumber = randomGenerator.getRandomNumber();
        if (randomNumber < weightChangingRate)
        {
            randomNumber = round(randomGenerator.getRandomNumber() * currentNeuralNetwork->neurons->size()-1);
            if (randomNumber >= 0)
            {
                biasMutate(currentNeuralNetwork->neurons->at(randomNumber));
            }
        }

        // handle neuron insertions
        randomNumber = randomGenerator.getRandomNumber();
        if (randomNumber < neuronAddingRate)
        {
            randomNumber = round(randomGenerator.getRandomNumber() * currentNeuralNetwork->connections->size()-1);
            if (randomNumber >= 0)
            {
                addNeuron(currentNeuralNetwork, currentNeuralNetwork->connections->at(randomNumber));
            }
        }

        // handle connecting 2 neurons 
        randomNumber = randomGenerator.getRandomNumber();
        if (randomNumber < connectionAddingRate)
        {
            // the -1 making sure that only layers 0 and 1 can be the origin of the connection
            int randomInLayerNumber = round(randomGenerator.getRandomNumber() * (currentNeuralNetwork->layers->size()-2));
            if(currentNeuralNetwork->layers->at(randomInLayerNumber)->neurons.size() == 0)
            {
                randomInLayerNumber --;
            }
            int randomInNeuronNumber = round(randomGenerator.getRandomNumber() * (currentNeuralNetwork->layers->at(randomInLayerNumber)->neurons.size()-1));
            neuron::Neuron* inNeuron = currentNeuralNetwork->layers->at(randomInLayerNumber)->neurons.at(randomInNeuronNumber);

            // the +1 making sure that only layers 1 and 2 can be the output of the connection
            int randomOutLayerNumber = round(randomGenerator.getRandomNumber() * (currentNeuralNetwork->layers->size()-2) + 1);
            if(currentNeuralNetwork->layers->at(randomOutLayerNumber)->neurons.size() == 0)
            {
                randomOutLayerNumber ++;
            }
            int randomOutNeuronNumber = round(randomGenerator.getRandomNumber() * (currentNeuralNetwork->layers->at(randomOutLayerNumber)->neurons.size()-1));
            neuron::Neuron* outNeuron = currentNeuralNetwork->layers->at(randomOutLayerNumber)->neurons.at(randomOutNeuronNumber);


            // check if the neuron is connected already
            bool alreadyConnected = false;
            for (int a = 0; a < inNeuron->connections_forward.size(); a++)
            {
                connection::Connection* currentConnection = inNeuron->connections_forward.at(a);
                if (currentConnection->outNeuronLocation.layer == randomOutLayerNumber && currentConnection->outNeuronLocation.number == randomOutNeuronNumber)
                {
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

void Population::speciate()
{
    species.clear();
    // Copying all the pointers of the networks
    std::vector<nn::NeuralNetwork*> networksCopy;
    networksCopy.reserve(networks->size());
    for(int i = 0; i < networks->size(); i++)
    {
        networksCopy.push_back(&networks->at(i));
    }

    // Speciating till all networks have their species
    while(networksCopy.size() > 0)
    {
        // Selecting 1 at random and comparing all to it
        int randomNumber = randomGenerator.getRandomNumber() * (networksCopy.size() - 1);
        std::vector<nn::NeuralNetwork*>::iterator randomNetworkIt = networksCopy.begin() + randomNumber;

        species.emplace_back(Species());
        species.back().networks.push_back(*randomNetworkIt);

        std::vector<nn::NeuralNetwork*>::iterator it = networksCopy.begin();

        while(*it != networksCopy.back())
        {
            if (compareNetworks(*randomNetworkIt, *it).differenceRatio <= speciationThreshold)
            {
                species.back().networks.push_back(*it);
                networksCopy.erase(it);
                it--;
            }
            it ++;
        }

        if(*randomNetworkIt == networksCopy.back()) networksCopy.pop_back();
        else    networksCopy.erase(randomNetworkIt);

    }

    if (species.size() > targetNumberOfSpecies) speciationThreshold += 0.05;
    else if (species.size() < targetNumberOfSpecies) speciationThreshold -= 0.05;
}

void Population::crossover()
{
    std::vector<nn::NeuralNetwork>* nextGeneration = new std::vector<nn::NeuralNetwork>();
    nextGeneration->reserve(getSize() + 2);

    computeChildrenAllowed();

    double totalFitness = getTotalFitness();
    for (Species& currentSpecies: species)
    {
        for (int i = 0; i < currentSpecies.numChildrenAllowed; i++)
        {
            double randomNumber1 = round((std::rand() / (double)RAND_MAX) * (currentSpecies.networks.size()-1));
            double randomNumber2 = round((std::rand() / (double)RAND_MAX) * (currentSpecies.networks.size()-1));

            nextGeneration->push_back(getChild(currentSpecies.networks.at(randomNumber1), currentSpecies.networks.at(randomNumber2)));
        }
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
            if (currentMatching->innovationNumber == currentConnection->innovationNumber)
            {
                double pivotNumber;
                pivotNumber = randomGenerator.getRandomNumber();
                if (pivotNumber >= 0.5)
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
    int innovationNumber;
    assignInnovationNumber({neuron1, neuron2, 0}, innovationNumber);
    targetNetwork->connect(neuron1, neuron2, innovationNumber);
}

void Population::addNeuron(nn::NeuralNetwork* targetNetwork, connection::Connection* target)
{
    int innovationNumber;
    target->enabled = false;
    assignInnovationNumber({target->inNeuronLocation, {1, (int)targetNetwork->layers->at(1)->getSize()-1}, 0}, innovationNumber);

    targetNetwork->addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1);
    targetNetwork->connect(target->inNeuronLocation, {1, (int)targetNetwork->layers->at(1)->getSize()-1}, innovationNumber);

    assignInnovationNumber({{1, (int)targetNetwork->layers->at(1)->getSize()-1}, target->outNeuronLocation, 0}, innovationNumber);
    targetNetwork->connect({1, (int)targetNetwork->layers->at(1)->getSize()-1}, target->outNeuronLocation, innovationNumber);
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
    comp.differenceRatio = (comp.nonMatchingGenes * nonMatchingGenesWeight) + (comp.weightDifferences * weightDifferenceWeight);
    return std::move(comp);
}

double Population::getMaxDifference(nn::NeuralNetwork* reference)
{
    double maxDifferenceRatio = 0; 

    // determining the highest difference ratio
    for(int i = 0; i < this->networks->size(); i++)
    {
        nn::NeuralNetwork* currentNetwork = &this->networks->at(i);
        double differenceRatio = compareNetworks(reference, &this->networks->at(i)).differenceRatio;

        if (differenceRatio > maxDifferenceRatio) maxDifferenceRatio = differenceRatio;
    }

    return maxDifferenceRatio;
}

void Population::computeChildrenAllowed()
{
    // Computing the total fitness
    double totalFitness = 0;
    double totalChildrenAllowed = 0;
    double highestChildrenAllowed = 0;
    Species* biggestSpecies = 0;

    for (Species& currentSpecies : species)
    {
        // Taking the sum of all networks in a generation
        currentSpecies.totalFitness = 0;
        for (nn::NeuralNetwork* currentNetwork : currentSpecies.networks)
        {
            currentNetwork->fitness /= pow(currentSpecies.networks.size(), 2);
            currentSpecies.totalFitness += currentNetwork->fitness;
        }

        // Normalizing the fitness
        totalFitness += currentSpecies.totalFitness;
    }

    for (Species& currentSpecies : species)
    {
        currentSpecies.numChildrenAllowed = round((currentSpecies.totalFitness / totalFitness) * networks->size());

        // Get the network with the most children allowed
        totalChildrenAllowed += currentSpecies.numChildrenAllowed;
        if (currentSpecies.numChildrenAllowed > highestChildrenAllowed) 
        {
            highestChildrenAllowed = currentSpecies.numChildrenAllowed;
            biggestSpecies = &currentSpecies;
        }
    }

    biggestSpecies->numChildrenAllowed += targetNumberOfOrganisms - totalChildrenAllowed;
}

int Population::getNumberOfSpecies()
{
    return species.size();
}
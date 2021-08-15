#include "Population.h"

using namespace population;

double abs(const double& x)
{
    if (x < 0) return x * -1;
    else return x;
}

Population::Population(const int& size, nn::NeuralNetwork* templateNetwork)
{
    currentInnovationNumber = 0;
    targetNumberOfOrganisms = size;

    // Assigning the innovationNumbers
    bool found;
    int innovationNumber;
    for (connection::Connection* currentConnection: *templateNetwork->connections)
    {
        found = false;
        assignInnovationNumber({currentConnection->inNeuronLocation, currentConnection->outNeuronLocation, 0}, found, innovationNumber);

        if (!found)
        {
            connectionDatabase.push_back({currentConnection->inNeuronLocation, currentConnection->outNeuronLocation, 0});
            innovationNumber = connectionDatabase.size() - 1;
        }

        currentConnection->innovationNumber = innovationNumber;
    }

    this->networks = new std::vector<nn::NeuralNetwork>();
    this->networks->reserve(size);
    for (int i = 0; i < size; i++)
    {
        this->networks->push_back(templateNetwork->getCopy<nn::NeuralNetwork>());
    }
}

Population::~Population()
{
    delete this->networks;
}

nn::NeuralNetwork& Population::getNetwork(const int& number)
{
    return this->networks->at(number);
}

const int& Population::getCurrentInnovationNumber()
{
    return currentInnovationNumber;
}

int Population::getNumberOfSpecies()
{
    return species.size();
}

double Population::compareNetworks(nn::NeuralNetwork* network1, nn::NeuralNetwork* network2)
{
    double differenceRatio = 0;
    double weightDifference = 0;

    // Find the number of matching genes
    int matchingGenes = 0;
    for (int i = 0; i < network1->connections->size(); i++)
    {
        for (int a = 0; a < network2->connections->size(); a++)
        {
            if (network1->connections->at(i)->innovationNumber == network2->connections->at(a)->innovationNumber)
            {
                matchingGenes++;

                // adding the weight difference 
                if (network1->connections->at(i)->enabled) weightDifference += abs(network1->connections->at(i)->weight - network2->connections->at(a)->weight);
            }
        }
    }
    if (matchingGenes != 0) weightDifference /= matchingGenes;
    //std::cout << "Matching Genes: " << matchingGenes << std::endl;
    //std::cout << "Weight Difference: " << weightDifference << std::endl;
    //std::cout << "Genome Size: " << network1->connections->size() << std::endl;
    //std::cout << std::endl;

    // Find the number of disjoint genes
    double size1 = network1->connections->size();
    double size2 = network2->connections->size();

    double genomeDifference = (size1 - matchingGenes) + (size2 - matchingGenes);
    
    differenceRatio += weightDifference + genomeDifference;

    return differenceRatio;
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
        int randomNumber = (std::rand() / (double)RAND_MAX) * (networksCopy.size() - 1);
        std::vector<nn::NeuralNetwork*>::iterator randomNetworkIt = networksCopy.begin() + randomNumber;

        species.emplace_back(Species());
        species.back().networks.push_back(*randomNetworkIt);

        std::vector<nn::NeuralNetwork*>::iterator it = networksCopy.begin();

        while(*it != networksCopy.back())
        {
            if (compareNetworks(*randomNetworkIt, *it) <= speciationThreshold)
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

    if (species.size() > targetNumberOfSpecies) speciationThreshold += 0.1;
    else if (species.size() < targetNumberOfSpecies) speciationThreshold -= 0.1;
}

void Population::assignInnovationNumber(const connection::ConnectionDummy& dummy, bool& found, int& innovationNumber)
{
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
}

void Population::addConnection(const connection::ConnectionDummy& dummy, nn::NeuralNetwork& currentNetwork, const bool& found, const int& innovationNumber)
{
    if (found)
    {
        currentNetwork.connect({dummy.inNeuronLocation.layer, dummy.inNeuronLocation.number}, {dummy.outNeuronLocation.layer, dummy.outNeuronLocation.number}, innovationNumber);
    }
    else
    {
        connectionDatabase.push_back(dummy);
        currentNetwork.connect({dummy.inNeuronLocation.layer, dummy.inNeuronLocation.number}, {dummy.outNeuronLocation.layer, dummy.outNeuronLocation.number}, connectionDatabase.size() - 1);
    }
}

void Population::mutate()
{
    for(int i = 0; i < networks->size(); i++)
    {
        nn::NeuralNetwork& currentNetwork = networks->at(i);
        
        // Handling weightChanging
        double randomNumber = std::rand() / (double)RAND_MAX;
        if (randomNumber < weightChangingRate)
        {
            int randomConnectionNumber = (std::rand() / (double)RAND_MAX) * (currentNetwork.connections->size() - 1);
            if (currentNetwork.connections->size() > 0)
            {
                double signNumber = std::rand() / (double)RAND_MAX;
                if (signNumber > 0.5)
                {
                    //currentNetwork.connections->at(randomConnectionNumber)->weight += currentNetwork.connections->at(randomConnectionNumber)->weight * learningRate; 
                    currentNetwork.connections->at(randomConnectionNumber)->weight += learningRate; 
                }
                else currentNetwork.connections->at(randomConnectionNumber)->weight -= learningRate; 
            }
        }

        // Handling connectionAddition
        randomNumber = std::rand() / (double)RAND_MAX;
        if (randomNumber < connectionAddingRate)
        {
            // Giving 20 chances to add a valid connection
            for (int a = 0; a < 20; a++)
            {
                int randomLayerNumber1 = (std::rand() / (double)RAND_MAX) * (currentNetwork.layers->size());
                int randomLayerNumber2 = (std::rand() / (double)RAND_MAX) * (currentNetwork.layers->size());

                // Validate the layer numbers
                if (randomLayerNumber1 > randomLayerNumber2 || (randomLayerNumber1 == randomLayerNumber2 && currentNetwork.layers->at(randomLayerNumber1)->layerType != nn::LayerType::CustomConnectedHidden))
                {
                    continue;
                }
                if (currentNetwork.layers->at(randomLayerNumber1)->getSize() == 0 || currentNetwork.layers->at(randomLayerNumber2)->getSize() == 0)
                {
                    continue;
                }

                // Check if the connection isn't already here
                int randomNeuronNumber1 = (std::rand() / (double)RAND_MAX) * (currentNetwork.layers->at(randomLayerNumber1)->neurons.size());
                int randomNeuronNumber2 = (std::rand() / (double)RAND_MAX) * (currentNetwork.layers->at(randomLayerNumber2)->neurons.size());
                connection::ConnectionDummy target = connection::ConnectionDummy({randomLayerNumber1, randomNeuronNumber1}, {randomLayerNumber2, randomNeuronNumber2}, 0);
                bool alreadyThere = false;
                for (int j = 0; j < currentNetwork.connections->size(); j++)
                {
                    connection::Connection* currentConnection = currentNetwork.connections->at(j);
                    connection::ConnectionDummy dummy = connection::ConnectionDummy(currentConnection->inNeuronLocation, currentConnection->outNeuronLocation, 0);

                    if (dummy == target)
                    {
                        alreadyThere = true;
                        break;
                    }
                }
                if (alreadyThere) continue;

                // Validating the locations
                if (currentNetwork.checkForRecursion({randomLayerNumber1, randomNeuronNumber1}, {randomLayerNumber2, randomNeuronNumber2}))
                {
                    continue;
                }
                
                // Assigning the innovationNumber
                connection::ConnectionDummy dummy = connection::ConnectionDummy({randomLayerNumber1, randomNeuronNumber1}, {randomLayerNumber2, randomNeuronNumber2}, 0);
                bool found = false;
                int innovNumber = 0;
                assignInnovationNumber(dummy, found, innovNumber);

                // Adding the connection
                addConnection(dummy, currentNetwork, found, innovNumber);
                break;
            }
        }

        // Adding a neuron
        randomNumber = std::rand() / (double)RAND_MAX;
        if (randomNumber < neuronAddingRate)
        {
            if (currentNetwork.connections->size() == 0) continue;
            int randomConnectionNumber = (std::rand() / (double)RAND_MAX) * (currentNetwork.connections->size() - 1);
            connection::Connection* currentConnection = currentNetwork.connections->at(randomConnectionNumber);
            currentConnection->enabled = false;
            currentNetwork.addNeuron(neuron::NeuronType::Hidden, neuron::Activation::Sigmoid, 1);
            
            // Assigning the innovationNumber
            connection::ConnectionDummy dummy = connection::ConnectionDummy({currentConnection->inNeuronLocation.layer, currentConnection->inNeuronLocation.number}, {1, currentNetwork.layers->at(1)->getSize() - 1}, 0);
            bool found = false;
            int innovNumber = 0;
            assignInnovationNumber(dummy, found, innovNumber);
            // Adding the connection
            addConnection(dummy, currentNetwork, found, innovNumber);

            // Assigning the innovationNumber
            dummy = connection::ConnectionDummy({1, currentNetwork.layers->at(1)->getSize() - 1}, {currentConnection->outNeuronLocation.layer, currentConnection->outNeuronLocation.number}, 0);
            found = false;
            innovNumber = 0;
            assignInnovationNumber(dummy, found, innovNumber);
            // Adding the connection
            addConnection(dummy, currentNetwork, found, innovNumber);
        }
    }
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
    std::cout << "Adjustment: " << targetNumberOfOrganisms - totalChildrenAllowed << std::endl;
}

void Population::findMatchingConnection(std::vector<connection::Connection*>* connections, const int& innovationNumber, bool& found, double& weight)
{
    for (int i = 0; i < connections->size(); i++)
    {
        if (connections->at(i)->innovationNumber == innovationNumber)
        {
            found = true;
            weight = connections->at(i)->weight;
        }
    }
}

nn::NeuralNetwork Population::getChild(nn::NeuralNetwork* first, nn::NeuralNetwork* second)
{
    nn::NeuralNetwork child;
    if (first->fitness > second->fitness) 
    {
        child = first->getCopy<nn::NeuralNetwork>();
        child.fitness = first->fitness;
    }
    else 
    {
        child = second->getCopy<nn::NeuralNetwork>();
        child.fitness = second->fitness;
    }

    for (int i = 0; i < child.connections->size(); i++)
    {
        connection::Connection* currentConnection = child.connections->at(i);

        bool found1, found2;
        double weight1, weight2;
        findMatchingConnection(first->connections, currentConnection->innovationNumber, found1, weight1);
        findMatchingConnection(second->connections, currentConnection->innovationNumber, found2, weight2);
        if (found1 && found2)
        {
            double randomNumber = std::rand() / (double)RAND_MAX;
            if (randomNumber >= 0.5) currentConnection->weight = weight1;
            else currentConnection->weight = weight2;
        }
    }

    return std::move(child);
}

void Population::crossover()
{
    computeChildrenAllowed();

    std::vector<nn::NeuralNetwork>* newGeneration = new std::vector<nn::NeuralNetwork>();
    // Reserving 2 slots more to prevent the program crashing if it was to add a network too much
    newGeneration->reserve(networks->size() + 2);

    for (Species& currentSpecies : species)
    {
        std::vector<nn::NeuralNetwork*> matingPool;
        for (int i = 0; i < currentSpecies.networks.size(); i++)
        {
            nn::NeuralNetwork* currentNetwork = currentSpecies.networks.at(i);

            for (int a = 0; a <= (currentNetwork->fitness / currentSpecies.totalFitness) * currentSpecies.networks.size(); a++)
            {
                matingPool.push_back(currentNetwork);
            }
        }

        for (int i = 0; i < currentSpecies.numChildrenAllowed; i++)
        {
            int randomNetworkNumber1 = std::rand() / (double)RAND_MAX * (matingPool.size()-1);
            int randomNetworkNumber2 = std::rand() / (double)RAND_MAX * (matingPool.size()-1);

            //newGeneration->push_back(getChild(matingPool.at(randomNetworkNumber1), matingPool.at(randomNetworkNumber2)));
            getChild(matingPool.at(randomNetworkNumber1), matingPool.at(randomNetworkNumber2));
            newGeneration->push_back(matingPool.at(randomNetworkNumber1)->getCopy<nn::NeuralNetwork>());
        }
    }

    delete networks;
    networks = newGeneration;
}

nn::NeuralNetwork* Population::getFittest()
{
    double highest = 0;
    nn::NeuralNetwork* fittest = &networks->at(0);

    for (int i = 0; i < networks->size(); i++)
    {
        if (networks->at(i).fitness > highest)
        {
            highest = networks->at(i).fitness;
            fittest = &networks->at(i);
        }
    }

    return fittest;
}
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

    // Find the number of disjoint genes
    double size1 = network1->connections->size();
    double size2 = network2->connections->size();

    differenceRatio += (size1 - matchingGenes) + (size2 - matchingGenes) + weightDifference;

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
        std::cout << randomNumber << std::endl;
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
}
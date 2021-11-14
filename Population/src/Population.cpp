#include "Population.h"

using namespace population;

Population::Population(const int& size, nn::NeuralNetwork* templateNetwork)
{
    targetNumberOfOrganisms = size;

    // Creating the networks vector
    this->networks = new std::vector<nn::NeuralNetwork>();
    this->networks->reserve(size);

    // Assigning innovation numbers to the network template
    for (connection::Connection* currentConnection: *templateNetwork->connections)
    {
        int innovationNumber;
        assignInnovationNumber({currentConnection->inNeuronLocation, currentConnection->outNeuronLocation, 0}, innovationNumber);
        currentConnection->innovationNumber = innovationNumber;
    }

    // Filling the networks
    for (int i = 0; i < size; i++)
    {
        this->networks->push_back(templateNetwork->getCopy<nn::NeuralNetwork>());
    }
}

Population::Population(const int& size, const int& numInputs, const int& numOutputs)
{
    nn::NeuralNetwork templateANN;
    templateANN.addLayer(numInputs, neuron::Activation::Without, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    templateANN.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    templateANN.addLayer(numOutputs, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::CustomConnected);

    nn::NeuralNetwork* templateNetwork = &templateANN;

    targetNumberOfOrganisms = size;

    // Creating the networks vector
    this->networks = new std::vector<nn::NeuralNetwork>();
    this->networks->reserve(size);

    // Assigning innovation numbers to the network template
    for (connection::Connection* currentConnection: *templateNetwork->connections)
    {
        int innovationNumber;
        assignInnovationNumber({currentConnection->inNeuronLocation, currentConnection->outNeuronLocation, 0}, innovationNumber);
        currentConnection->innovationNumber = innovationNumber;
    }

    // Filling the networks
    for (int i = 0; i < size; i++)
    {
        this->networks->push_back(templateNetwork->getCopy<nn::NeuralNetwork>());
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

void Population::assignInnovationNumber(const connection::ConnectionDummy& dummy, int& innovationNumber)
{
    // Find the connectionDummy in the vector
    std::vector<connection::ConnectionDummy>::iterator it = std::find_if(connectionDatabase.begin(), connectionDatabase.end(), [dummy](const connection::ConnectionDummy& currentDummy){return currentDummy == dummy;});

    if (it == connectionDatabase.end())
    {
        // Add it to the database if it isn't found
        connectionDatabase.push_back(dummy);
        innovationNumber = connectionDatabase.size() - 1;
    }
    else
    {
        // Returning the innovationNumber else
        innovationNumber = std::distance(connectionDatabase.begin(), it);
    }
}

connection::NeuronLocation Population::generateRandomNeuronLocation(nn::NeuralNetwork* currentNeuralNetwork)
{
    int randomLayerNumber = round(randomGenerator.getRandomNumber() * (currentNeuralNetwork->layers->size() - 1));
    int randomNeuronNumber = round(randomGenerator.getRandomNumber() * (currentNeuralNetwork->layers->at(randomLayerNumber)->neurons.size() - 1));
    return {randomLayerNumber, randomNeuronNumber};
}

bool Population::validateConnection(nn::NeuralNetwork* currentNeuralNetwork, const connection::ConnectionDummy& connectionDummy)
{
    // Checking if the location is valid
    if (connectionDummy.inNeuronLocation.number < 0 || connectionDummy.outNeuronLocation.number < 0) return false;
    // Preventing connections in the same layer and backward connections
    if (connectionDummy.inNeuronLocation.layer == connectionDummy.outNeuronLocation.layer && connectionDummy.inNeuronLocation.layer != 1) return false;
    if (connectionDummy.inNeuronLocation.layer > connectionDummy.outNeuronLocation.layer) return false;
    // Checking if it's the same neuron

    // CRITICAL IF STATEMENT PLEASE DEBUG THIS
    //if (connectionDummy.inNeuronLocation.layer == connectionDummy.outNeuronLocation.layer && connectionDummy.inNeuronLocation.number == connectionDummy.outNeuronLocation.number) return false;
    if (connectionDummy.inNeuronLocation.layer == connectionDummy.outNeuronLocation.layer) return false;
    if (connectionDummy.inNeuronLocation.number == connectionDummy.outNeuronLocation.number) return false;

    // Checking if the connection would cause recursion
    if (currentNeuralNetwork->checkForRecursion({connectionDummy.inNeuronLocation.layer,
                                                 connectionDummy.inNeuronLocation.number},
                                                 {connectionDummy.outNeuronLocation.layer,
                                                 connectionDummy.outNeuronLocation.number})) return false;

    // check if the connection already exists
    neuron::Neuron* inNeuron = currentNeuralNetwork->layers->at(connectionDummy.inNeuronLocation.layer)->neurons.at(connectionDummy.inNeuronLocation.number);
    std::vector<connection::Connection*>::iterator it = std::find_if(inNeuron->connections_forward.begin(),
                                                                     inNeuron->connections_forward.end(),
                                                                     [connectionDummy](connection::Connection* currentConnection){return currentConnection->outNeuronLocation.layer == connectionDummy.outNeuronLocation.layer &&
                                                                                                                                         currentConnection->outNeuronLocation.number == connectionDummy.outNeuronLocation.number;});
    if (it != inNeuron->connections_forward.end()) return false;

    return true;
}

void Population::handleWeightMutations(nn::NeuralNetwork* currentNeuralNetwork)
{
    for (int i = 0; i < currentNeuralNetwork->connections->size(); i++)
    {
        if (randomGenerator.getRandomNumber() < weightChangingRate)
        {
            connection::Connection* currentConnection = currentNeuralNetwork->connections->at(i);
            weightMutate(currentConnection);
        }
    }
}

void Population::handleBiasMutations(nn::NeuralNetwork* currentNeuralNetwork)
{
    //if (randomGenerator.getRandomNumber() < weightChangingRate)
    //{
        //double randomNumber = round(randomGenerator.getRandomNumber() * currentNeuralNetwork->neurons->size()-1);
        //if (randomNumber >= 0)
        //{
            //biasMutate(currentNeuralNetwork->neurons->at(randomNumber));
        //}
    //}
    for (int i = 0; i < currentNeuralNetwork->neurons->size(); i++)
    {
        if (randomGenerator.getRandomNumber() < weightChangingRate)
        {
            neuron::Neuron* currentNeuron = currentNeuralNetwork->neurons->at(i);
            biasMutate(currentNeuron);
        }
    }
}

void Population::handleConnectionMutations(nn::NeuralNetwork* currentNeuralNetwork)
{
    if (randomGenerator.getRandomNumber() < connectionAddingRate)
    {
        // Giving 20 chances to find a valid connection
        for (int i = 0; i < 20; i++)
        {
            // Generating random locations
            connection::NeuronLocation inputNeuron = generateRandomNeuronLocation(currentNeuralNetwork);
            connection::NeuronLocation outputNeuron = generateRandomNeuronLocation(currentNeuralNetwork);
            
            // Validating the connection
            connection::ConnectionDummy dummy = {inputNeuron, outputNeuron, 0};
            bool connectionIsValid = validateConnection(currentNeuralNetwork, dummy);

            // Add the connection
            if (connectionIsValid) addConnection(currentNeuralNetwork, {inputNeuron.layer, inputNeuron.number}, {outputNeuron.layer, outputNeuron.number});
        }
    }

}

void Population::handleNeuronMutations(nn::NeuralNetwork* currentNeuralNetwork)
{
    if (randomGenerator.getRandomNumber() < neuronAddingRate)
    {
        double randomNumber = round(randomGenerator.getRandomNumber() * currentNeuralNetwork->connections->size()-1);
        if (randomNumber >= 0)
        {
            addNeuron(currentNeuralNetwork, currentNeuralNetwork->connections->at(randomNumber));
        }
    }
}

void Population::handleNumberMutations(nn::NeuralNetwork* currentNeuralNetwork)
{
    handleWeightMutations(currentNeuralNetwork);
    handleBiasMutations(currentNeuralNetwork);
}

void Population::handleStructuralMutations(nn::NeuralNetwork* currentNeuralNetwork)
{
    handleNeuronMutations(currentNeuralNetwork);
    handleConnectionMutations(currentNeuralNetwork);
}

void Population::mutate()
{
    for(int i = 0; i < this->networks->size(); i++)
    {
        nn::NeuralNetwork* currentNeuralNetwork = getNetwork(i);

        handleNumberMutations(currentNeuralNetwork);

        handleStructuralMutations(currentNeuralNetwork);
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

    nextGeneration->reserve(networks->size()+100);

    computeChildrenAllowed();
    double children = 0;
    double maxAllowed = 0;

    double totalFitness = getTotalFitness();
    for (Species& currentSpecies: species)
    {
        // Adding a mating pool so fitter networks are preferred in the selection
        std::vector<nn::NeuralNetwork*> matingPool;
        matingPool.reserve(currentSpecies.networks.size());

        for (int i = 0; i < currentSpecies.networks.size(); i++)
        {
            nn::NeuralNetwork* currentNetwork = currentSpecies.networks.at(i);
            for (int i = 0; round(i < currentNetwork->fitness / currentSpecies.totalFitness * currentSpecies.networks.size()); i++)
            {
                matingPool.push_back(currentNetwork);
            }
        }

        for (int i = 0; i < currentSpecies.numChildrenAllowed; i++)
        {
            double randomNumber1 = round((std::rand() / (double)RAND_MAX) * (matingPool.size()-1));
            double randomNumber2 = round((std::rand() / (double)RAND_MAX) * (matingPool.size()-1));

            nextGeneration->push_back(getChild(matingPool.at(randomNumber1), matingPool.at(randomNumber2)));
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
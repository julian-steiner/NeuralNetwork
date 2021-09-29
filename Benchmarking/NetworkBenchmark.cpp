#include "Population.h"
#include "Profiler.h"
#include "Timer.h"
#include "Timer.h"

int main()
{
    nn::NeuralNetwork templateNetwork;
    templateNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);
    //templateNetwork.addLayer(0, neuron::Activation::Sigmoid, nn::LayerType::CustomConnectedHidden, nn::LayerConnectionType::CustomConnected);
    templateNetwork.addLayer(2, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
    templateNetwork.addLayer(1, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);
    //templateNetwork.connect({0, 0}, {2, 0});
    //templateNetwork.connect({0, 1}, {2, 0});

    int numberOfNetworks = 200;
    population::Population testPopulation(numberOfNetworks, &templateNetwork);

    testPopulation.weightChangingRate = 0.01;
    testPopulation.connectionAddingRate = 0.0015;
    testPopulation.neuronAddingRate = 0.0001;
    testPopulation.connectionAddingRate = 0;
    testPopulation.neuronAddingRate = 0;
    testPopulation.learningRate = 0.5;

    testPopulation.targetNumberOfSpecies = 4;

    nn::NeuralNetwork* fittest;

    int generation = 0; 
    int counter = 0;
    double highestFitness = 0;

    int numberOfGenerations = 20000;

    bool solution = false;

    while (generation <= numberOfGenerations)
    {
        testPopulation.mutate();
        
        double totalFitness = 0;
        for (int i = 0; i < numberOfNetworks; i++)
        {
            nn::NeuralNetwork* currentNetwork = testPopulation.getNetwork(i);

            double error = 0; 
            double result = 0;

            result = currentNetwork->predict({1, 1}).at(0);
            error += pow(pow(result - 0, 2), 0.5);
            result = currentNetwork->predict({0, 0}).at(0);
            error += pow(pow(result - 0, 2), 0.5);
            result = currentNetwork->predict({1, 0}).at(0);
            error += pow(pow(result - 1, 2), 0.5);
            result = currentNetwork->predict({0, 1}).at(0);
            error += pow(pow(result - 1, 2), 0.5);

            currentNetwork->fitness = 4 - error;

            if (currentNetwork->fitness > highestFitness)
            {
                highestFitness = currentNetwork->fitness;
                fittest = currentNetwork;
            }

            totalFitness += currentNetwork->fitness;
        }

        if(counter == 100)
        {
            std::cout << "Total Fitness: " << totalFitness << "     Generation: " << generation << "    Fittest: " << highestFitness << "   Species: " << testPopulation.getNumberOfSpecies() <<  std::endl;
            counter = 0;
        }
        counter ++;

        generation++;

        if (highestFitness >= 3.99999)
        {
            break;
        }

        highestFitness = 0;

        testPopulation.speciate();

        if(generation < numberOfGenerations)
        {
            testPopulation.crossover();
        }


    }

    std::cout << "Fittest Number of connections: " << fittest->connections->size() << std::endl;
    std::cout << "Fittest Number of neurons: " << fittest->neurons->size() << std::endl;
    std::cout << "Fitness of fittest: " << fittest->fitness << std::endl;

    std::vector<nn::Layer*> layers;
    for (int i = 0; i < fittest->layers->size(); i++)
    {
        layers.push_back(fittest->layers->at(i));
    }

    double result = fittest->predict({1, 1}).at(0);
    std::cout << "With {1, 1}: " << result << std::endl;
    double error = pow(result - 0, 2.0);

    result = fittest->predict({0, 0}).at(0);
    std::cout << "With {0, 0}: " << result << std::endl;
    error += pow(result - 0, 2.0);

    result = fittest->predict({1, 0}).at(0);
    std::cout << "With {1, 0}: " << result << std::endl;
    error += pow(result - 1, 2.0);
    
    result = fittest->predict({0, 1}).at(0);
    std::cout << "With {0, 1}: " << result << std::endl;
    error += pow(result - 1, 2.0);

    fittest->fitness = 4.0 - error;

    std::cout << "Error: " << error << std::endl;
    std::cout << fittest->fitness << std::endl;

    fittest->saveConnectionScheme("Schemes/NeuralNetwork.tex");

    for (connection::Connection* conn : *fittest->connections)
    {
        std::cout << conn->weight << " {" << conn->inNeuronLocation.layer << " " << conn->inNeuronLocation.number << "} " << " ---> " << " {" << conn->outNeuronLocation.layer << " " << conn->outNeuronLocation.number << "} " << std::endl;
    }
}
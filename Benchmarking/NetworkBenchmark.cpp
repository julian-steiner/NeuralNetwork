#include "NeuralNetwork.h"
#include "Profiler.h"
#include "Timer.h"
#include "Timer.h"

int main()
{
    profiling::Instrumentor::Get().BeginSession("Benchmarking");

    PROFILE_SCOPE("Main");
    nn::NeuralNetwork network;
    {
        std::vector<double> inputs;
        std::vector<double> results;
        int size;
        int capacity;

        {
            PROFILE_SCOPE("CreateNetwork");

            network.addLayer(284, neuron::Activation::Sigmoid, nn::LayerType::Input, nn::LayerConnectionType::FullyConnected);

            for (int i = 0; i < 0; i++)
            {
                network.addLayer(284, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
            }
            network.addLayer(16, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
            network.addLayer(16, neuron::Activation::Sigmoid, nn::LayerType::Hidden, nn::LayerConnectionType::FullyConnected);
            network.addLayer(9, neuron::Activation::Sigmoid, nn::LayerType::Output, nn::LayerConnectionType::FullyConnected);

            std::cout << network.connections.size() << std::endl; 
            std::cout << network.neurons.size() << std::endl;
        }

        {
            PROFILE_SCOPE("CreateVector");
            inputs.reserve(284);
            for (int i = 0; i < 284; i++)
            {
                inputs.push_back(i);
            }
        }

        {
            PROFILE_SCOPE("FeedForward");
            results = network.predict(inputs);
        }

        {
            PROFILE_SCOPE("PrintResults");
            for (double value : results)
            {
                std::cout << value << ", ";
            }
            std::cout << std::endl;
        }
    }

    profiling::Instrumentor::Get().EndSession();
}
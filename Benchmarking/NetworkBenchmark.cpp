#include "NeuralNetwork.h"
#include "Timer.h"

int main()
{
    profiling::Instrumentor::Get().BeginSession("Benchmarking");

    PROFILE_SCOPE("Main");
    nn::NeuralNetwork network;
    {
        std::vector<double> inputs;
        std::vector<double> results;

        {
        PROFILE_SCOPE("CreateNetwork");
        network.addLayer(284, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
        network.addLayer(16, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
        network.addLayer(16, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
        network.addLayer(9, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
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
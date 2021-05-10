#include "NeuralNetwork.h"
#include "Timer.h"

int main()
{
    profiling::Instrumentor::Get().BeginSession("Benchmarking");

    PROFILE_SCOPE("Main");
    nn::NeuralNetwork network;
    {
        PROFILE_SCOPE("CreateNetwork");
        network.addLayer(284, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
        network.addLayer(16, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
        network.addLayer(16, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
        network.addLayer(9, neuron::Activation::Sigmoid, nn::LayerType::FullyConnected);
    }

    profiling::Instrumentor::Get().EndSession();
}
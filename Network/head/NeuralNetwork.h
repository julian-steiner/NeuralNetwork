#ifndef NEURALNETWORKHEADER
#define NEURALNETWORKHEADER

#include <vector>
#include <memory>
#include "NetworkBuffer.h"
#include <future>

namespace nn
{
    enum optimizers {SGD};

    class NeuralNetwork : public NetworkBuffer
    {
        public:
        std::vector<double> predict(std::vector<double> inputs);
        //std::vector<double> train(std::vector<std::vector<double>> inputs, int cycles, nn::optimizers optimizer);

        private:
        static void calculateNeuron(neuron::Neuron* neuron);
    };
}

#endif
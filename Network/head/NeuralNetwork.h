#ifndef NEURALNETWORKHEADER
#define NEURALNETWORKHEADER

#include <vector>
#include <memory>
#include "NetworkBuffer.h"

namespace nn
{
    enum optimizers {SGD};

    class NeuralNetwork : public NetworkBuffer
    {
        public:
        std::vector<double> predict(std::vector<double> inputs);
        std::vector<double> train(std::vector<std::vector<double>> inputs, int cycles, nn::optimizers optimizer);
    };
}

#endif
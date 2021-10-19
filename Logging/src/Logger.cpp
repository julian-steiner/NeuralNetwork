#include "Logger.h"

using namespace logging;

Logger::Logger(const std::string& filename) : filename(filename), networksCounter(0)
{
    document.open(filename);
    document.clear();

    if (!document)
    {
        std::cout << "Error opening the document" << std::endl;
    }

    document << "\\documentclass{report}" << std::endl;
    document << "\\usepackage{tikz}" << std::endl;
    document << "\\usetikzlibrary {graphs}" << std::endl;
    document << "\\begin{document}" << std::endl;
}

Logger::~Logger()
{
    if (!document)
    {
        std::cout << "Error opening the document" << std::endl;
    }

    document << "\\end{document}" << "\n";

    document.close();
}

void Logger::addNetwork(nn::NeuralNetwork& network, const int& generation) 
{
    networksCounter ++;
    if (!document)
    {
        std::cout << "Error opening the document" << std::endl;
    }

    document << "\\noindent\\rule{\\textwidth}{1pt}" << "\n";
    document << "Network: " << networksCounter << "\\hspace{7cm}" << "After Generation: " << generation << "\\newline" << "\\newline" << "\n";

    std::vector<std::string> connectionScheme = network.getConnectionScheme();
    document << "\\subsubsection{Structure}" << "\n";
    document << "\\tikz \\graph {" << std::endl;
    for (const std::string& str : connectionScheme)
    {
        document << str << ";" << std::endl;
    }
    document << "};" << "\\newline" << std::endl;

    document << "\\begin{minipage}[t]{0.5\\textwidth}" << "\n";
    document << "\\subsubsection{Weights}" << "\n";
    for (int i = 0; i < network.connections->size(); i++)
    {
        connection::Connection* connection = network.connections->at(i);
        document << "[" << connection->inNeuronLocation.layer <<
                ", " << connection->inNeuronLocation.number << "] $\\rightarrow$ ["
                << connection->outNeuronLocation.layer << ", "
                << connection->outNeuronLocation.number << "] "<<  connection->weight << "\\newline" << "\n";
    }
    document << "\\end{minipage}%" << "\n";

    document << "\\hfill" << "\n";
    document << "\\vrule" << "\n";
    document << "\\hfill" << "\n";

    document << "\\begin{minipage}[t]{0.4\\textwidth}" << "\n";
    document << "\\subsubsection{Bias}" << "\n";
    for (int i = 0; i < network.neurons->size(); i++)
    {
        neuron::Neuron* neuron = network.neurons->at(i);
        document << "Neuron[" << i << "] " << neuron->bias << "\\newline" << "\n";
    }
    document << "\\end{minipage}" << "\n";

    document << "\\newline \n";
    document << "\\newline \n";
}
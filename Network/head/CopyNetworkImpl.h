#ifndef COPYNETWORK
#define COPYNETWORK

namespace nn
{
    template<typename T, typename F>
    void copyNetwork(T* to, F* from)
    {
        // reserve space in the vectors
        to->layers->reserve(from->layers->size());
        to->neurons->reserve(from->neurons->size());
        to->connections->reserve(from->connections->size());

        // Add every layer
        for (int i = 0; i < from->layers->size(); i++)
        {
            nn::Layer* currentLayer = from->layers->at(i);

            // Add the layers
            switch(currentLayer->layerType)
            {
                case(nn::LayerType::Input):
                    to->layers->emplace_back(new nn::InputLayer(currentLayer->getSize(), currentLayer->activation, currentLayer->layerConnectionType));
                    to->inputLayer = (nn::InputLayer*)to->layers->back();
                    break;

                case(nn::LayerType::Output):
                    to->layers->emplace_back(new nn::OutputLayer(currentLayer->getSize(), currentLayer->activation, currentLayer->layerConnectionType));
                    to->outputLayer = (nn::OutputLayer*)to->layers->back();
                    break;

                case(nn::LayerType::Hidden):
                    to->layers->emplace_back(new nn::HiddenLayer(currentLayer->getSize(), currentLayer->activation, currentLayer->layerConnectionType));
                    break;

                case(nn::LayerType::CustomConnectedHidden):
                    to->layers->emplace_back(new nn::CustomConnectedHiddenLayer(currentLayer->getSize(), currentLayer->activation, currentLayer->layerConnectionType));
                    break;
            }

            // reserve space in the vectors
            to->layers->back()->neurons.reserve(currentLayer->neurons.size()); 

            // Add the neurons
            for (neuron::Neuron* currentNeuron : currentLayer->neurons)
            {
                to->addNeuron(currentNeuron->type, currentNeuron->activation, currentNeuron->layerNumber, currentNeuron->bias);
            }
        }

        // Add the connections (here because otherwise you get nullptrs)
        for (int i = 0; i < from->layers->size(); i++)
        {
            nn::Layer* currentLayer = from->layers->at(i);

            // Add the connections
            for (connection::ConnectionDummy currentConnectionDummy : currentLayer->connectionDummys)
            {
                to->connect(currentConnectionDummy.inNeuronLocation, currentConnectionDummy.outNeuronLocation, currentConnectionDummy.innovationNumber);
            }

            for (int a = 0; a < currentLayer->neurons.size(); a++)
            {
                neuron::Neuron* currentNeuron = currentLayer->neurons.at(a);
                neuron::Neuron* correspondingNeuron = to->layers->at(i)->neurons.at(a);

                for (int j = 0; j < currentNeuron->connections_back.size(); j++)
                {
                    connection::Connection* currentConnection = currentNeuron->connections_back.at(j);
                    connection::Connection* correspondingConnection = correspondingNeuron->connections_back.at(j);

                    if(*currentConnection != *correspondingConnection)
                    {
                        for(connection::Connection* tempCorrespondingConnection : correspondingNeuron->connections_back)
                        {
                            if (*tempCorrespondingConnection == *currentConnection)
                            {
                                correspondingConnection = tempCorrespondingConnection;
                                break;
                            }
                        }
                    }

                    correspondingConnection->weight = currentConnection->weight;
                    correspondingConnection->enabled = currentConnection->enabled;
                }
            }
        }

        // Setting other important parameters
        to->previousLayerSize = from->previousLayerSize;
        to->currentLayerNumber = from->currentLayerNumber;
    }
}

#endif
#include "../Neuron/Neuron.h"

#ifndef FIRST_MACHINE_LEARNING_HIDDENNEURON_H
#define FIRST_MACHINE_LEARNING_HIDDENNEURON_H


class HiddenNeuron: public Neuron {
public:
    explicit HiddenNeuron(int inputLength);
    double process(double *arr) override;
    double sigmoid(double summary) override;
};


#endif //FIRST_MACHINE_LEARNING_HIDDENNEURON_H

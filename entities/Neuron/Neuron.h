//
// Created by Matheus Morillo on 3/19/2021.
//

#ifndef FIRST_MACHINE_LEARNING_NEURON_H
#define FIRST_MACHINE_LEARNING_NEURON_H

class Neuron {
protected:
    int inputLength;
    double *weights;
    double bias;

public:
    Neuron(int inputLength);

    double* getWeights();

    double getBias();

    void setBias(double b);
};

#endif //FIRST_MACHINE_LEARNING_NEURON_H

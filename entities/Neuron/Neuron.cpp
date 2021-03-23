#include <iostream>
#include "Neuron.h"

using namespace std;

Neuron::Neuron(int inputLength) {
    this->inputLength = inputLength;
    this->weights = (double *) malloc(sizeof(double) * inputLength);

    for(int i = 0; i < inputLength; i++) {
        this->weights[i] = -1 + (double)(rand()) / ((double)(RAND_MAX/(2)));
    }

    this->bias = -1 + (double)(rand()) / ((double)(RAND_MAX/(2)));
}

double* Neuron::getWeights() {
    return this->weights;
}

double Neuron::getBias() {
    return this->bias;
}

void Neuron::setBias(double b) {
    this->bias = b;
}

void Neuron::amIANeuron() {
    cout << "Yes, Im a neuron!!" << endl;
}
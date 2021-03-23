#include <iostream>
#include <cmath>
#include "HiddenNeuron.h"

using namespace std;

HiddenNeuron::HiddenNeuron(int inputLength): Neuron(inputLength) {}

double HiddenNeuron::process(double *inputs) {
    double summary = this->bias;

    for(int i = 0; i < this->inputLength; i++) {
        summary += this->weights[i] * inputs[i];
    }

    return this->sigmoid(summary);
}

double HiddenNeuron::sigmoid(double summary) {
    return 1.0 / (1.0 + exp(-summary));
}
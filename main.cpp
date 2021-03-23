#include <iostream>
#include <ctime>
#include <cmath>
#include "entities/Neuron/Neuron.h"
#include "entities/HiddenNeuron/HiddenNeuron.h"

using namespace std;

int main() {
    srand(time(nullptr));

    const int INPUT_QTT = 2;
    const int NEURON_QTT = 5;
    const int OUTPUT_QTT = 1;
    const double LEARNING_RATE = 0.5;
    const double MAXIMUM_ERROR = 0.05;

    const double samples[4][3] = {
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0}
    };

    Neuron *neurons = (HiddenNeuron *) malloc(sizeof(HiddenNeuron) * NEURON_QTT);
    Neuron *outputNeurons = (HiddenNeuron *) malloc(sizeof(HiddenNeuron) * OUTPUT_QTT);

    for(int i = 0; i < NEURON_QTT; i++) {
        neurons[i] = HiddenNeuron(INPUT_QTT);
    }

    for(int i = 0; i < OUTPUT_QTT; i++) {
        outputNeurons[i] = HiddenNeuron(NEURON_QTT);
    }

    double inputs[INPUT_QTT];

    double resultPerNeuron[NEURON_QTT];
    double backPropagation[NEURON_QTT];

    double outputs[OUTPUT_QTT];
    double obtained[OUTPUT_QTT];

    double *weights;
    HiddenNeuron *currentNeuron;

    double summarizedErrors;
    double outputDelta;
    double hiddenDelta;

    int totalIterations = 0;

    cout << "Input\tOutput\tObtained" << endl;

    do {
        summarizedErrors = 0.0;

        for(const double *sample : samples) {

            for(int inputColumn = 0; inputColumn < INPUT_QTT; inputColumn++) {
                cout << "" << sample[inputColumn];
                inputs[inputColumn] = sample[inputColumn];
            }

            cout << "\t|";

            for(int outputColumn = 0, sampleIndex = INPUT_QTT; outputColumn < OUTPUT_QTT; outputColumn++, sampleIndex++) {
                cout << " " << sample[sampleIndex];
                outputs[outputColumn] = sample[sampleIndex];
            }

            for(int neuronIndex = 0; neuronIndex < NEURON_QTT; neuronIndex++) {
                currentNeuron = ((HiddenNeuron *) &neurons[neuronIndex]);
                resultPerNeuron[neuronIndex] = currentNeuron->process(inputs);
            }

            cout << "\t|";

            for(int neuronIndex = 0; neuronIndex < OUTPUT_QTT; neuronIndex++) {
                currentNeuron = ((HiddenNeuron *) &outputNeurons[neuronIndex]);
                obtained[neuronIndex] = currentNeuron->process(resultPerNeuron);

                cout << " " << obtained[neuronIndex];
            }

            cout << endl;

            // update output layer weights
            for(int neuronIndex = 0; neuronIndex < OUTPUT_QTT; neuronIndex++) {
                currentNeuron = ((HiddenNeuron *) &outputNeurons[neuronIndex]);
                weights = currentNeuron->getWeights();

                outputDelta
                        = (outputs[neuronIndex] - obtained[neuronIndex])
                          * obtained[neuronIndex] * (1.0 - obtained[neuronIndex]);

                summarizedErrors
                        += pow(outputs[neuronIndex] - obtained[neuronIndex], 2.0);

                currentNeuron->setBias(currentNeuron->getBias() + LEARNING_RATE * outputDelta * 1.0);

                for(int weightIndex = 0; weightIndex < NEURON_QTT; weightIndex++) {
                    weights[weightIndex] = weights[weightIndex] + LEARNING_RATE * outputDelta * resultPerNeuron[weightIndex];

                    backPropagation[weightIndex] = outputDelta * weights[weightIndex];
                }
            }

            // update hidden layer weights
            for(int neuronIndex = 0; neuronIndex < NEURON_QTT; neuronIndex++) {
                currentNeuron = (HiddenNeuron *) &neurons[neuronIndex];
                weights = currentNeuron->getWeights();

                hiddenDelta = resultPerNeuron[neuronIndex] * (1.0 - resultPerNeuron[neuronIndex]) * backPropagation[neuronIndex];

                currentNeuron->setBias(currentNeuron->getBias() + LEARNING_RATE * hiddenDelta * 1.0);

                for(int weightIndex = 0; weightIndex < INPUT_QTT; weightIndex++) {
                    weights[weightIndex] = weights[weightIndex] + LEARNING_RATE * hiddenDelta * resultPerNeuron[weightIndex];
                }
            }
        }

        cout << "Summarized error: " << summarizedErrors << endl;

        ++totalIterations;

    } while (summarizedErrors > MAXIMUM_ERROR);

    cout << "\nTotal of iterations: " << totalIterations << endl;

    return 0;
}

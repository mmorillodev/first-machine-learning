cmake_minimum_required(VERSION 3.17)
project(first_machine_learning)

set(CMAKE_CXX_STANDARD 14)

add_executable(first_machine_learning main.cpp entities/Neuron/Neuron.cpp entities/HiddenNeuron/HiddenNeuron.cpp entities/HiddenNeuron/HiddenNeuron.h)
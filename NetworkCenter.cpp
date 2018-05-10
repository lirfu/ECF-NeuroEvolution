//
// Created by lirfu on 10.05.18..
//


#include "NetworkCenter.h"

NetworkCenter::NetworkCenter(IProblem *problem, DerivativeFunction *hiddenFunction, DerivativeFunction *outputFunction)
        : hiddenFunction_(hiddenFunction), outputFunction_(outputFunction), descendMethod_(new VanillaGradientDescend) {
    problem_ = problem;
    initializer_ = new RandomWeightInitializer(-1, 1);
}

NetworkCenter::~NetworkCenter() {
    delete problem_;
    delete initializer_;
}

double NetworkCenter::testNetwork(std::vector<uint> architecture) {
// Build neural network
    vector<InnerLayer<Matrix> *> layers;
    uint lastSize = problem_->inputSize();
    for (uint val : architecture) {
        layers.push_back(new FullyConnectedLayer<Matrix>(lastSize, val, hiddenFunction_, descendMethod_));
        lastSize = val;
    }
    layers.push_back(new FullyConnectedLayer<Matrix>(
            lastSize, problem_->outputSize(), outputFunction_, descendMethod_));
    NeuralNetwork net(new InputLayer<Matrix>(problem_->inputSize()), layers);

    // Initialize weights
    net.initialize(initializer_);

    // Train NN
    vector<Data *> &data = *problem_->getDataset();
    double loss = 10;
    ulong iteration = 0;
    while (loss > 1e-3 && iteration < 1000) {
        iteration++;
        loss = net.backpropagate(1e-3, data);
//        std::cout << "Iteration " << iteration << " has loss: " << loss << std::endl;
    }

//    std::cout << "Outputs:" << std::endl;
//    for (uint i = 0; i < data.size(); i++) {
//        for (uint j = 0; j < data[i]->testSize(); j++) {
//            std::cout << "Prediction: " << problem_->toLabel(net.getOutput(*data[i]->getValidationInputs()->at(j)))
//                      << "   Label: " << problem_->toLabel(*data[i]->getValidationOutputs()->at(j)) << std::endl;
//        }
//    }

    return loss;
}


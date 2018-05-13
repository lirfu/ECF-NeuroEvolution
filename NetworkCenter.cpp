//
// Created by lirfu on 10.05.18..
//

#include "NetworkCenter.h"

NetworkCenter::NetworkCenter(IProblem *problem, DerivativeFunction *hiddenFunction, DerivativeFunction *outputFunction,
                             double learningRate, double minLoss, uint maxIterations)
        : hiddenFunction_(hiddenFunction), outputFunction_(outputFunction), descendMethod_(new VanillaGradientDescend) {
    problem_ = problem;
    initializer_ = new RandomWeightInitializer(-1, 1);
    learningRate_ = learningRate;
    minLoss_ = minLoss;
    maxIterations_ = maxIterations;
}

NetworkCenter::~NetworkCenter() {
    delete problem_;
    delete initializer_;
}

double NetworkCenter::testNetwork(std::vector<uint> architecture, bool silent) {
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
    double loss = minLoss_ + 1;
    ulong iteration = 0;
    while (loss > minLoss_ && iteration < maxIterations_) {
        iteration++;
        loss = net.backpropagate(learningRate_, data);
        if (!silent)
            std::cout << "Iteration " << iteration << " has loss: " << loss << std::endl;
    }

    return loss;
}


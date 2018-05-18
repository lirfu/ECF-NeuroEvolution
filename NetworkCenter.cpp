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

NeuralNetwork NetworkCenter::buildNetwork(std::vector<uint> &architecture) {
    // Build neural network
    vector<InnerLayer<Matrix> *> layers;
    uint lastSize = problem_->inputSize();
    for (uint val : architecture) {
        layers.push_back(new FullyConnectedLayer<Matrix>(lastSize, val, hiddenFunction_, descendMethod_));
        lastSize = val;
    }
    layers.push_back(new FullyConnectedLayer<Matrix>(
            lastSize, problem_->outputSize(), outputFunction_, descendMethod_));
    return {new InputLayer<Matrix>(problem_->inputSize()), layers};
}

double NetworkCenter::trainNetwork(std::vector<uint> &architecture, bool silent, bool graph) {
    NeuralNetwork net = buildNetwork(architecture);
    return trainNetwork(net, silent, graph);
}

double NetworkCenter::trainNetwork(NeuralNetwork &net, bool silent, bool graph) {
    // Initialize weights
    net.initialize(initializer_);

    if (graph)
        cout << "init 2" << endl;

    // Train NN
    vector<Data *> &data = problem_->getTrainBundle();
    double loss = minLoss_ + 1;
    ulong iteration = 0;
    while (loss > minLoss_ && iteration < maxIterations_) {
        loss = net.backpropagate(learningRate_, data);
        iteration++;

        if (iteration % 1000 == 0) {
            if (!silent) {
                if (graph)
                    std::cout << "echo ";
//                std::cout << "Iteration " << iteration << " has loss: " << loss << std::endl;
            }
            if (graph) {
                cout << "clear" << endl; // Clear the graphs.
                vector<Matrix *> &inputs = problem_->getInputs();
                vector<Matrix *> &outputs = problem_->getOutputs();
                for (uint i = 0; i < inputs.size(); i++) {
                    // Add the target and predicted output values.
                    std::cout << "add " << net.getOutput(*inputs.at(i)).get(0, 0)
                              << " " << outputs.at(i)->get(0, 0) << std::endl;
                }
            }
        }
    }
    if (!silent) {
        if (graph)
            std::cout << "echo ";
//        std::cout << "Final iteration " << iteration << " has loss: " << loss << std::endl;
        std::cout<<iteration<< " "<<loss<<endl;
    }
    return loss;
}


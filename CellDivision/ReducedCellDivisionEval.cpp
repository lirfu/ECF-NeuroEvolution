//
// Created by lirfu on 28.04.18..
//

#include <ecf/ECF_base.h>
#include <ecf/tree/Tree.h>
#include <ecf/FitnessMin.h>
#include <NeuralNetwork.h>
#include <weightinitializers/RandomWeightInitializer.h>
#include "ReducedCellDivisionEval.h"
#include "../problems/IProblem.h"
#include "../problems/XORProblem.h"

ReducedCellDivisionEval::ReducedCellDivisionEval(StateP state) {
    // Construct reduced grammar
    TreeP tree(new Tree::Tree);

    Tree::PrimitiveP parallel(new ParallelSplit);
    tree->addFunction(parallel);
    Tree::PrimitiveP sequential(new SequentialSplit);
    tree->addFunction(sequential);
    Tree::PrimitiveP endBranch(new EndBranch);
    tree->addTerminal(endBranch);

    // Set constructed grammar as genotype
    state->addGenotype(tree);
}

void ReducedCellDivisionEval::registerParameters(StateP state) {
    EvaluateOp::registerParameters(state);

    //TODO
    // Learning rate of the learning algorithm.
//    state->getRegistry()->registerEntry("learningRate", (voidP) (new double(1e-3)), ECF::DOUBLE);
    // Maximum network depth.
//    state->getRegistry()->registerEntry("maxDepth", (voidP) (new uint(1000)), ECF::UINT);

    // Add penalty points to evaluation time (length).
//    state->getRegistry()->registerEntry("penalizeTime", (voidP) (new uint(0)), ECF::UINT);
}

bool ReducedCellDivisionEval::initialize(StateP state) {
    //TODO
//    learningRate_ = *((uint *)state->getRegistry()->getEntry("learningRate").get());
    return true;
}

FitnessP ReducedCellDivisionEval::evaluate(IndividualP p) {
    Tree::Tree *tree = (Tree::Tree *) p->getGenotype().get();
    // Populate architecture using a machine state.
    MachineState state;
    tree->execute(&state);

    FitnessP fitness(new FitnessMin);

    // TODO Load problem (mabye in constructor)
    IProblem *problem = new XORProblem();

    // Build neural network
    shared_ptr<DescendMethod> descendMethod(new VanillaGradientDescend());
    shared_ptr<DerivativeFunction> sigmoid(new Sigmoid());
    shared_ptr<DerivativeFunction> linear(new Linear());
    vector<InnerLayer<Matrix> *> layers;
    uint lastSize = problem->inputSize();
    for (uint i = 0; i < state.architecture.size(); i++) {
        layers.push_back(new FullyConnectedLayer<Matrix>(lastSize, state.architecture[i], sigmoid, descendMethod));
        lastSize = state.architecture[i];
    }
    layers.push_back(new FullyConnectedLayer<Matrix>(lastSize, problem->outputSize(), linear, descendMethod));
    NeuralNetwork net(new InputLayer<Matrix>(problem->inputSize()), layers);

    // Initialize weights
    WeightInitializer *initializer = new RandomWeightInitializer(-1, 1);
    net.initialize(initializer);
    delete initializer;

    // Train NN
    double loss = 10;
    ulong iteration = 0;
    while (loss > 1e-3) {
        iteration++;
        loss = net.backpropagate(1e-3, problem->getDataset());
        std::cout << "Iteration " << iteration << " has loss: " << loss << std::endl;
    }

    fitness->setValue(loss);
    return fitness;
}

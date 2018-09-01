//
// Created by lirfu on 28.04.18..
//

#include <ecf/ECF_base.h>
#include <ecf/tree/Tree.h>
#include <ecf/FitnessMin.h>
#include <NeuralNetwork.h>
#include <cfloat>
#include "ReducedCellDivisionEval.h"

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

FitnessP ReducedCellDivisionEval::evaluate(IndividualP p) {
    Tree::Tree *tree = (Tree::Tree *) p->getGenotype().get();
    // Populate architecture using a machine state.
    MachineState state = {.layer = 0, .architecture=std::vector<uint>()};
    tree->execute(&state);

    // Build neural network
    vector<InnerLayer<Matrix> *> layers;
    uint lastSize = networkCenter_->inputSize();
    for (uint val : state.architecture) {
        layers.push_back(new FullyConnectedLayer<Matrix>(lastSize, val, networkCenter_->hiddenFunction_,
                                                         networkCenter_->descendMethod_));
        lastSize = val;
    }
    layers.push_back(new FullyConnectedLayer<Matrix>(lastSize, networkCenter_->outputSize(), networkCenter_->outputFunction_, networkCenter_->descendMethod_));
    NeuralNetwork network({new InputLayer<Matrix>(networkCenter_->inputSize()), layers});

    // Train and get validation.
    double loss = networkCenter_->trainNetwork(network, true, false);
    if (isnan(loss)) {
        loss = DBL_MAX;
    }
    FitnessP fitness(new FitnessMin);
    fitness->setValue(loss);
    return fitness;
}

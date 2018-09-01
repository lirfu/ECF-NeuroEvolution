//
// Created by lirfu on 28.04.18..
//

#include <ecf/ECF_base.h>
#include <ecf/tree/Tree.h>
#include <ecf/FitnessMin.h>
#include <NeuralNetwork.h>
#include <cfloat>
#include "CellDivisionEval.h"

CellDivisionEval::CellDivisionEval(StateP state) {
    TreeP tree(new Tree::Tree);

    Tree::PrimitiveP parallel(new ParallelSplit);
    tree->addFunction(parallel);
    Tree::PrimitiveP sequential(new SequentialSplit);
    tree->addFunction(sequential);
    Tree::PrimitiveP endBranch(new EndBranch);
    tree->addTerminal(endBranch);

    Tree::PrimitiveP posBias(new PositiveBias);
    tree->addTerminal(posBias);
    Tree::PrimitiveP negBias(new NegativeBias);
    tree->addTerminal(negBias);
    Tree::PrimitiveP zeroBias(new ZeroBias);
    tree->addTerminal(zeroBias);

    Tree::PrimitiveP incIndex(new IncreaseIndex);
    tree->addTerminal(incIndex);
    Tree::PrimitiveP decIndex(new DecreaseIndex);
    tree->addTerminal(decIndex);
    Tree::PrimitiveP posWght(new PositiveWeight);
    tree->addTerminal(posWght);
    Tree::PrimitiveP negWght(new NegativeWeight);
    tree->addTerminal(negWght);
    Tree::PrimitiveP zeroWght(new ZeroWeight);
    tree->addTerminal(zeroWght);
//    Tree::PrimitiveP cut(new CutWeight); //TODO
//    tree->addTerminal(cut);

    // Set constructed grammar as genotype
    state->addGenotype(tree);
}

FitnessP CellDivisionEval::evaluate(IndividualP p) {
    Tree::Tree *tree = (Tree::Tree *) p->getGenotype().get();
    // Populate architecture using a machine state.
    UnstructuredLayer *layer = new UnstructuredLayer(networkCenter_->inputSize(), networkCenter_->outputSize(),
                                                     networkCenter_->hiddenFunction_, networkCenter_->outputFunction_);
    MachineState state = {.neuronIndex = 0, .weightIndex=0, .layer=*layer};
    tree->execute(&state);

    // Build neural network
    vector<InnerLayer<Matrix> *> layers;
    uint lastSize = networkCenter_->inputSize();
    layers.push_back(layer);
    layers.push_back(new FullyConnectedLayer<Matrix>(lastSize, networkCenter_->outputSize(),
                                                     networkCenter_->outputFunction_, networkCenter_->descendMethod_));
    NeuralNetwork network({new InputLayer<Matrix>(networkCenter_->inputSize()), layers});

    // Build, train and validate neural network on given architecture.
    double loss = networkCenter_->trainNetwork(network, true, false);
    if (isnan(loss)) {
        loss = DBL_MAX;
    }
    FitnessP fitness(new FitnessMin);
    fitness->setValue(loss);
    return fitness;
}

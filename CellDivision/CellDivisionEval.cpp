//
// Created by lirfu on 28.04.18..
//

#include <ecf/ECF_base.h>
#include <ecf/tree/Tree.h>
#include <ecf/FitnessMin.h>
#include <NeuralNetwork.h>
#include <cfloat>
#include <utility>
#include "CellDivisionEval.h"
#include "../problems/XORProblem.h"
#include "../problems/RegressionProblem.h"
#include "../problems/WineQualityProblem.h"

CellDivisionEval::CellDivisionEval(StateP state) : ReducedCellDivisionEval() {
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

CellDivisionEval::CellDivisionEval(StateP state, IProblem &problem) : CellDivisionEval(std::move(state)) {
    codeDefinedProblem_ = &problem;
}

FitnessP CellDivisionEval::evaluate(IndividualP p) {
    Tree::Tree *tree = (Tree::Tree *) p->getGenotype().get();
    // Populate architecture using a machine state.
    MachineState state = {.index = 0, .architecture=std::vector<uint>()};
    tree->execute(&state);

//    cout << "Archit: ";
//    for (uint v:state.architecture)
//        cout << v << ", ";
//    cout << endl;

    // Build, train and validate neural network on given architecture.
    double loss = networkCenter_->trainNetwork(state.architecture, true, false);
    if (isnan(loss))
        loss = DBL_MAX;

    FitnessP fitness(new FitnessMin);
    fitness->setValue(loss);
    return fitness;
}

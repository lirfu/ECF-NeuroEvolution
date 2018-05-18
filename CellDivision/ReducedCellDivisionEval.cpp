//
// Created by lirfu on 28.04.18..
//

#include <ecf/ECF_base.h>
#include <ecf/tree/Tree.h>
#include <ecf/FitnessMin.h>
#include <NeuralNetwork.h>
#include <cfloat>
#include <utility>
#include "ReducedCellDivisionEval.h"
#include "../problems/XORProblem.h"
#include "../problems/RegressionProblem.h"
#include "../problems/WineQualityProblem.h"

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

ReducedCellDivisionEval::ReducedCellDivisionEval(StateP state, IProblem &problem) : ReducedCellDivisionEval(
        std::move(state)) {
    codeDefinedProblem_ = &problem;
}

ReducedCellDivisionEval::~ReducedCellDivisionEval() {
    delete networkCenter_;
}

void ReducedCellDivisionEval::registerParameters(StateP state) {
    EvaluateOp::registerParameters(state);

    //TODO
    // Penalize network depth.
//    state->getRegistry()->registerEntry("maxDepth", (voidP) (new uint(1000)), ECF::UINT);
    // Add penalty points to evaluation time (length).
//    state->getRegistry()->registerEntry("penalizeTime", (voidP) (new uint(0)), ECF::UINT);

    // Problem params.
    state->getRegistry()->registerEntry(paramProblem_, (voidP) nullptr, ECF::STRING);
    state->getRegistry()->registerEntry(paramProblemExtra_, (voidP) new std::string("onedim"), ECF::STRING);
    // Network params.
    state->getRegistry()->registerEntry(paramHiddenFunction_, (voidP) new std::string("sigmoid"), ECF::STRING);
    state->getRegistry()->registerEntry(paramOutputFunction_, (voidP) new std::string("linear"), ECF::STRING);
    state->getRegistry()->registerEntry(paramLearningRate_, (voidP) new double(1e-3), ECF::DOUBLE);
    state->getRegistry()->registerEntry(paramMinLoss_, (voidP) new double(0), ECF::DOUBLE);
    state->getRegistry()->registerEntry(paramMaxIterations_, (voidP) new uint(UINT_MAX), ECF::UINT);
}

DerivativeFunction *ReducedCellDivisionEval::strToFun(std::string *str) {
    if (!str) {
        throw runtime_error("Function not defined!"
                                    " Please define activation function for hidden layer and for output layer.");
    } else if (*str == "linear") {
        return new Linear();
    } else if (*str == "sigmoid") {
        return new Sigmoid();
    } else {
        throw runtime_error("Unrecognized function: " + *str);
    }
}

bool ReducedCellDivisionEval::initialize(StateP state) {
    // Problem params
    std::string *problemString = ((std::string *) state->getRegistry()->getEntry(paramProblem_).get());
    std::string *problemExtra = ((std::string *) state->getRegistry()->getEntry(paramProblemExtra_).get());

    // Network params.
    std::string *hiddenFString = ((std::string *) state->getRegistry()->getEntry(paramHiddenFunction_).get());
    std::string *outoutFString = ((std::string *) state->getRegistry()->getEntry(paramOutputFunction_).get());
    double learningRate = *((double *) state->getRegistry()->getEntry(paramLearningRate_).get());
    double minLoss = *((double *) state->getRegistry()->getEntry(paramMinLoss_).get());
    uint maxIter = *((uint *) state->getRegistry()->getEntry(paramMaxIterations_).get());

    try {
        IProblem *problem = codeDefinedProblem_;
        if (!problem) {  // if the problem wasn't defined by code, try using state defined problem.
            if (!problemString) {
                throw runtime_error("Problem not defined!");
            } else if (*problemString == "xor") {
                problem = new XORProblem();
            } else if (*problemString == "function") {
                problem = new RegressionProblem(*problemExtra, 30);
            } else if (*problemString == "wine") {
                problem = new WineQualityProblem(*problemExtra);
            } else {
                throw runtime_error("Unrecognized problem: " + *problemString);
            }
        }

        networkCenter_ = new NetworkCenter(problem, strToFun(hiddenFString), strToFun(outoutFString),
                                           learningRate, minLoss, maxIter);
    } catch (runtime_error &e) {
        std::cerr << "ReducedCellDivisionEval: " << e.what() << std::endl;
    }
    return true;
}

FitnessP ReducedCellDivisionEval::evaluate(IndividualP p) {
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

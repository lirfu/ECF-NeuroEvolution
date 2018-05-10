//
// Created by lirfu on 28.04.18..
//

#include <ecf/ECF_base.h>
#include <ecf/tree/Tree.h>
#include <ecf/FitnessMin.h>
#include <NeuralNetwork.h>
#include "ReducedCellDivisionEval.h"
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

    state->getRegistry()->registerEntry(paramProblem_, (voidP) nullptr, ECF::STRING);
    state->getRegistry()->registerEntry(paramHiddenFunction_, (voidP) new std::string("sigmoid"), ECF::STRING);
    state->getRegistry()->registerEntry(paramOutputFunction_, (voidP) new std::string("sigmoid"), ECF::STRING);

    state->getRegistry()->registerEntry(paramLearningRate_, (voidP) new double(1e-3), ECF::DOUBLE);
    state->getRegistry()->registerEntry(paramMinLoss_, (voidP) new double(0), ECF::DOUBLE);
    state->getRegistry()->registerEntry(paramMaxIterations_, (voidP) new uint(-1), ECF::UINT);
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
    std::string *problemString = ((std::string *) state->getRegistry()->getEntry(paramProblem_).get());
    std::string *hiddenFString = ((std::string *) state->getRegistry()->getEntry(paramHiddenFunction_).get());
    std::string *outoutFString = ((std::string *) state->getRegistry()->getEntry(paramOutputFunction_).get());

    double learningRate = *((double *) state->getRegistry()->getEntry(paramLearningRate_).get());
    double minLoss = *((double *) state->getRegistry()->getEntry(paramMinLoss_).get());
    uint maxIter = *((uint *) state->getRegistry()->getEntry(paramMaxIterations_).get());

    try {
        IProblem *problem = nullptr;
        if (!problemString) {
            throw runtime_error("Problem not defined!");
        } else if (*problemString == "xor") {
            problem = new XORProblem();
        } else if (*problemString == "function") {
            //TODO
        } else if (*problemString == "wine") {
            //TODO
        } else {
            throw runtime_error("Unrecognized problem: " + *problemString);
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
    double loss = networkCenter_->testNetwork(state.architecture);

    FitnessP fitness(new FitnessMin);
    fitness->setValue(loss);
    return fitness;
}

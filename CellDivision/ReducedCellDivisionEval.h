//
// Created by lirfu on 28.04.18..
//

#ifndef NEUROEVOLUTION_REDUCEDCELLDIVISIONEVAL_H
#define NEUROEVOLUTION_REDUCEDCELLDIVISIONEVAL_H


#include <ecf/EvaluateOp.h>
#include "layers/InputLayer.h"
#include "layers/InnerLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "descentmethods/VanillaGradientDescend.h"
#include "functions/Sigmoid.h"
#include "functions/Linear.h"
#include "../problems/IProblem.h"
#include "../NetworkCenter.h"

class ReducedCellDivisionEval : public EvaluateOp {
private:
    /* Parameter constants */
    std::string paramProblem_ = "nev.problem";
    std::string paramProblemExtra_ = "nev.problem.extra";

    std::string paramHiddenFunction_ = "nev.hiddenFunction";
    std::string paramOutputFunction_ = "nev.outputFunction";
    std::string paramLearningRate_ = "nev.learningRate";
    std::string paramMinLoss_ = "nev.minLoss";
    std::string paramMaxIterations_ = "nev.maxIterations";

    DerivativeFunction *strToFun(std::string *);

protected:
    /* Tree primitives */
    class ParallelSplit : public Tree::Primitives::Primitive {
    public:
        ParallelSplit() {
            nArguments_ = 2;
            name_ = "P";
        }

        void execute(void *mState, Tree::Tree &tree) {
            // Execute children without changing the layer index.
            getNextArgument(mState, tree);
            getNextArgument(mState, tree);
        }
    };

    class SequentialSplit : public Tree::Primitives::Primitive {
    public:
        SequentialSplit() {
            nArguments_ = 2;
            name_ = "S";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            // Execute left child in current layer.
            getNextArgument(mState, tree);
            // Execute right child in next layer.
            state.index++;
            getNextArgument(mState, tree);
            // Return to original layer.
            state.index--;
        }
    };

    class EndBranch : public Tree::Primitives::Primitive {
    public:
        EndBranch() {
            nArguments_ = 0;
            name_ = "E";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            // New layer if increased index.
            if (state.index >= state.architecture.size())
                state.architecture.resize(state.architecture.size() + 1);
            // Increase neuron number in this layer.
            state.architecture[state.index]++;
        }
    };

    IProblem *codeDefinedProblem_ = nullptr;

    ReducedCellDivisionEval(){}

public:
    NetworkCenter *networkCenter_;

    typedef struct {
        uint index;
        std::vector<uint> architecture;
    } MachineState;

    /** Constructor.<br> Constructs the tree genotype and sets it to given state. */
    ReducedCellDivisionEval(StateP state);

    /** Constructor.<br> Constructs the tree genotype and sets it to given state and uses the given problem. */
    ReducedCellDivisionEval(StateP state, IProblem &);

    ~ReducedCellDivisionEval();

    /* Inherited methods */

    FitnessP evaluate(IndividualP p) override;

    void registerParameters(StateP p) override;

    bool initialize(StateP p) override;
};


#endif //NEUROEVOLUTION_REDUCEDCELLDIVISIONEVAL_H

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
#include "ANeuralEval.h"

class ReducedCellDivisionEval : public ANeuralEval {
private:
    /* Tree primitives */
    class ParallelSplit : public Tree::Primitives::Primitive {
    public:
        ParallelSplit() {
            nArguments_ = 2;
            name_ = "P";
        }

        void execute(void *mState, Tree::Tree &tree) {
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
            getNextArgument(mState, tree);
            state.layer++;
            getNextArgument(mState, tree);
            state.layer--;
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
            if (state.layer >= state.architecture.size())
                state.architecture.resize(state.architecture.size() + 1);
            state.architecture[state.layer]++;
        }
    };

public:
    typedef struct {
        uint layer;
        std::vector<uint> architecture;
    } MachineState;

    /** Constructor.<br> Constructs the tree genotype and sets it to given state. */
    explicit ReducedCellDivisionEval(StateP state);

    /* Inherited methods */

    FitnessP evaluate(IndividualP p) override;
};


#endif //NEUROEVOLUTION_REDUCEDCELLDIVISIONEVAL_H

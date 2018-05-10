//
// Created by lirfu on 28.04.18..
//

#ifndef NEUROEVOLUTION_EVALUATECELLDIVISION_H
#define NEUROEVOLUTION_EVALUATECELLDIVISION_H


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
    typedef struct {
        uint index;
        std::vector<uint> architecture;
    } MachineState;

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
            // Execute left child in current layer, right child in next layer.
            getNextArgument(mState, tree);
            state.index++;
            getNextArgument(mState, tree);
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

    NetworkCenter *networkCenter_;

    DerivativeFunction* strToFun(std::string *);

public:
    /** Constructor.<br> Constructs the tree genotype and sets it to given state. */
    ReducedCellDivisionEval(StateP state);

    ~ReducedCellDivisionEval();

    /* Inherited methods */

    FitnessP evaluate(IndividualP p) override;

    void registerParameters(StateP p) override;

    bool initialize(StateP p) override;
};


#endif //NEUROEVOLUTION_EVALUATECELLDIVISION_H

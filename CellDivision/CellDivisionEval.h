//
// Created by lirfu on 28.04.18..
//

#ifndef NEUROEVOLUTION_CELLDIVISIONEVAL_H
#define NEUROEVOLUTION_CELLDIVISIONEVAL_H


#include "ReducedCellDivisionEval.h"

class CellDivisionEval : public ReducedCellDivisionEval {
private:
    /* Tree primitives */
    class ParallelSplit : public Tree::Primitives::Primitive {
    public:
        ParallelSplit() {
            nArguments_ = 2;
            name_ = "P";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            mState
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
            state.layer++;
            getNextArgument(mState, tree);
            // Return to original layer.
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
            // New layer if increased index.
            if (state.layer >= state.architecture.size())
                state.architecture.resize(state.architecture.size() + 1);
            // Increase neuron number in this layer.
            state.architecture[state.layer]++;
        }
    };

    class PositiveBias : public Tree::Primitives::Primitive {
    public:
        PositiveBias() {
            nArguments_ = 1;
            name_ = "A";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            // Execute left child in current layer.
            getNextArgument(mState, tree);//TODO
        }
    };

    class ZeroBias : public Tree::Primitives::Primitive {
    public:
        ZeroBias() {
            nArguments_ = 1;
            name_ = "O";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            // Execute left child in current layer.
            getNextArgument(mState, tree);//TODO
        }
    };

    class IncreaseIndex : public Tree::Primitives::Primitive {
    public:
        IncreaseIndex() {
            nArguments_ = 1;
            name_ = "I";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            // Execute left child in current layer.
            getNextArgument(mState, tree);//TODO
        }
    };

    class DecreaseIndex : public Tree::Primitives::Primitive {
    public:
        DecreaseIndex() {
            nArguments_ = 1;
            name_ = "D";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            // Execute left child in current layer.
            getNextArgument(mState, tree);//TODO
        }
    };

    class PositiveWeight : public Tree::Primitives::Primitive {
    public:
        PositiveWeight() {
            nArguments_ = 1;
            name_ = "+";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            // Execute left child in current layer.
            getNextArgument(mState, tree);//TODO
        }
    };

    class NegativeWeight : public Tree::Primitives::Primitive {
    public:
        NegativeWeight() {
            nArguments_ = 1;
            name_ = "-";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            // Execute left child in current layer.
            getNextArgument(mState, tree);//TODO
        }
    };

    class ZeroWeight : public Tree::Primitives::Primitive {
    public:
        ZeroWeight() {
            nArguments_ = 1;
            name_ = "C";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            // Execute left child in current layer.
            getNextArgument(mState, tree);//TODO
        }
    };

    typedef struct {
        int index;
        double value;
    } Pair;

    typedef struct {
        uint currentIndex;
        std::vector<Pair> changes;
    } NeuronSpec;

public:
    typedef struct {
        uint layer;
        std::vector<NeuronSpec> neuronSpecs;
    } MachineState;

    /** Constructor.<br> Constructs the tree genotype and sets it to given state. */
    CellDivisionEval(StateP state);

    /** Constructor.<br> Constructs the tree genotype and sets it to given state and uses the given problem. */
    CellDivisionEval(StateP state, IProblem &);

    ~CellDivisionEval() : ReducedCellDivisionEval::~ReducedCellDivisionEval() {}

    FitnessP evaluate(IndividualP p) override;
};


#endif //NEUROEVOLUTION_CELLDIVISIONEVAL_H

//
// Created by lirfu on 28.04.18..
//

#ifndef NEUROEVOLUTION_CELLDIVISIONEVAL_H
#define NEUROEVOLUTION_CELLDIVISIONEVAL_H


#include <ecf/tree/Primitive.h>
#include "ReducedCellDivisionEval.h"
#include "../UnstructuredLayer.h"

class CellDivisionEval : public ANeuralEval {
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
            UnstructuredLayer::Neuron *n = state.layer.parallelSplitNeuronAt(state.neuronIndex);
            getNextArgument(mState, tree);
            state.layer.addNeuron(n);
            state.neuronIndex++;
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
            UnstructuredLayer::Neuron *n = state.layer.serialSplitNeuronAt(state.neuronIndex);
            getNextArgument(mState, tree);
            state.layer.addNeuron(n);
            state.neuronIndex++;
            getNextArgument(mState, tree);
        }
    };

    class EndBranch : public Tree::Primitives::Primitive {
    public:
        EndBranch() {
            nArguments_ = 0;
            name_ = "E";
        }

        void execute(void *mState, Tree::Tree &tree) {/*do nothing*/}
    };

    class PositiveBias : public Tree::Primitives::Primitive {
    public:
        PositiveBias() {
            nArguments_ = 1;
            name_ = "A";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            (*state.layer.getNeuronWeight(state.neuronIndex, -1u)) = 1;
            getNextArgument(mState, tree);
        }
    };

    class NegativeBias : public Tree::Primitives::Primitive {
    public:
        NegativeBias() {
            nArguments_ = 1;
            name_ = "O";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            (*state.layer.getNeuronWeight(state.neuronIndex, -1u)) = -1;
            getNextArgument(mState, tree);
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
            (*state.layer.getNeuronWeight(state.neuronIndex, -1u)) = 0;
            getNextArgument(mState, tree);
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
            state.weightIndex++;
            getNextArgument(mState, tree);
            state.weightIndex--;
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
            state.weightIndex--;
            getNextArgument(mState, tree);
            state.weightIndex++;
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
            (*state.layer.getNeuronWeight(state.neuronIndex, state.weightIndex)) = 1;
            getNextArgument(mState, tree);
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
            (*state.layer.getNeuronWeight(state.neuronIndex, state.weightIndex)) = -1;
            getNextArgument(mState, tree);
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
            (*state.layer.getNeuronWeight(state.neuronIndex, state.weightIndex)) = 0;
            getNextArgument(mState, tree);
        }
    };

    class CutWeight : public Tree::Primitives::Primitive {
    public:
        CutWeight() {
            nArguments_ = 1;
            name_ = "X";
        }

        void execute(void *mState, Tree::Tree &tree) {
            MachineState &state = *(MachineState *) mState;
            state.layer.cutConnection(state.neuronIndex, state.weightIndex);
            getNextArgument(mState, tree);
        }
    };

public:
    typedef struct {
        uint neuronIndex;
        uint weightIndex;
        UnstructuredLayer &layer;
    } MachineState;

    /** Constructor.<br> Constructs the tree genotype and sets it to given state. */
    explicit CellDivisionEval(StateP state);

    /* Inherited methods */

    FitnessP evaluate(IndividualP p) override;
};


#endif //NEUROEVOLUTION_CELLDIVISIONEVAL_H

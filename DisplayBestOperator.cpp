//
// Created by lirfu on 10.05.18..
//

#include <ecf/ECF_base.h>
#include <ecf/tree/Tree.h>
#include "DisplayBestOperator.h"
#include "CellDivision/ReducedCellDivisionEval.h"

DisplayBestOperator::DisplayBestOperator(NetworkCenter *center) {
    networkCenter_ = center;
}

bool DisplayBestOperator::initialize(StateP p) {
    return Operator::initialize(p);
}

bool DisplayBestOperator::operate(StateP p) {
    std::vector<IndividualP> hof = p->getHoF()->getBest();
    Tree::Tree *tree = (Tree::Tree *) hof[0]->getGenotype().get();

    ReducedCellDivisionEval::MachineState state = {.layer = 0, .architecture=std::vector<uint>()};
    tree->execute(&state);
    networkCenter_->trainNetwork(state.architecture, false, true);
    return true;
}

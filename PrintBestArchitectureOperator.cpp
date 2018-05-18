//
// Created by lirfu on 10.05.18..
//

#include <ecf/ECF_base.h>
#include <ecf/tree/Tree.h>
#include "PrintBestArchitectureOperator.h"
#include "CellDivision/ReducedCellDivisionEval.h"

bool PrintBestArchitectureOperator::operate(StateP p) {
    std::vector<IndividualP> hof = p->getHoF()->getBest();
    Tree::Tree *tree = (Tree::Tree *) hof[0]->getGenotype().get();

    ReducedCellDivisionEval::MachineState state = {.layer = 0, .architecture=std::vector<uint>()};
    tree->execute(&state);

    std::cout << "Best architecture: [";
    bool coldStart = true;
    for (uint v : state.architecture) {
        if (!coldStart) {
            std::cout << ", ";
        }
        std::cout << v;
        coldStart = false;
    }
    std::cout << "]" << std::endl;
    std::cout << tree->toString() << std::endl;

    return true;
}

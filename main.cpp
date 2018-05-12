#include <ecf/ECF.h>
#include "CellDivision/ReducedCellDivisionEval.h"
#include "PrintBestArchitectureOperator.h"

int main(int argc, char **argv) {
    StateP state(new State);
    state->setEvalOp(new ReducedCellDivisionEval(state));

    state->addOperator((PrintBestArchitectureOperatorP) new PrintBestArchitectureOperator);
//    state->addOperator(new DisplayBestOperator);  // TODO Add graph drawing operator

    // initialize and start evaluation
    if (!state->initialize(argc, argv)) {
        std::cerr << "Cannot initialize state!" << std::endl;
        return 1;
    }
    state->run();

//    // after the evolution: show best evolved ant's behaviour on learning trails
//    std::vector<IndividualP> hof = state->getHoF()->getBest();
//    IndividualP ind = hof[0];
//    std::cout << ind->toString();
//    std::cout << "\nBest ant's performance on learning trail(s):" << std::endl;

    return 0;
}

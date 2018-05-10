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
//
//
//    // show ant movement on the trail(s)
//    AntEvalOp::trace = 1;
//
//    // optional: show movements step by step (interactive)
//    //AntEvalOp::step = 1;
//
//    state->getEvalOp()->evaluate(ind);
//
//
//    // also, simulate best evolved ant on a (different) test trail!
//    std::cout << "\nBest ant's performance on test trail(s):" << std::endl;
//    AntEvalOp *evalOp = new AntEvalOp;
//
//    // substitute test trails for learning trails (defined in config file):
//    state->getRegistry()->modifyEntry("learning_trails", state->getRegistry()->getEntry("test_trails"));
//    evalOp->initialize(state);
//    evalOp->evaluate(ind);

    return 0;
}

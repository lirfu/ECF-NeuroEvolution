#include <ecf/ECF.h>
#include "CellDivision/ReducedCellDivisionEval.h"
#include "DirectTest.h"

int main(int argc, char **argv) {
    DirectTest::testAll();

//    StateP state(new State);
//
//    ReducedCellDivisionEval *eval = new ReducedCellDivisionEval(state);
//    state->setEvalOp(eval);
//
//    state->addOperator((PrintBestArchitectureOperatorP) new PrintBestArchitectureOperator);
////    state->addOperator(new DisplayBestOperator);  // TODO Add graph drawing operator
//
//    // initialize and start evaluation
//    if (!state->initialize(argc, argv)) {
//        std::cerr << "Cannot initialize state!" << std::endl;
//        return 1;
//    }
//    state->run();
//
//    // Test results after evolution
//
//    // Get best genotype.
//    std::vector<IndividualP> hof = state->getHoF()->getBest();
//    Tree::Tree *tree = (Tree::Tree *) hof[0]->getGenotype().get();
//
//    // Construct the architecture.
//    ReducedCellDivisionEval::MachineState s = {.layer = 0, .architecture=std::vector<uint>()};
//    tree->execute(&s);
//
//
//    std::cout << "Best architecture: [";
//    bool coldStart = true;
//    for (uint v : s.architecture) {
//        if (!coldStart) {
//            std::cout << ", ";
//        }
//        std::cout << v;
//        coldStart = false;
//    }
//    std::cout << "]" << std::endl;
//    std::cout << tree->toString() << std::endl;
//
//    NetworkCenter &net = *eval->networkCenter_;
//    net.maxIterations_ = 100000;
//    double loss = net.trainNetwork(s.architecture, true, false);
//    std::cout << "Min loss achieved: " << loss << std::endl;

    return 0;
}

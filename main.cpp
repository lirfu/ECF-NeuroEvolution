#include <ecf/ECF.h>
#include "CellDivision/ReducedCellDivisionEval.h"
#include "PrintBestArchitectureOperator.h"
#include "DirectTest.h"

int main(int argc, char **argv) {
//    DirectTest::testAll();

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

    std::vector<IndividualP> hof = state->getHoF()->getBest();
    Tree::Tree *tree = (Tree::Tree *) hof[0]->getGenotype().get();

    ReducedCellDivisionEval::MachineState s = {.index = 0, .architecture=std::vector<uint>()};
    tree->execute(&s);

    std::cout << "Best architecture: [";
    bool coldStart = true;
    for (uint v : s.architecture) {
        if (!coldStart) {
            std::cout << ", ";
        }
        std::cout << v;
        coldStart = false;
    }
    std::cout << "]" << std::endl;
    std::cout << "Fitness: " << hof[0]->getFitness() << std::endl;
    std::cout << tree->toString() << std::endl;

    return 0;
}

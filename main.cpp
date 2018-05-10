#include <ecf/ECF.h>
#include <NeuralNetwork.h>
#include "CellDivision/ReducedCellDivisionEval.h"
#include "problems/IProblem.h"
#include "problems/XORProblem.h"
#include "weightinitializers/RandomWeightInitializer.h"

int main(int argc, char **argv) {
    // TODO Load problem (mabye in constructor)
    IProblem *problem = new XORProblem();

    vector<uint> architecture({4, 3});

    // Build neural network
    shared_ptr<DescendMethod> descendMethod(new VanillaGradientDescend());
    shared_ptr<DerivativeFunction> sigmoid(new Sigmoid());
    shared_ptr<DerivativeFunction> linear(new Linear());
    vector<InnerLayer<Matrix> *> layers;
    uint lastSize = problem->inputSize();
    for (uint i = 0; i < architecture.size(); i++) {
        layers.push_back(new FullyConnectedLayer<Matrix>(lastSize, architecture[i], sigmoid, descendMethod));
        lastSize = architecture[i];
    }
    layers.push_back(new FullyConnectedLayer<Matrix>(lastSize, problem->outputSize(), linear, descendMethod));
    NeuralNetwork net(new InputLayer<Matrix>(problem->inputSize()), layers);

    // Initialize weights
    WeightInitializer *initializer = new RandomWeightInitializer(-1, 1);
    net.initialize(initializer);
    delete initializer;

    // Train NN
    double loss = 10;
    ulong iteration = 0;
    while (loss > 1e-3) {
        iteration++;
        loss = net.backpropagate(1e-3, problem->getDataset());
        std::cout << "Iteration " << iteration << " has loss: " << loss << std::endl;
    }




//    StateP state(new State);
//    state->setEvalOp(new ReducedCellDivisionEval(state));
//
//    // initialize and start evaluation
//    if (!state->initialize(argc, argv)) {
//        std::cerr << "Cannot initialize state!" << std::endl;
//        return 1;
//    }
//    state->run();


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

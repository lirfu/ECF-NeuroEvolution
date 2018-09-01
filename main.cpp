#include <ecf/ECF.h>
#include <data/SimpleData.h>
#include "CellDivision/ReducedCellDivisionEval.h"
#include "UnstructuredLayer.h"

class MyReg : public IProblem {
private:
    SquareLoss<Matrix> loss_;
    vector<Data *> bundle_;
public:
    MyReg() {
        for (int i = -5; i <= 5; i++)
            bundle_.push_back(new SimpleData(new Matrix(1, 1, {i}), new Matrix(1, 1, {i / 5 * sin(i)})));
    }

    uint inputSize() override {
        return 1;
    }

    uint outputSize() override {
        return 1;
    }

    vector<Matrix *> &getInputs() override {
        vector<Matrix *> v;
        return v;//todo
    }

    vector<Matrix *> &getOutputs() override {
        vector<Matrix *> v;
        return v;//todo
    }

    vector<Data *> &getTrainBundle() override {
        return bundle_;
    }

    LossFunction<Matrix> &getLossFunction() override {
        return loss_;
    }

    string toLabel(Matrix &matrix) override {
        return std::to_string(matrix.get(0, 0));
    }
};

int main(int argc, char **argv) {
//    DirectTest::testAll();

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

    shared_ptr<DescendMethod> desc = make_shared<VanillaGradientDescend>(0);
    shared_ptr<DerivativeFunction> f = make_shared<Sigmoid>();
    shared_ptr<DerivativeFunction> fo = make_shared<Linear>();

    uint hiddenSize = 15;

    UnstructuredLayer &l = *new UnstructuredLayer(1, 1, f, fo, false);
    for (uint i = 1; i < hiddenSize; i++) {
        l.addNeuron(l.parallelSplitNeuronAt(0));
//        *l.getNeuronWeight(i, 0) = 1 - rand() % 3;
    }

//    FullyConnectedLayer<Matrix> *fc = new FullyConnectedLayer<Matrix>(1, hiddenSize, f, desc);
//    FullyConnectedLayer<Matrix> *fco = new FullyConnectedLayer<Matrix>(hiddenSize, 1, fo, desc);
//    NeuralNetwork net(new InputLayer<Matrix>(1), {fc, fco});
    NeuralNetwork net(new InputLayer<Matrix>(1), {&l});
    net.initialize(new RandomWeightInitializer(-1, 1));

//    Matrix m(1, 2, {2, 3});
//    Matrix g(1, 1, {1});
//    cout << net.getOutput(m).toString() << endl;
//    l.backwardPass(g, m, 1);
//    l.updateWeights(1);
//    cout << l.toString() << endl;

//    cout << l.toString() << endl;

// Hyperparameters.
    double lr = 0.7;
    double lr_decay = 0.99;
    uint decay_time = 1000;
    uint max_iter = 500000;
    double max_loss = 1e-6;

// Training procedure.
    double loss = max_loss * 2;
    uint iter = 0;
    cout << "init 2" << endl;
    //    IProblem *prob = new MyReg();
    IProblem *prob = new RegressionProblem(RegressionProblem::ONEDIM, 100, true);
    cout << "echo lala " << prob->getTrainBundle().at(0)->trainSize() << " "
         << prob->getTrainBundle().at(0)->validationSize() << endl;

    uint dataNum = 0;
    for (Data *batch:prob->getTrainBundle())
        dataNum += batch->validationSize();
    while (iter++ < max_iter && loss > max_loss) {
        loss = 0;
        for (Data *batch:prob->getTrainBundle()) {
            loss += net.backpropagate(lr, prob->getLossFunction(), *batch);
        }
        loss /= dataNum;
        if (iter % decay_time == 0) {
            cout << "echo Iteration: " << iter << " has loss: " << loss << "  lr: " << lr << endl;
            cout << "clear" << endl;
//            for (uint i = 0; i < prob->getInputs().size(); i++) {
//                cout << "add " << prob->getOutputs().at(i)->get(0, 0);
//                cout << " " << net.getOutput(*prob->getInputs().at(i)).get(0, 0) << endl;
//            }
            for (Data *b:prob->getTrainBundle()) {
                for (uint i = 0; i < b->validationSize(); i++) {
                    cout << "add " << b->getValidationOutputs()->at(i)->get(0, 0)
                         << " " << net.getOutput(*b->getValidationInputs()->at(i)).get(0, 0) << endl;
                }
            }
            lr *= lr_decay;
        }
    }
//    cout << l.toString() << endl;
    return 0;
}

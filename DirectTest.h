//
// Created by lirfu on 13.05.18..
//

#pragma once

#include <data/SimpleData.h>
#include <descentmethods/DescendMethod.h>
#include <descentmethods/VanillaGradientDescend.h>
#include <functions/Sigmoid.h>
#include <layers/FullyConnectedLayer.h>
#include <NeuralNetwork.h>
#include <weightinitializers/RandomWeightInitializer.h>
#include "Utils.h"
#include "UnstructuredLayer.h"
#include "problems/IProblem.h"
#include "problems/RegressionProblem.h"
#include "problems/XORProblem.h"
#include "problems/WineQualityProblem.h"

#define OUTPUT_RESULTS

class DirectTest {
public:
    static void testAll() {
        bool splitDS = false;
        string str("/home/lirfu/PersonalWorkspace/NeuroEvolution/wine_data/winequality-joined.csv");
        std::vector<IProblem *> problems({
//                                                 new RegressionProblem(RegressionProblem::LINREG, 100, splitDS),
//                                                 new RegressionProblem(RegressionProblem::SQUAREPOLY, 100, splitDS),
//                                                 new RegressionProblem(RegressionProblem::ONEDIM, 100, splitDS),
//                                                 new RegressionProblem(RegressionProblem::IMPULSE, 100, splitDS),
                                                 new RegressionProblem(RegressionProblem::ROSENBROCK, 100, splitDS),
//                                                 new XORProblem(),
//                                                 new WineQualityProblem(str)
                                         });

        shared_ptr<DescendMethod> desc = make_shared<VanillaGradientDescend>(0);
        shared_ptr<DerivativeFunction> f = make_shared<Sigmoid>();
        shared_ptr<DerivativeFunction> fo = make_shared<Linear>();
        uint hiddenSize = 50;

        for (IProblem *prob:problems) {
            UnstructuredLayer &l = *new UnstructuredLayer(prob->inputSize(), prob->outputSize(), f, fo, false);
            for (uint i = 1; i < hiddenSize; i++) {
                l.addNeuron(l.parallelSplitNeuronAt(0));
//              *l.getNeuronWeight(i, 0) = 1 - rand() % 3;
            }

//            FullyConnectedLayer<Matrix> *fc = new FullyConnectedLayer<Matrix>(prob->inputSize(), hiddenSize, f, desc);
//            FullyConnectedLayer<Matrix> *fco = new FullyConnectedLayer<Matrix>(hiddenSize, prob->outputSize(), fo, desc);
//            NeuralNetwork net(new InputLayer<Matrix>(prob->inputSize()), {fc, fco});
            NeuralNetwork net(new InputLayer<Matrix>(prob->inputSize()), {&l});

            net.initialize(new RandomWeightInitializer(-0.1, 0.1));
// Hyperparameters.
            double lr = 1e-3;
            double lr_decay = 0.999;
            uint decay_time = 1000;
            uint max_iter = 100000;
            double max_loss = 1e-9;
// Training procedure.
            double loss = max_loss * 2;
            uint iter = 0;
#ifdef OUTPUT_RESULTS
            std::cout << "init 2" << std::endl;
            std::cout << "echo ";
#endif
            std::cout << "Train:Valid size = " << prob->getTrainBundle().at(0)->trainSize()
                      << ":" << prob->getTrainBundle().at(0)->validationSize() << std::endl;
            // Loss normalization.
            uint dataNum = 0;
            for (Data *batch:prob->getTrainBundle())
                dataNum += batch->validationSize();
            // Track time.
            nnpp_utils::Stopwatch st;
            st.start();
            while (iter++ < max_iter && loss > max_loss) {
                loss = 0;
                for (Data *batch:prob->getTrainBundle())
                    loss += net.backpropagate(lr, prob->getLossFunction(), *batch);
                loss /= dataNum;
                if (isnan(loss) || isinf(loss)) break;
                if (iter % decay_time == 0) {
#ifdef OUTPUT_RESULTS
                    std::cout << "echo ";
#endif
                    std::cout << "(" << st.lap() << "s) Iteration: " << iter
                              << " has loss: " << loss << "  lr: " << lr << std::endl;
                    lr *= lr_decay;
#ifdef OUTPUT_RESULTS
                    std::cout << "clear" << std::endl;
//                  for (uint i = 0; i < prob->getInputs().size(); i++) {
//                    std::cout << "add " << prob->getOutputs().at(i)->get(0, 0);
//                    std::cout << " " << net.getOutput(*prob->getInputs().at(i)).get(0, 0) << std::endl;
//                  }
                    for (Data *b:prob->getTrainBundle())
                        for (uint i = 0; i < b->validationSize(); i++)
                            std::cout << "add " << b->getValidationOutputs()->at(i)->get(0, 0)
                                      << " " << net.getOutput(*b->getValidationInputs()->at(i)).get(0, 0) << std::endl;
#endif
                }
            }
#ifdef OUTPUT_RESULTS
            std::cout << "echo " << endl;
#endif
            std::cout << "(" << st.stop() << "s) Final iteration: " << iter << " has loss: "
                      << loss << "  lr: " << lr << std::endl;
        }
    }
};

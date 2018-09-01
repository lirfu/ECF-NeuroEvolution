//
// Created by lirfu on 18.08.18..
//

#ifndef NEUROEVOLUTION_ANEURALEVAL_H
#define NEUROEVOLUTION_ANEURALEVAL_H


#include <ecf/ECF_base.h>
#include <ecf/EvaluateOp.h>
#include "../NetworkCenter.h"
#include "../lib/nnpp/functions/DerivativeFunction.h"
#include "../problems/XORProblem.h"
#include "../problems/RegressionProblem.h"
#include "../problems/WineQualityProblem.h"
#include "../lib/nnpp/functions/Linear.h"
#include "../lib/nnpp/functions/Sigmoid.h"

class ANeuralEval : public EvaluateOp {
private:
    /* Parameter constants */
    std::string paramProblem_ = "nev.problem";
    std::string paramProblemExtra_ = "nev.problem.extra";

    std::string paramHiddenFunction_ = "nev.hiddenFunction";
    std::string paramOutputFunction_ = "nev.outputFunction";
    std::string paramLearningRate_ = "nev.learningRate";
    std::string paramMinLoss_ = "nev.minLoss";
    std::string paramMaxIterations_ = "nev.maxIterations";
protected:
    IProblem *codeDefinedProblem_ = nullptr;

    DerivativeFunction *strToFun(std::string *str) {
        if (!str) {
            throw runtime_error("Function not defined!"
                                        " Please define activation function for hidden layer and for output layer.");
        } else if (*str == "linear") {
            return new Linear();
        } else if (*str == "sigmoid") {
            return new Sigmoid();
        } else {
            throw runtime_error("Unrecognized function: " + *str);
        }
    }

public:
    NetworkCenter *networkCenter_;

    void setProblem(IProblem &problem) {
        codeDefinedProblem_ = &problem;
    }

    ~ANeuralEval() {
        delete networkCenter_;
    }

    /* Inherited methods */
    void registerParameters(StateP state) override {
        EvaluateOp::registerParameters(state);
        // Penalize network depth.
        //    state->getRegistry()->registerEntry("maxDepth", (voidP) (new uint(1000)), ECF::UINT);
        // Add penalty points to evaluation time.
        //    state->getRegistry()->registerEntry("penalizeTime", (voidP) (new uint(0)), ECF::UINT);

        // Problem params.
        state->getRegistry()->registerEntry(paramProblem_, (voidP) nullptr, ECF::STRING);
        state->getRegistry()->registerEntry(paramProblemExtra_, (voidP) new std::string("onedim"), ECF::STRING);
        // Network params.
        state->getRegistry()->registerEntry(paramHiddenFunction_, (voidP) new std::string("sigmoid"), ECF::STRING);
        state->getRegistry()->registerEntry(paramOutputFunction_, (voidP) new std::string("linear"), ECF::STRING);
        state->getRegistry()->registerEntry(paramLearningRate_, (voidP) new double(1e-3), ECF::DOUBLE);
        state->getRegistry()->registerEntry(paramMinLoss_, (voidP) new double(0), ECF::DOUBLE);
        state->getRegistry()->registerEntry(paramMaxIterations_, (voidP) new uint(UINT_MAX), ECF::UINT);
    }

    bool initialize(StateP state) override {
        // Problem params
        std::string *problemString = ((std::string *) state->getRegistry()->getEntry(paramProblem_).get());
        std::string *problemExtra = ((std::string *) state->getRegistry()->getEntry(paramProblemExtra_).get());
        // Network params.
        std::string *hiddenFString = ((std::string *) state->getRegistry()->getEntry(paramHiddenFunction_).get());
        std::string *outoutFString = ((std::string *) state->getRegistry()->getEntry(paramOutputFunction_).get());
        double learningRate = *((double *) state->getRegistry()->getEntry(paramLearningRate_).get());
        double minLoss = *((double *) state->getRegistry()->getEntry(paramMinLoss_).get());
        uint maxIter = *((uint *) state->getRegistry()->getEntry(paramMaxIterations_).get());
        try {
            IProblem *problem = codeDefinedProblem_;
            if (!problem) {  // if the problem wasn't defined by code, try using state defined problem.
                if (!problemString) {
                    throw runtime_error("Problem not defined!");
                } else if (*problemString == "xor") {
                    problem = new XORProblem();
                } else if (*problemString == "function") {
                    if(!problemExtra){
                        throw runtime_error(paramProblemExtra_+" not defined!");
                    }
                    problem = new RegressionProblem((RegressionProblem::FUNC) stoi(*problemExtra), 30, true);
                } else if (*problemString == "wine") {
                    problem = new WineQualityProblem(*problemExtra);
                } else {
                    throw runtime_error("Unrecognized problem: " + *problemString);
                }
            }
            networkCenter_ = new NetworkCenter(problem, strToFun(hiddenFString), strToFun(outoutFString),
                                               learningRate, minLoss, maxIter);
        } catch (runtime_error &e) {
            std::cerr << "ReducedCellDivisionEval: " << e.what() << std::endl;
        }
        return true;
    }
};


#endif //NEUROEVOLUTION_ANEURALEVAL_H

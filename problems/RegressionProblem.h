//
// Created by lirfu on 12.05.18..
//

#ifndef NEUROEVOLUTION_REGRESSIONPROBLEM_H
#define NEUROEVOLUTION_REGRESSIONPROBLEM_H


#include <cmath>
#include "IProblem.h"

class RegressionProblem : public IProblem {
private:
    std::string paramOneDim_ = "onedim";
    std::string paramRosenbrock_ = "rosenbrock";
    std::string paramImpulse_ = "impulse";

    double oneDimensionalFunc(double x) {
        return x * sin(x * x) + 5;
    }

    double Rosenbrock(double x, double y) {
        return (4 - x) * (4 - x) + 2 * (y - x * x) * (y - x * x);
    }

    double sgn(double x) {
        return (x > 0) - (x < 0);
    }

    double impulseFunc(double x) {
        return (sgn(x + 2) - sgn(x - 2)) / 2.;
    }

    uint inputSize_;
    vector<Matrix *> inputs_;
    vector<Matrix *> outputs_;
    vector<Data *> trainBundle_;

public:
    explicit RegressionProblem(string &functionName, uint samplesNum);

    ~RegressionProblem();

    uint inputSize() override {
        return inputSize_;
    }

    uint outputSize() override {
        return 1;
    }

    vector<Matrix *> &getInputs() override {
        return inputs_;
    }

    vector<Matrix *> &getOutputs() override {
        return outputs_;
    }

    vector<Data *> &getTrainBundle() override {
        return trainBundle_;
    }

    string toLabel(Matrix &matrix) override;
};


#endif //NEUROEVOLUTION_REGRESSIONPROBLEM_H

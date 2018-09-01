//
// Created by lirfu on 12.05.18..
//

#ifndef NEUROEVOLUTION_REGRESSIONPROBLEM_H
#define NEUROEVOLUTION_REGRESSIONPROBLEM_H


#include <cmath>
#include <loss/SquareLoss.h>
#include "IProblem.h"

class RegressionProblem : public IProblem {
private:
    SquareLoss<Matrix> lossFunction_;

    double linearFunc(double x) {
        return 2 * x - 1;
    }

    double squarePolyFunc(double x) {
        return 0.1 * x * x + 0.1;
    }

    double oneDimensionalFunc(double x) {
        return 0.2 * sin(x) + 0.2 * sin(4 * x + M_PI / 7) + 0.5;
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
    enum FUNC {
        LINREG, SQUAREPOLY, ONEDIM, ROSENBROCK, IMPULSE
    };

    explicit RegressionProblem(FUNC function, uint samplesNum, bool separateData);

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

    LossFunction<Matrix> &getLossFunction() override;
};


#endif //NEUROEVOLUTION_REGRESSIONPROBLEM_H

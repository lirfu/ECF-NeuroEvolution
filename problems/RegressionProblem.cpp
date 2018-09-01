//
// Created by lirfu on 12.05.18..
//

#include <data/DataUtils.h>
#include <data/SimpleData.h>
#include "RegressionProblem.h"

RegressionProblem::RegressionProblem(FUNC function, uint samplesNum, bool separateData) {
    double trainPercentage;
    switch (function) {
        case LINREG:
            inputSize_ = 1;
            trainPercentage = 0.5;
            for (int i = -(int) samplesNum / 2; i <= (int) samplesNum / 2; i++) {
                inputs_.push_back(new Matrix(1, 1, {(double) i}));
                outputs_.push_back(new Matrix(1, 1, {linearFunc(i)}));
            }
            break;
        case SQUAREPOLY:
            inputSize_ = 1;
            trainPercentage = 0.6;
            for (int i = -(int) samplesNum / 2; i <= (int) samplesNum / 2; i++) {
                inputs_.push_back(new Matrix(1, 1, {(double) i}));
                outputs_.push_back(new Matrix(1, 1, {squarePolyFunc(i)}));
            }
            break;
        case ONEDIM:
            inputSize_ = 1;
            trainPercentage = 0.8;
            for (int i = 0; i < (int) samplesNum; i++) {
                double x = i * 2. * M_PI / samplesNum;
                inputs_.push_back(new Matrix(1, 1, {x}));
                outputs_.push_back(new Matrix(1, 1, {oneDimensionalFunc(x)}));
            }
            break;
        case ROSENBROCK: {
            inputSize_ = 2;
            trainPercentage = 0.7;
            uint axisSamples = static_cast<uint>(sqrt(samplesNum));
            for (int x = 0; x < axisSamples; x++) {
                for (int y = 0; y < axisSamples; y++) {
                    inputs_.push_back(new Matrix(1, 2, {x * 1., y * 1.}));
                    outputs_.push_back(new Matrix(1, 1, {Rosenbrock(x, y)}));
                }
            }
            break;
        }
        case IMPULSE: {
            inputSize_ = 1;
            trainPercentage = 0.7;
            double inc = 10. / 40;
            for (double x = -5; x <= 5; x += inc) {
                inputs_.push_back(new Matrix(1, 1, {x * 1.}));
                outputs_.push_back(new Matrix(1, 1, {impulseFunc(x)}));
            }
            break;
        }
        default:
            throw runtime_error("Undefined function " + function);
    }
    // Store data bundle.
    if (separateData) {
        trainBundle_.push_back(
                DataUtils::separateData<Matrix>(&inputs_, &outputs_, trainPercentage));
    } else {
//        for (uint i = 0; i < inputs_.size(); i++)
//            trainBundle_.push_back(
//                    new SimpleData(new vector<Matrix *>({inputs_[i]}), new vector<Matrix *>({outputs_[i]})));
        trainBundle_.push_back(new SimpleData(&inputs_, &outputs_));
    }
}

RegressionProblem::~RegressionProblem() {
    for (Data *d : trainBundle_)
        delete d;
}

LossFunction<Matrix> &RegressionProblem::getLossFunction() {
    return lossFunction_;
}

string RegressionProblem::toLabel(Matrix &matrix) {
    return std::to_string(matrix.get(0, 0));
}

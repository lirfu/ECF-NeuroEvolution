//
// Created by lirfu on 12.05.18..
//

#include <data/DataUtils.h>
#include "RegressionProblem.h"

RegressionProblem::RegressionProblem(string *functionName, uint samplesNum) {
    if (*functionName == paramOneDim_) {
        inputSize_ = 1;
        // Generate data.
        vector<Matrix *> inputs;
        vector<Matrix *> outputs;
        for (int i = 0; i < samplesNum; i++) {
            double x = i * 2. * M_PI / samplesNum;
            inputs.push_back(new Matrix(1, 1, {x}));
            outputs.push_back(new Matrix(1, 1, {oneDimensionalFunc(x)}));
        }
        // Store data divided into train/validation sets
        dataset_.push_back(DataUtils::shuffleData(DataUtils::separateData<Matrix>(&inputs, &outputs, 0.7)));
    } else if (*functionName == paramRosenbrock_) {
        inputSize_ = 2;
        // Generate data.
        vector<Matrix *> inputs;
        vector<Matrix *> outputs;
        uint axisSamples = static_cast<uint>(sqrt(samplesNum));
        for (int x = 0; x < axisSamples; x++) {
            for (int y = 0; y < axisSamples; y++) {
                inputs.push_back(new Matrix(1, 2, {x * 1., y * 1.}));
                outputs.push_back(new Matrix(1, 1, {Rosenbrock(x, y)}));
            }
        }
        // Store data divided into train/validation sets
        dataset_.push_back(DataUtils::shuffleData(DataUtils::separateData<Matrix>(&inputs, &outputs, 0.7)));
    } else if (*functionName == paramImpulse_) {
        inputSize_ = 1;
        // Generate data.
        vector<Matrix *> inputs;
        vector<Matrix *> outputs;
        int bound = samplesNum / 2;
        for (int x = -bound; x < bound; x++) {
            inputs.push_back(new Matrix(1, 1, {x * 1.}));
            outputs.push_back(new Matrix(1, 1, {impulseFunc(x)}));
        }
        // Store data divided into train/validation sets
        dataset_.push_back(DataUtils::shuffleData(DataUtils::separateData<Matrix>(&inputs, &outputs, 0.7)));
    }
}

RegressionProblem::~RegressionProblem() {
    for (Data *d : dataset_)
        delete d;
}

string RegressionProblem::toLabel(Matrix &matrix) {
    return std::to_string(matrix.get(0, 0));
}

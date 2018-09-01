//
// Created by lirfu on 09.05.18..
//

#include "XORProblem.h"
#include "data/SimpleData.h"


XORProblem::XORProblem() {
    inputs_.push_back(new Matrix(1, 2, {-1, -1}));
    inputs_.push_back(new Matrix(1, 2, {-1, 1}));
    inputs_.push_back(new Matrix(1, 2, {1, -1}));
    inputs_.push_back(new Matrix(1, 2, {1, 1}));

    outputs_.push_back(new Matrix(1, 1, {-1}));
    outputs_.push_back(new Matrix(1, 1, {1}));
    outputs_.push_back(new Matrix(1, 1, {1}));
    outputs_.push_back(new Matrix(1, 1, {-1}));

    for (uint i = 0; i < inputs_.size(); i++)
        trainBundle_.push_back(new SimpleData(inputs_[i], outputs_[i]));
}

XORProblem::~XORProblem() {
    for (Data *d : trainBundle_)
        delete d;
}

uint XORProblem::inputSize() {
    return 2;
}

uint XORProblem::outputSize() {
    return 1;
}

std::string XORProblem::toLabel(Matrix &matrix) {
    return std::to_string(matrix.get(0, 0) > 0 ? 1 : -1);
}

LossFunction<Matrix> &XORProblem::getLossFunction() {
    return lossFunction_;
}

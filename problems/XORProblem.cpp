//
// Created by lirfu on 09.05.18..
//

#include "XORProblem.h"
#include "data/SimpleData.h"


XORProblem::XORProblem() {
    vector<Matrix *> *inputs = new vector<Matrix *>();
    inputs->push_back(new Matrix(1, 2, {-1, -1}));
    inputs->push_back(new Matrix(1, 2, {-1, 1}));
    inputs->push_back(new Matrix(1, 2, {1, -1}));
    inputs->push_back(new Matrix(1, 2, {1, 1}));

    vector<Matrix *> *outputs = new vector<Matrix *>();
    outputs->push_back(new Matrix(1, 1, {-1}));
    outputs->push_back(new Matrix(1, 1, {1}));
    outputs->push_back(new Matrix(1, 1, {1}));
    outputs->push_back(new Matrix(1, 1, {-1}));

    batches_.push_back(new SimpleData(inputs, outputs));
}

XORProblem::~XORProblem() {
    for (Data *d : batches_)
        delete d;
}

vector<Data *> *XORProblem::getDataset() {
    return &batches_;
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

//
// Created by lirfu on 09.05.18..
//

#ifndef NEUROEVOLUTION_XORPROBLEM_H
#define NEUROEVOLUTION_XORPROBLEM_H

#include <functions/SquareLoss.h>
#include "IProblem.h"

class XORProblem : public IProblem {
private:
    vector<Matrix *> inputs_;
    vector<Matrix *> outputs_;
    vector<Data *> trainBundle_;
    SquareLoss<Matrix> lossFunction_;
public:
    XORProblem();

    ~XORProblem();

    uint inputSize() override;

    uint outputSize() override;

    vector<Matrix *> &getInputs() override {
        return inputs_;
    }

    vector<Matrix *> &getOutputs() override {
        return outputs_;
    }

    vector<Data *> &getTrainBundle() override {
        return trainBundle_;
    }

    std::string toLabel(Matrix &matrix) override;

    LossFunction<Matrix> &getLossFunction() override;

};

#endif //NEUROEVOLUTION_XORPROBLEM_H

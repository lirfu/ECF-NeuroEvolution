//
// Created by lirfu on 13.05.18..
//

#ifndef NEUROEVOLUTION_WINEQUALITYPROBLEM_H
#define NEUROEVOLUTION_WINEQUALITYPROBLEM_H


#include <functions/SquareLoss.h>
#include "IProblem.h"

class WineQualityProblem : public IProblem {
private:
    vector<Matrix *> inputs_;
    vector<Matrix *> outputs_;
    vector<Data *> dataset_;
    SquareLoss<Matrix> lossFunction_;
public:
    WineQualityProblem(std::string &filepath);

    ~WineQualityProblem();

    uint inputSize() override;

    uint outputSize() override;

    vector<Matrix *> &getInputs() override {
        return inputs_;
    }

    vector<Matrix *> &getOutputs() override {
        return outputs_;
    }

    vector<Data *> &getTrainBundle() override {
        return dataset_;
    }

    string toLabel(Matrix &matrix) override;

    LossFunction<Matrix> &getLossFunction() override;
};


#endif //NEUROEVOLUTION_WINEQUALITYPROBLEM_H

//
// Created by lirfu on 13.05.18..
//

#ifndef NEUROEVOLUTION_WINEQUALITYPROBLEM_H
#define NEUROEVOLUTION_WINEQUALITYPROBLEM_H


#include "IProblem.h"

class WineQualityProblem : public IProblem{
private:
    vector<Data *> dataset_;
public:
    WineQualityProblem();

    ~WineQualityProblem();

    uint inputSize() override;

    uint outputSize() override;

    vector<Data *> *getDataset() override;

    string toLabel(Matrix &matrix) override;
};


#endif //NEUROEVOLUTION_WINEQUALITYPROBLEM_H

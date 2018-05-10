//
// Created by lirfu on 09.05.18..
//

#ifndef NEUROEVOLUTION_XORPROBLEM_H
#define NEUROEVOLUTION_XORPROBLEM_H

#include "IProblem.h"

class XORProblem : public IProblem {
private:
    vector<Data *> batches_;
public:
    XORProblem();

    ~XORProblem();

    uint inputSize() override;

    uint outputSize() override;

    vector<Data *> &getDataset() override;
};

#endif //NEUROEVOLUTION_XORPROBLEM_H

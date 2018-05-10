//
// Created by lirfu on 10.05.18..
//

#ifndef NEUROEVOLUTION_IPROBLEM_H
#define NEUROEVOLUTION_IPROBLEM_H


#include <data/Data.h>

class IProblem {
public:
    virtual uint inputSize()=0;

    virtual uint outputSize()=0;

    virtual std::vector<Data *> *getDataset()=0;

    virtual std::string toLabel(Matrix &)=0;
};


#endif //NEUROEVOLUTION_IPROBLEM_H

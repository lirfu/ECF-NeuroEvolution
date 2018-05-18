//
// Created by lirfu on 10.05.18..
//

#ifndef NEUROEVOLUTION_IPROBLEM_H
#define NEUROEVOLUTION_IPROBLEM_H


#include <data/Data.h>

class IProblem {
public:
    /** Size of the input. */
    virtual uint inputSize()=0;

    /** Size of the output. */
    virtual uint outputSize()=0;

    /** Get input matrices of the problem. */
    virtual std::vector<Matrix *> &getInputs()=0;

    /** Get output matrices of the problem. */
    virtual std::vector<Matrix *> &getOutputs()=0;

    /** Get data bundle used for training net (might include cross-validation, pre-processed inputs, etc.). */
    virtual std::vector<Data *> &getTrainBundle()=0;

    /** Get string representation of the network's output matrix (useful for one-hot-to-class in classification). */
    virtual std::string toLabel(Matrix &)=0;
};


#endif //NEUROEVOLUTION_IPROBLEM_H

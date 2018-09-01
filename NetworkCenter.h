//
// Created by lirfu on 10.05.18..
//

#ifndef NEUROEVOLUTION_NETWORKCENTER_H
#define NEUROEVOLUTION_NETWORKCENTER_H

#include <NeuralNetwork.h>
#include <functions/DerivativeFunction.h>
#include <descentmethods/DescendMethod.h>
#include <descentmethods/VanillaGradientDescend.h>
#include <weightinitializers/RandomWeightInitializer.h>
#include <layers/FullyConnectedLayer.h>
#include <cfloat>
#include "problems/IProblem.h"

class NetworkCenter {
private:
    /* Network definition parameters. */
    WeightInitializer *initializer_;
public:
    /* Problem to solve. */
    IProblem *problem_;
    /* Training parameters. */
    double learningRate_;
    double minLoss_;
    uint maxIterations_;
    shared_ptr<DerivativeFunction> hiddenFunction_;
    shared_ptr<DerivativeFunction> outputFunction_;
    shared_ptr<DescendMethod> descendMethod_;

    NetworkCenter(IProblem *, DerivativeFunction *hiddenFunction,
                  DerivativeFunction *outputFunction, double learningRate, double minLoss, uint maxIterations);

    ~NetworkCenter();

    double trainNetwork(NeuralNetwork &network, bool silent, bool graph = false);

    uint inputSize() {
        return problem_->inputSize();
    }

    uint outputSize() {
        return problem_->outputSize();
    }
};


#endif //NEUROEVOLUTION_NETWORKCENTER_H

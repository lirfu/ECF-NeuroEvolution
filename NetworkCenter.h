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
#include "problems/IProblem.h"

class NetworkCenter {
private:
    IProblem *problem_;
    WeightInitializer *initializer_;
    shared_ptr<DerivativeFunction> hiddenFunction_;
    shared_ptr<DerivativeFunction> outputFunction_;
    shared_ptr<DescendMethod> descendMethod_;
public:
    NetworkCenter(IProblem *, DerivativeFunction *hiddenFunction,
                  DerivativeFunction *outputFunction);

    ~NetworkCenter();

    double testNetwork(std::vector<uint> architecture);
};


#endif //NEUROEVOLUTION_NETWORKCENTER_H

//
// Created by lirfu on 10.05.18..
//

#ifndef NEUROEVOLUTION_DISPLAYBESTOPERATOR_H
#define NEUROEVOLUTION_DISPLAYBESTOPERATOR_H


#include <ecf/Operator.h>
#include <ecf/ECF_base.h>
#include "NetworkCenter.h"

class DisplayBestOperator : public Operator {
private:
    NetworkCenter *networkCenter_;
public:
    explicit DisplayBestOperator(NetworkCenter *center);
    bool initialize(StateP p) override;

    bool operate(StateP p) override;
};

typedef boost::shared_ptr<DisplayBestOperator> DisplayBestOperatorP;

#endif //NEUROEVOLUTION_DISPLAYBESTOPERATOR_H

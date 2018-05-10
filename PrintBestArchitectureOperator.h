//
// Created by lirfu on 10.05.18..
//

#ifndef NEUROEVOLUTION_PRINTBESTARCHITECTUREOPERATOR_H
#define NEUROEVOLUTION_PRINTBESTARCHITECTUREOPERATOR_H


#include <ecf/Operator.h>
#include <ecf/ECF_base.h>

class PrintBestArchitectureOperator : public Operator {
public:
    bool operate(StateP p) override;
};

typedef boost::shared_ptr<PrintBestArchitectureOperator> PrintBestArchitectureOperatorP;

#endif //NEUROEVOLUTION_PRINTBESTARCHITECTUREOPERATOR_H

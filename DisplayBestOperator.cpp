//
// Created by lirfu on 10.05.18..
//

#include <ecf/ECF_base.h>
#include "DisplayBestOperator.h"

bool DisplayBestOperator::initialize(StateP p) {
    // TODO Add extended commands to java program
//    std::cout << "init 2 1" << std::endl;
//    std::cout << "clear 1" << std::endl;
//    std::cout << "clear 2" << std::endl;
    return Operator::initialize(p);
}

bool DisplayBestOperator::operate(StateP p) {
//    // Draw the output of best network.
//    std::cout << "clear 1" << std::endl;
//    for (uint i = 0; i < data.size(); i++) {
//        for (uint j = 0; j < data[i]->testSize(); j++) {
//            std::cout << "add 1 " << problem_->toLabel(net.getOutput(*data[i]->getValidationInputs()->at(j)))
//                      << " " << problem_->toLabel(*data[i]->getValidationOutputs()->at(j)) << std::endl;
//        }
//    }

//    // Add loss value to loss graph.
//    std::cout << "add 2 "<< loss << std::endl;
    return true;
}

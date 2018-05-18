//
// Created by lirfu on 13.05.18..
//

#ifndef NEUROEVOLUTION_DIRECTTEST_H
#define NEUROEVOLUTION_DIRECTTEST_H


class DirectTest {
public:
    static void testAll(){
//        testXOR();
        testFunctionOnedim(); // TODO NOT OPTIMAL YET!!
//        testFunctionRosenbrock(); // TODO NOT OPTIMAL YET!!
//        testFunctionImpulse(); // TODO NOT OPTIMAL YET!!
//        testWineQuality();
    }

    static void testXOR();

    static void testFunctionOnedim();

    static void testFunctionRosenbrock();

    static void testFunctionImpulse();

    static void testWineQuality();
};


#endif //NEUROEVOLUTION_DIRECTTEST_H

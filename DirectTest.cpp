//
// Created by lirfu on 13.05.18..
//

#include <functions/Sigmoid.h>
#include <functions/Linear.h>
#include "DirectTest.h"
#include "problems/XORProblem.h"
#include "NetworkCenter.h"
#include "problems/RegressionProblem.h"
#include "problems/WineQualityProblem.h"

void DirectTest::testXOR() {
//    cout << "Running XOR..." << endl;
    for(uint i=0; i<20; i++) {
        vector<uint> architecture({6,3});
        NetworkCenter net(new XORProblem(), new Sigmoid(), new Linear(), 2e-3, 1e-9, 1000000);
        double loss = net.trainNetwork(architecture, false);
    }
//    if (loss <= 1e-9) {
//        cout << "PASS" << endl;
//    } else {
//        cout << "LOSS too high: " << loss << endl;
//    }
}

void DirectTest::testFunctionOnedim() {
//    cout << "Running onedim..." << endl;
//    for(uint i=0; i<20; i++) {
        vector<uint> architecture({7, 9});
        string func("onedim");
        NetworkCenter net(new RegressionProblem(func, 30), new Sigmoid(), new Linear(), 1e-2, 5e-3, 1000000);
        double loss = net.trainNetwork(architecture, false, true);
//    }
//    if (loss <= 1e-9) {
//        cout << "PASS" << endl;
//    } else {
//        cout << "LOSS too high: " << loss << endl;
//    }
}

void DirectTest::testFunctionRosenbrock() {
    cout << "Running rosenbrock..." << endl;
    vector<uint> architecture({20, 50, 15});
    string func("rosenbrock");
    NetworkCenter net(new RegressionProblem(func, 60), new Sigmoid(), new Linear(), 1e-4, 1e-9, 1000000);
    double loss = net.trainNetwork(architecture, false, true);
    if (loss <= 1e-9) {
        cout << "PASS" << endl;
    } else {
        cout << "LOSS too high: " << loss << endl;
    }
}

void DirectTest::testFunctionImpulse() {
    cout << "Running impulse..." << endl;
    vector<uint> architecture({100, 20, 5});
    string func("impulse");
    NetworkCenter net(new RegressionProblem(func, 100), new Sigmoid(), new Linear(), 7e-3, 1e-9, 1000000);
    double loss = net.trainNetwork(architecture, false, true);
    if (loss <= 1e-9) {
        cout << "PASS" << endl;
    } else {
        cout << "LOSS too high: " << loss << endl;
    }
}

void DirectTest::testWineQuality() {
    cout << "Running winequality..." << endl;
    vector<uint> architecture({60, 10, 5});
    string datasetPath("/home/lirfu/PersonalWorkspace/NeuroEvolution/wine_data/winequality-joined.csv");
    NetworkCenter net(new WineQualityProblem(datasetPath),
                      new Sigmoid(), new Linear(), 5e-2, 1e-9, 1000000);
    double loss = net.trainNetwork(architecture, false);
    if (loss <= 1e-9) {
        cout << "PASS" << endl;
    } else {
        cout << "LOSS too high: " << loss << endl;
    }
}

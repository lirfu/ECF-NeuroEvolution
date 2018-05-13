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
    cout << "Running XOR..." << endl;
    vector<uint> architecture({2});
    NetworkCenter net(new XORProblem(), new Sigmoid(), new Linear(), 1e-3, 1e-9, 1000000);
    double loss = net.testNetwork(architecture, true);
    if (loss <= 1e-9) {
        cout << "PASS" << endl;
    } else {
        cout << "LOSS too high: " << loss << endl;
    }
}

void DirectTest::testFunctionOnedim() {
    cout << "Running onedim..." << endl;
    vector<uint> architecture({20, 20, 20});
    string func("onedim");
    NetworkCenter net(new RegressionProblem(&func, 200), new Sigmoid(), new Linear(), 1e-3, 1e-9, 1000000);
    double loss = net.testNetwork(architecture, true);
    if (loss <= 1e-9) {
        cout << "PASS" << endl;
    } else {
        cout << "LOSS too high: " << loss << endl;
    }
}

void DirectTest::testFunctionRosenbrock() {
    cout << "Running rosenbrock..." << endl;
    vector<uint> architecture({20, 20, 20});
    string func("rosenbrock");
    NetworkCenter net(new RegressionProblem(&func, 200), new Sigmoid(), new Linear(), 1e-3, 1e-9, 1000000);
    double loss = net.testNetwork(architecture, true);
    if (loss <= 1e-9) {
        cout << "PASS" << endl;
    } else {
        cout << "LOSS too high: " << loss << endl;
    }
}

void DirectTest::testFunctionImpulse() {
    cout << "Running impulse..." << endl;
    vector<uint> architecture({40, 10});
    string func("impulse");
    NetworkCenter net(new RegressionProblem(&func, 200), new Sigmoid(), new Linear(), 1e-3, 1e-9, 1000000);
    double loss = net.testNetwork(architecture, true);
    if (loss <= 1e-9) {
        cout << "PASS" << endl;
    } else {
        cout << "LOSS too high: " << loss << endl;
    }
}

void DirectTest::testWineQuality() {
    cout << "Running winequality..." << endl;
    vector<uint> architecture({50, 20});
    string func("impulse");
    NetworkCenter net(new WineQualityProblem(), new Sigmoid(), new Linear(), 1e-3, 1e-9, 1000000);
    double loss = net.testNetwork(architecture, false);
    if (loss <= 1e-9) {
        cout << "PASS" << endl;
    } else {
        cout << "LOSS too high: " << loss << endl;
    }
}

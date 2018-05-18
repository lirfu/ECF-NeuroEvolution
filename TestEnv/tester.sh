#!/usr/bin/env bash

mkdir logs 2> /dev/null

#f="tests/xor.xml"
#f="tests/function_onedim.xml"
#f="tests/function_rosenbrock.xml"
#f="tests/function_impulse.xml"
#f="tests/wine.xml"
for f in tests/*
do
    echo "Running $f ..."
    for run in {1..4}
    do
        echo "Starting run $run/20"
        ././../cmake-build-debug/NeuroEvolution $f > "logs/$(basename $f)_$run.log" &
    done
    wait
    for run in {5..8}
    do
        echo "Starting run $run/20"
        ././../cmake-build-debug/NeuroEvolution $f > "logs/$(basename $f)_$run.log" &
    done
    wait
    for run in {9..12}
    do
        echo "Starting run $run/20"
        ././../cmake-build-debug/NeuroEvolution $f > "logs/$(basename $f)_$run.log" &
    done
    wait
    for run in {13..16}
    do
        echo "Starting run $run/20"
        ././../cmake-build-debug/NeuroEvolution $f > "logs/$(basename $f)_$run.log" &
    done
    wait
    for run in {17..20}
    do
        echo "Starting run $run/20"
        ././../cmake-build-debug/NeuroEvolution $f > "logs/$(basename $f)_$run.log" &
    done
    wait
    echo "Done!"
done

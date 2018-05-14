#!/usr/bin/env bash

mkdir logs 2> /dev/null

for f in tests/*
do
    echo "Running $f ..."
    for run in {1..7}
    do
        echo "Starting run $run/7"
        ././../cmake-build-debug/NeuroEvolution $f > "logs/$(basename $f)_$run.log" &
    done

    wait
    echo "Done!"
done

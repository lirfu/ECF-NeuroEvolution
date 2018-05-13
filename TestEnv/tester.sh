#!/usr/bin/env bash

mkdir logs 2> /dev/null

for run in {1..5}
do
    for f in tests/*
    do
        echo "Running $f ..."
        ././../cmake-build-debug/NeuroEvolution $f > "logs/$(basename $f)_$run.log" &
    done

    wait
done

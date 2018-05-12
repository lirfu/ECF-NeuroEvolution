#!/usr/bin/env bash

mkdir logs 2> /dev/null

for f in tests/*
do
    echo "Running $f ..."
    ././../cmake-build-debug/NeuroEvolution $f > "logs/$(basename $f).log" &
done

wait
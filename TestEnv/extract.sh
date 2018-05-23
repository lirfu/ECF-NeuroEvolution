#!/usr/bin/env bash

for d in logs/*
do
    echo "Iterations: $d"
    for p in $d/*
    do
        echo "Problem: $p"
        for f in $p/*
        do
            loss=$(grep "FitnessM" $f | sed -E "s/[ \t]*<FitnessM.. value=\"(.+)\".*/\1/")
            archi=$(grep "Best architecture:" $f | tail -1 | sed -E "s/Best architecture: \[(.+)\]/\1/")
            echo "$loss"
        done
    done
done
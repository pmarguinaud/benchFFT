#!/bin/bash

set -x

if [ 0 -eq 1 ]
then
N=20
L=6
P=2

nvprof ./benchFFT.x $N $L $L $L 1 1 $P 1 1
fi

N=4000
L=100000
P=0

nvprof ./benchFFT.x $N $L $L $L 1 1 $P 1 10

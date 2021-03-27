#!/bin/bash

set -x

if [ 0 -eq 1 ]
then
N=20
L=6
P=2

nvprof ./benchFFT.x $N $L $L $L 1 1 $P 1 1
fi

# NDLON=4000
# NDGLG=4000
# NFLEVG=100
# NFIELDS=10

N=4000
L=100000
P=0

nvprof ./benchFFT.x $N $L $L $L 1 1 $P 1 10

N=4000
L=1000
let "DIST1=$N+2"
let "DIST2=$DIST1/2"
P=0

nvprof ./benchFFT.x $N $L 1 1 $DIST1 $DIST2 $P 1 10000


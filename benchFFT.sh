#!/bin/bash

set -x
# -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/compilers/lib -L/home/ms/fr/sor/install/pgi-20.9/cuda/lib64 \

nvcc -o benchFFT.x -g benchFFT.cu \
  -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/compilers/lib \
  -lcufft -lcudadevice -lcudart

./benchFFT.x

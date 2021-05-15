

PREFIX1=/gpfslocalsys/nvhpc/20.11/Linux_x86_64/20.11/math_libs/10.2
PREFIX2=/gpfslocalsys/nvhpc/20.11/Linux_x86_64/20.11/compilers
PREFIX3=/gpfslocalsys/nvhpc/20.11/Linux_x86_64/20.11/cuda/11.0/targets/x86_64-linux

all: benchFFT.gpu.x benchFFT.cpu.x 

benchFFT.gpu.x: benchFFT.gpu.cu
	nvcc -o benchFFT.gpu.x -g -I$(PREFIX1)/include benchFFT.gpu.cu -L$(PREFIX1)/lib64 -lcufft -L$(PREFIX2)/lib \
           -lcudadevice -lcudart -lnvcpumath -lnvomp -lacchost -lacccuda -lpgc -laccdevice \
           --linker-options "-rpath,$(PREFIX1)/lib64" --linker-options "-rpath,$(PREFIX2)/lib" --linker-options "-rpath,$(PREFIX3)/lib"

benchFFT.cpu.x: benchFFT.cpu.c
	icc  -o benchFFT.cpu.x -g benchFFT.cpu.c -lfftw3

clean:
	\rm -f *.o *.x


benchFFT.x: benchFFT.cu
	nvcc -o benchFFT.x -g benchFFT.cu -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/compilers/lib -lcufft -lcudadevice -lcudart

clean:
	\rm -f *.o *.x

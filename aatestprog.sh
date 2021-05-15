#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu_p2,gpu_p1
#SBATCH --time 00:05:00
#SBATCH --exclusive

set -x
cd /linkhome/rech/gensoq01/ufh62jk/benchFFT

N=20
L=6
let "DIST1=$N+2"
let "DIST2=$DIST1/2"
P=2


N=4000
L=1000
let "DIST1=$N+2"
let "DIST2=$DIST1/2"
P=0

export N L DIST1 DIST2 P

(
  set +x
  module load nvidia-compilers/20.11
  set -x

  nvprof \
  ./benchFFT.gpu.x $N $L 1 1 $DIST1 $DIST2 $P 1 10000 > out.gpu.eo 2>&1

  perl -i -pe 's/-0\.0/ 0.0/go;' out.gpu.eo
  cat out.gpu.eo
)

(
  set +x
  module load intel-compilers
  module load fftw
  set -x
  
  for iomp in 20 30 40
  do

  perl -e ' my $N = shift; my $n = shift;  my @X; for my $i (0 .. $n-1) { my @x = (0) x $N; $x[$i] = 1; push @X, join ("", @x) } print join (":", @X) . "\n" '  40 $iomp > linux_bind.txt

  OMP_NUM_THREADS=$iomp \
  /usr/bin/time -f 'real=%e' ./benchFFT.cpu.x $N $L 1 1 $DIST1 $DIST2 $P 1 10000 > out.cpu.eo 2>&1
  perl -i -pe 's/-0\.0/ 0.0/go;' out.cpu.eo
  cat out.cpu.eo
  done
 
  
)

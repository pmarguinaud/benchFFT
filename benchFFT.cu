#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>

#define cufftSafeCall(err) __cufftSafeCall (err, __FILE__, __LINE__)

static const char *_cudaGetErrorEnum (cufftResult error)
{
  switch (error)
    {
#define cr(x) case CUFFT_##x: return #x
      cr (SUCCESS);
      cr (INVALID_PLAN);
      cr (ALLOC_FAILED);
      cr (INVALID_TYPE);
      cr (INVALID_VALUE);
      cr (INTERNAL_ERROR);
      cr (EXEC_FAILED);
      cr (SETUP_FAILED);
      cr (INVALID_SIZE);
      cr (UNALIGNED_DATA);
#undef cr
    }
  return "UNKNOWN";
}

inline void __cufftSafeCall (cufftResult err, const char * file, const int line)
{
  if (CUFFT_SUCCESS != err) 
    {
      fprintf (stderr, "CUFFT error in file '%s'\n",__FILE__);
      fprintf (stderr, "CUFFT error %d: %s\nterminating!\n", err, _cudaGetErrorEnum (err)); 
      cudaDeviceReset (); 
    }
}


int main (int argc, char * argv[])
{

  if (argc < 8)
    {
      fprintf (stderr, "Usage: %s N LOT istride ostride idist odist\n", argv[0]);
      return 1;
    }

  int N       = atoi (argv[1]);
  int LOT     = atoi (argv[2]);
  int istride = atoi (argv[3]);
  int ostride = atoi (argv[4]);
  int idist   = atoi (argv[5]);
  int odist   = atoi (argv[6]);
  int llprint = atoi (argv[7]);

  cufftHandle plan;

  if (cudaDeviceSynchronize() != cudaSuccess)
    {
      fprintf(stderr, "Cuda error: Failed to synchronize\n");
      return 1;	
    }

  int embed[1] = {1};

  cufftSafeCall (cufftCreate (&plan));

  cufftSafeCall (cufftPlanMany (&plan, 1, &N, embed, istride, idist, embed, ostride, odist, CUFFT_D2Z, LOT));

  if (llprint)
  printf (" N = %d, LOT = %d, istride = %d, ostride = %d, idist = %d, odist = %d\n", N, LOT, istride, ostride, idist, odist);

  if (cudaDeviceSynchronize () != cudaSuccess)
    {
      fprintf(stderr, "Cuda error: Failed to synchronize\n");
      return 1;	
    }

  double * z = (double *)malloc (sizeof (double) * LOT * idist);

  for (int j = 0; j < LOT; j++)
  for (int i = 0; i < idist; i++)
    z[j*idist+i] = (i >= N) ? 9999. : (i %2) ? +1. : -1.;

  if (llprint)
  for (int j = 0; j < LOT; j++)
    {
      for (int i = 0; i < idist; i++)
        printf (" %8.1f", z[j*idist+i]);
      printf ("\n");
    }

  cufftDoubleComplex * data = NULL;

  size_t sz = sizeof (double) * LOT * idist;

  cudaMalloc ((void**)&data, sz);

  cudaMemcpy (data, z, sz, cudaMemcpyHostToDevice);

  cufftSafeCall (cufftExecD2Z (plan, (cufftDoubleReal*)data, data));

  cudaMemcpy (z, data, sz, cudaMemcpyDeviceToHost);

  if (llprint)
  for (int j = 0; j < LOT; j++)
    {
      for (int i = 0; i < idist; i++)
        printf (" %8.1f", z[j*idist+i]);
      printf ("\n");
    }


  return 0;
}

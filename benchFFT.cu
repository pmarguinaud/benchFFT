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
  int N = 20;
  int LOT = 6;

  cufftHandle plan;

  if (cudaDeviceSynchronize() != cudaSuccess)
    {
      fprintf(stderr, "Cuda error: Failed to synchronize\n");
      return 1;	
    }


  int embed[1] = {1};
  int stride = 1;
  int dist = 26;


  cufftSafeCall (cufftCreate (&plan));

  cufftSafeCall (cufftPlanMany (&plan, 1, &N, embed, stride, dist, embed, stride, dist/2, CUFFT_D2Z, LOT));

  printf (" N = %d\n", N);

  if (cudaDeviceSynchronize () != cudaSuccess)
    {
      fprintf(stderr, "Cuda error: Failed to synchronize\n");
      return 1;	
    }

  double z[LOT][dist];

  for (int j = 0; j < LOT; j++)
  for (int i = 0; i < dist; i++)
    z[j][i] = (i >= N) ? 9999. : (i %2) ? +1. : -1.;

  for (int j = 0; j < LOT; j++)
  for (int i = 0; i < dist; i++)
    printf (" %2d %2d %12.4f\n", j, i, z[j][i]);

  printf ("------------\n");

  cufftDoubleComplex * data = NULL;

  cudaMalloc ((void**)&data, sizeof (z));

  cudaMemcpy (data, z, sizeof (z), cudaMemcpyHostToDevice);

  cufftSafeCall (cufftExecD2Z (plan, (cufftDoubleReal*)data, data));

  cudaMemcpy (z, data, sizeof (z), cudaMemcpyDeviceToHost);

  for (int j = 0; j < LOT; j++)
  for (int i = 0; i < dist; i++)
    printf (" %2d %2d %12.4f\n", j, i, z[j][i]);


  return 0;
}

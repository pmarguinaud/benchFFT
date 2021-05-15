#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

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

  if (argc < 10)
    {
      fprintf (stderr, "Usage: %s N LOT istride ostride idist odist llprint kfunc ntime\n", argv[0]);
      return 1;
    }

  int N       = atoi (argv[1]);
  int LOT     = atoi (argv[2]);
  int istride = atoi (argv[3]);
  int ostride = atoi (argv[4]);
  int idist   = atoi (argv[5]);
  int odist   = atoi (argv[6]);
  int llprint = atoi (argv[7]);
  int kfunc   = atoi (argv[8]);
  int ntime   = atoi (argv[9]);

  assert ((istride == 1) || (idist == 1));
  assert ((ostride == 1) || (odist == 1));

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

  size_t sz = LOT * idist + N * istride + 2 * LOT;

  if (llprint)
    printf (" sz = %ld\n", sz);

  double * z = (double *)malloc (sz * sizeof (double));

  for (int i = 0; i < sz; i++)
    z[i] = 9999.;


  for (int j = 0; j < LOT; j++)
  for (int i = 0; i < N; i++)
    {
      double zval = 0.;
      switch (kfunc)
        {
          case 1: zval = (i % 4) ? +1. : -1.; break;
          case 2: zval = (i % 2) ? +1. : -1.; break;
          default: zval = 1.;
        }
      z[j*idist+i*istride] = zval;
    }


  if (llprint == 1)
  for (int j = 0; j < LOT; j++)
    {
      for (int i = 0; i < N+2; i++)
        printf (" %8.1f", z[j*idist+i*istride]);
      printf ("\n");
    }

  if (llprint == 2)
  for (int i = 0; i < sz; i++)
    {
      printf (" %8.1f", z[i]);
      if ((((i + 1) % 20) == 0) || (i == sz - 1)) printf ("\n");
    }

  cufftDoubleComplex * data = NULL;

  cudaMalloc ((void**)&data, sz * sizeof (double));

  cudaMemcpy (data, z, sz * sizeof (double), cudaMemcpyHostToDevice);


  clock_t t0 = clock ();
  for (int itime = 0; itime < ntime; itime++)
    cufftSafeCall (cufftExecD2Z (plan, (cufftDoubleReal*)data, data));
  clock_t t1 = clock ();

  printf (" sz = %ld, dt = %f\n", sz, (double)(t1-t0)/1e+6);

  cudaMemcpy (z, data, sz * sizeof (double), cudaMemcpyDeviceToHost);

  if (llprint == 1)
  for (int j = 0; j < LOT; j++)
    {
      for (int i = 0; i < N+2; i++)
        printf (" %8.1f", z[j*idist+i*istride]);
      printf ("\n");
    }

  if (llprint == 2)
  for (int i = 0; i < sz; i++)
    {
      printf (" %8.1f", z[i]);
      if ((((i + 1) % 20) == 0) || (i == sz - 1)) printf ("\n");
    }


  return 0;
}

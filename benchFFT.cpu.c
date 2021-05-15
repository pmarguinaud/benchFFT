#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

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

  {
  int rank = 0, size = 1;
  linux_bind_ (&rank, &size);
  }

  assert ((istride == 1) || (idist == 1));
  assert ((ostride == 1) || (odist == 1));

  int embed[1] = {1};

  fftw_plan p = fftw_plan_many_dft_r2c 
      (1, &N, 1, (double *)NULL, embed, istride, idist, (fftw_complex *)NULL, 
       embed, ostride, odist, FFTW_ESTIMATE+FFTW_NO_SIMD);

  if (llprint)
  printf (" N = %d, LOT = %d, istride = %d, ostride = %d, idist = %d, odist = %d\n", N, LOT, istride, ostride, idist, odist);

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

  struct timespec t0, t1;
  clock_gettime (CLOCK_REALTIME, &t0);

  for (int itime = 0; itime < ntime; itime++)
    {
#pragma omp parallel for
      for (int j = 0; j < LOT; j++)
        {
          double * zj = &z[j*idist];
          fftw_execute_dft_r2c (p, zj, (fftw_complex *)zj);
        }
    }

  clock_gettime (CLOCK_REALTIME, &t1);

  printf (" sz = %ld, dt = %f\n", sz, (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9);

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

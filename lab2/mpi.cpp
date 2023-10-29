// Header inclusions, if any...

#include <mpi.h>
#include <cstring>

#include "lib/gemm.h"
#include "lib/common.h"
// You can directly use aligned_alloc
// with lab2::aligned_alloc(...)

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{
  float *a_contiguous = (float *)lab2::aligned_alloc(1024, kI * kK * sizeof(float));
  float *b_contiguous = (float *)lab2::aligned_alloc(1024, kK * kJ * sizeof(float));
  float *c_contiguous = (float *)lab2::aligned_alloc(1024, kI * kJ * sizeof(float));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
  {
    for (int i = 0; i < kI; ++i)
    {
      for (int j = 0; j < kK; ++j)
        a_contiguous[kK * i + j] = a[i][j];
    }
    for (int i = 0; i < kK; ++i)
    {
      for (int j = 0; j < kJ; ++j)
        b_contiguous[kJ * i + j] = b[i][j];
    }
    for (int i = 0; i < kI; ++i)
    {
      for (int j = 0; j < kJ; ++j)
        c_contiguous[kJ * i + j] = c[i][j];
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int chunk_size = kI / size;
  float *rec_A = (float *)lab2::aligned_alloc(1024, chunk_size * kK * sizeof(float));
  float *rec_C = (float *)lab2::aligned_alloc(1024, chunk_size * kJ * sizeof(float));
  MPI_Scatter(a_contiguous, chunk_size * kK, MPI_FLOAT, rec_A, chunk_size * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_contiguous, kK * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);
  for (int ii = 0; ii < chunk_size; ii += 64)
    for (int kk = 0; kk < kK; kk += 8)
      for (int jj = 0; jj < kJ; jj += 1024)
        for (int i = ii; i < ii + 64; ++i)
          for (int j = jj; j < jj + 1024; ++j)
          {
            float reg = 0;
            for (int k = kk; k < kk + 8; ++k)
              reg += rec_A[i * kK + k] * b_contiguous[k * kJ + j];
            rec_C[i * kJ + j] = reg;
          }

  MPI_Gather(rec_C, chunk_size * kJ, MPI_FLOAT, c_contiguous, chunk_size * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    for (int i = 0; i < kI; ++i)
    {
      for (int j = 0; j < kJ; ++j)
        c[i][j] = c_contiguous[kJ * i + j];
    }
  }
}

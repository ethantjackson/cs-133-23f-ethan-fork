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
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int num_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  float *a_contiguous = (float *)lab2::aligned_alloc(64, kI * kK * sizeof(float));
  float *b_contiguous = (float *)lab2::aligned_alloc(64, kK * kJ * sizeof(float));

  if (rank == 0)
  {
    for (int i = 0; i < kI; ++i)
    {
      for (int k = 0; k < kK; ++k)
        a_contiguous[i * kK + k] = a[i][k];
    }
    for (int k = 0; k < kK; ++k)
    {
      for (int j = 0; j < kJ; ++j)
        b_contiguous[k * kJ + j] = b[k][j];
    }
  }

  float *a_local = (float *)lab2::aligned_alloc(64, kI * kK / num_processes * sizeof(float));
  float *c_local = (float *)lab2::aligned_alloc(64, kI * kJ / num_processes * sizeof(float));
  memset(c_local, 0, kI * kJ / num_processes * sizeof(float));

  float *c_contiguous = (float *)lab2::aligned_alloc(64, kI * kJ * sizeof(float));

  MPI_Scatter(a_contiguous, kI * kK / num_processes, MPI_FLOAT, a_local, kI * kK / num_processes, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_contiguous, kK * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);
  for (int ii = 0; ii < kI / num_processes; ii += 64)
    for (int jj = 0; jj < kJ; jj += 1024)
      for (int kk = 0; kk < kK; kk += 8)
        for (int i = ii; i < ii + 64; ++i)
          for (int j = jj; j < jj + 1024; ++j)
          {
            for (int k = kk; k < kk + 8; ++k)
              c_local[i * kJ + j] += a_local[i * kK + k] * b_contiguous[k * kJ + j];
          }

  MPI_Gather(c_local, kI * kJ / num_processes, MPI_FLOAT, c_contiguous, kI * kJ / num_processes, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    for (int i = 0; i < kI; ++i)
    {
      for (int j = 0; j < kJ; ++j)
      {
        c[i][j] = c_contiguous[i * kJ + j];
      }
    }
  }
}

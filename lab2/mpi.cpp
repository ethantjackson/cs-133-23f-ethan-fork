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
  int chunk_size = kI / 4;
  float rec_A[chunk_size][kK];
  float rec_C[chunk_size][kJ];
  MPI_Scatter((void *)a, chunk_size * kK, MPI_FLOAT, rec_A, chunk_size * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void *)b, kK * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);
  for (int i = 0; i < chunk_size; ++i)
    std::memset(rec_C[i], 0, sizeof(float) * kJ);

  for (int ii = 0; ii < chunk_size; ii += 64)
    for (int kk = 0; kk < kK; kk += 8)
      for (int jj = 0; jj < kJ; jj += 1024)
        for (int i = ii; i < ii + 64; ++i)
          for (int j = jj; j < jj + 1024; ++j)
          {
            float reg = rec_C[i][j];
            for (int k = kk; k < kk + 8; ++k)
              reg += rec_A[i][k] * b[k][j];
            rec_C[i][j] = reg;
          }

  MPI_Gather(rec_C, chunk_size * kJ, MPI_FLOAT, c, chunk_size * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

#include <cstring>

#include "lib/gemm.h"

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{

#pragma omp parallel for
  for (int i = 0; i < kI; ++i)
    std::memset(c[i], 0, sizeof(float) * kJ);

  int i, j, k, ii, jj, kk;
#pragma omp parallel for private(i, j, k, ii, jj, kk)
  for (ii = 0; ii < kI; ii += 64)
    for (kk = 0; kk < kK; kk += 8)
      for (jj = 0; jj < kJ; jj += 1024)
        for (i = ii; i < ii + 64; ++i)
          for (j = jj; j < jj + 1024; ++j)
          {
            float reg = c[i][j];
            for (k = kk; k < kk + 8; ++k)
              reg += a[i][k] * b[k][j];
            c[i][j] = reg;
          }
}

#include "timing.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/*
  Matrix multiplication of row blocks [row, row + delta) with column blocks 
  [col, col + delta).
 */
double matmul_block(int N, double* A, double* B, double* C, int row, int col, int delta) {
  double wcs, wce, cputime;
  timing(&wcs, &cputime);

  for (int i = 0; i < delta; i++) {
    int iC = i * N + col;
    int iA = (2 * row + i + 1) * i / 2; // (row + 1) + ... + (row + i)
    for (int j = 0; j < delta; j++, iC++) {
      int iB = (2 * col + j + 1) * j / 2; // same as above
      C[iC] = 0.;
      for (int k = 0; k <= MIN(i + row, j + col); k++) C[iC] += A[iA + k] * B[iB + k]; // avoids using known 0 entries
    }
  }

  timing(&wce, &cputime);
  return(wce - wcs);
}



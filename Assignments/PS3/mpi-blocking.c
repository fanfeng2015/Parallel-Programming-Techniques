#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "timing.h"

/*
  This is the blocking MPI program for CPSC424/524 Assignment #3.
  Author: Fan Feng (ff242)
  Date: 10/31/2018
*/

// (x + 1) + ... + (x + l)
int block_size(int x, int l) {
  return (2 * x + l + 1) * l / 2;
}

double matmul_block(int N, double *A, double *B, double *C, int row, int col, int delta);

int main(int argc, char **argv) {
  // This array contains the sizes of the test cases
  int sizes[4] = { 1000, 2000, 4000, 8000 };
  // This array contains the file names for the true answers
  char files[4][50] = { "/home/fas/cpsc424/ahs3/assignment3/C-1000.dat", \
                        "/home/fas/cpsc424/ahs3/assignment3/C-2000.dat", \
                        "/home/fas/cpsc424/ahs3/assignment3/C-4000.dat", \
                        "/home/fas/cpsc424/ahs3/assignment3/C-8000.dat" };

  double *A, *B, *C, *Ctrue; // A is lower triangular, B is upper triangular
  long sizeAB, sizeC;

  double wcs, wce, cputime;
  double commTime, compTime, Fnorm;
  FILE *fptr;

  int rank, size;
  MPI_Init(&argc, &argv); // MPI initialization
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // rank of each process
  MPI_Comm_size(MPI_COMM_WORLD, &size); // number of processes
  MPI_Status status;
    
  // Print a table heading
  if (rank == 0) {
    if (argc == 1) { // Task 2 Part A
      printf("Matrix multiplication times: \n");
      printf(" Rank      Total Time \n");
      printf("-------    ---------- \n");  
    }
    else { // Task 2 Part B
      printf("Matrix multiplication times: \n");
      printf(" Rank      Comp. Time    Comm. Time    Total Time \n");
      printf("-------    ----------    ----------    ---------- \n");      
    }
  }

  int run = (argc == 1) ? 0 : 3;
  // Now run the four test cases
  for ( ; run < 4; run++) {
    int N = sizes[run];
    // number of rows/cols per block, the row/column blocks of the i-th process is [i * delta, (i + 1) * delta)
    int delta = N / size;

    sizeAB = N * (N + 1) / 2; // Only enough space for the non-zero portions of the matrices
    sizeC = N * N; // All of C will be non-zero, in general!

    A = (double *) calloc(sizeAB, sizeof(double));
    B = (double *) calloc(sizeAB, sizeof(double));
    C = (double *) calloc(sizeC, sizeof(double));

    if (rank == 0) { // master process
      printf("N = %d \n", N);

      srand(12345); // Use a standard seed value for reproducibility

      // This assumes A is stored by rows, and B is stored by columns. Other storage schemes are permitted.
      for (int i = 0; i < sizeAB; i++) A[i] = ((double) rand() / (double) RAND_MAX);
      for (int i = 0; i < sizeAB; i++) B[i] = ((double) rand() / (double) RAND_MAX);

      MPI_Barrier(MPI_COMM_WORLD);
      timing(&wcs, &cputime);

      int row, col, offset; // row/col index
      // partition A into p row blocks, and assign each to one of the p MPI processes
      for (int i = 1; i < size; i++) {
        row = i * delta;
        offset = row * (row + 1) / 2; // 1 + ... + row
        // (row + 1) + ... + (row + delta)
        MPI_Send(A + offset, block_size(row, delta), MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
      }

      // partition B into p column blocks, and assign each to one of the p MPI processes
      for (int i = 1; i < size; i++) { // initial assignment
        col = i * delta;
        offset = col * (col + 1) / 2;
        MPI_Send(&col, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        MPI_Send(B + offset, block_size(col, delta), MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
      }

      // compute in the master process
      compTime += matmul_block(N, A, B, C, 0, 0, delta);

      // rotate column block and compute
      col = 0;
      double *tempB = B; // needed to be able to free B after the rotation      
      for (int i = 1; i < size; i++) {  
        // Send current column index and block to the next process (1)
        MPI_Send(&col, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
        MPI_Send(tempB, block_size(col, delta), MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
        // Recv column index and block from the previous process (size - 1)
        MPI_Recv(&col, 1, MPI_INT, size - 1, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(tempB, block_size(col, delta), MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD, &status);
        compTime += matmul_block(N, A, tempB, C, 0, col, delta);
      }

      // recv results from non-master processes
      for (int i = 1; i < size; i++) {
        MPI_Recv(C + i * delta * N, delta * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
      }

      timing(&wce, &cputime);
      commTime = wce - wcs - compTime;

      // Remainder of the code checks the result against the correct answer (read into Ctrue)
      Ctrue = (double *) calloc(sizeC, sizeof(double));
      fptr = fopen(files[run], "rb");
      fread(Ctrue, sizeof(double), sizeC, fptr);
      fclose(fptr);    

      // Compute the Frobenius norm of Ctrue - C
      Fnorm = 0.;
      for (int i = 0; i < sizeC; i++) Fnorm += (Ctrue[i] - C[i]) * (Ctrue[i] - C[i]);
      Fnorm = sqrt(Fnorm);

      free(A);
      free(B);
      free(C);
      free(Ctrue);

      // Print a table row
      if (argc == 1) { // Task 2 Part A
        printf ("%5d      %9.4f \n", 0, wce - wcs);
      }
      else { // Task 2 Part B
        printf ("%5d      %9.4f      %9.4f      %9.4f \n", 0, compTime, commTime, wce - wcs);
      }
      printf("F-norm of Error: %15.10f \n", Fnorm);
    } 
    else { // other processes
      int row = rank * delta, col = rank;

      MPI_Barrier(MPI_COMM_WORLD);
      timing(&wcs, &cputime);

      // recv from master process
      MPI_Recv(A, block_size(row, delta), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
      MPI_Recv(&col, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
      MPI_Recv(B, block_size(col, delta), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

      // compute in the current process
      compTime += matmul_block(N, A, B, C, row, col, delta);

      // rotate column block and compute
      int tempCol;
      double *tempB = (double *) calloc(sizeAB, sizeof(double));
      for (int i = 1; i < size; i++) {
        // Recv column index and block from the previous process
        MPI_Recv(&tempCol, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(tempB, block_size(tempCol, delta), MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status);
        // Send current column index and block to the next process
        MPI_Send(&col, 1, MPI_INT, (rank + 1) % size, 1, MPI_COMM_WORLD);
        MPI_Send(B, block_size(col, delta), MPI_DOUBLE, (rank + 1) % size, 1, MPI_COMM_WORLD);

        col = tempCol;
        memcpy(B, tempB, sizeof(double) * block_size(col, delta));
        compTime += matmul_block(N, A, B, C, row, col, delta);
      }

      // send result to master process
      MPI_Send(C, delta * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

      timing(&wce, &cputime);
      commTime = wce - wcs - compTime;

      free(A);
      free(B);
      free(tempB);
      free(C);

      if (argc == 1) { // Task 2 Part A
        printf ("%5d      %9.4f \n", rank, wce - wcs);
      }
      else { // Task 2 Part B
        printf ("%5d      %9.4f      %9.4f      %9.4f \n", rank, compTime, commTime, wce - wcs);
      }
    }
  }

  MPI_Finalize();
}



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double matmul(int, double*, double*, double*);

int main(int argc, char **argv) {
  /*
    This is the serial main program for CPSC424/524 Assignment #3.
    Author: Andrew Sherman, Yale University
    Date: 3/18/2018
  */
  int N, i, run;
  double *A, *B, *C, *Ctrue;
  long sizeAB, sizeC;

  // This array contains the sizes of the test cases
  int sizes[4] = { 1000, 2000, 4000, 8000 };
  // This array contains the file names for the true answers
  char files[4][50] = { "/home/fas/cpsc424/ahs3/assignment3/C-1000.dat", \
                        "/home/fas/cpsc424/ahs3/assignment3/C-2000.dat", \
                     	  "/home/fas/cpsc424/ahs3/assignment3/C-4000.dat", \
                     	  "/home/fas/cpsc424/ahs3/assignment3/C-8000.dat" };
  double wctime, Fnorm;
  FILE *fptr;

  // Print a table heading
  printf("Matrix multiplication times: \n "); 
  printf("   N      TIME (secs)    F-norm of Error \n");
  printf(" -----   -------------  ----------------- \n");

  // Now run the four test cases
  for (run = 0; run < 4; run++) {
    N = sizes[run];

    sizeAB = N * (N + 1) / 2; // Only enough space for the non-zero portions of the matrices
    sizeC = N * N; // All of C will be non-zero, in general!

    A = (double *) calloc(sizeAB, sizeof(double));
    B = (double *) calloc(sizeAB, sizeof(double));
    C = (double *) calloc(sizeC, sizeof(double));
  
    srand(12345); // Use a standard seed value for reproducibility

    // This assumes A is stored by rows, and B is stored by columns. Other storage schemes are permitted
    for (i = 0; i < sizeAB; i++) A[i] = ((double) rand() / (double) RAND_MAX);
    for (i = 0; i < sizeAB; i++) B[i] = ((double) rand() / (double) RAND_MAX);

    // Time the serial matrix multiplication computation
    wctime = matmul(N, A, B, C);

    free(A);
    free(B);

    // Remainder of the code checks the result against the correct answer (read into Ctrue)
    Ctrue = (double *) calloc(sizeC, sizeof(double));

    fptr = fopen(files[run], "rb");
    fread(Ctrue, sizeof(double), sizeC, fptr);
    fclose(fptr);    

    // Compute the Frobenius norm of Ctrue - C
    Fnorm = 0.;
    for (i = 0; i < sizeC; i++) Fnorm += (Ctrue[i] - C[i]) * (Ctrue[i] - C[i]);
    Fnorm = sqrt(Fnorm);

    // Print a table row
    printf (" %5d    %9.4f    %15.10f\n", N, wctime, Fnorm);

    free(Ctrue);  
    free(C);
  }

}




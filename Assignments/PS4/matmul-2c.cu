#define FP float

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

// A: n * p, B: p * m, C: n * m
__global__ void gpu_matrixmult(FP *a, FP *b, FP *c, int n, int p, int m, int TW) {
  extern __shared__ FP bigarray[];
  FP *atile = &bigarray[0], *btile = &bigarray[TW * TW];

  int tx = threadIdx.x; int ty = threadIdx.y; FP cvalue = 0.;
  // current thread is responsible for computing the entry at (row, col) of C
  int col = tx + blockDim.x * blockIdx.x;
  int row = ty + blockDim.y * blockIdx.y;

  for (int tile = 0; tile < p / TW; tile++) {
    int indexa = row * p + tile * TW + tx;   // current thread caches this entry of A
    int indexb = (tile * TW + ty) * m + col; // current thread caches this entry of B

    atile[ty * TW + tx] = (tile * TW + tx < p && row < n) ? a[indexa] : 0.;
    btile[ty * TW + tx] = (tile * TW + ty < p && col < m) ? b[indexb] : 0.;
    __syncthreads(); // make sure tiles are loaded before all threads can use them

    for (int k = 0; k < TW; k++) cvalue += atile[ty * TW + k] * btile[k * TW + tx];
    __syncthreads(); // make sure we’re done using the tiles before overwriting them
  }

  if (col < m && row < n) {
    c[row * m + col] = cvalue;
  }
}

int main(int argc, char *argv[]) {
  int i, j; // loop counters

  int gpucount = 0; // Count of available GPUs
  int gpunum = 0; // Device number to use
  int Grid_Dim_x = 1; // Grid dimension, x 
  int Grid_Dim_y = 1; // Grid dimension, y
  int Block_Dim_x = 1; // Block dimension, x 
  int Block_Dim_y = 1; // Block dimension, y
  int TW;

  int n, p, m; // matrix dimension (A: n * p, B: p * m, C: n * m)
  FP *a, *b, *c;
  FP *dev_a, *dev_b, *dev_c;
  size_t sizeA, sizeB, sizeC; // number of bytes in arrays

  cudaEvent_t start, stop; // using cuda events to measure time
  float elapsed_time_ms; // which is applicable for asynchronous code also
  cudaError_t errorcode;

  // -------------------- SET PARAMETERS AND DATA -----------------------

  errorcode = cudaGetDeviceCount(&gpucount);
  if (errorcode == cudaErrorNoDevice) {
    printf("No GPUs are visible\n");
    exit(-1);
  }
  else {
    printf("Device count = %d\n", gpucount);
  }

  if (argc != 5) {
    printf("Usage: matmul-1a <matrix dim n> <matrix dim p> <matrix dim m> <block dim> \n");
    exit (-1);  
  }

  n = atoi(argv[1]);
  p = atoi(argv[2]);
  m = atoi(argv[3]);

  Block_Dim_x = atoi(argv[4]); // Square block
  Block_Dim_y = Block_Dim_x;
  TW = Block_Dim_x; // Simplest case: TW = Block_Dim_x = Block_Dim_y
  if (Block_Dim_x * Block_Dim_y > 1024) {
    printf("Error, too many threads in block\n");
    exit (-1);
  }

  Grid_Dim_x = (m - 1) / Block_Dim_x + 1;
  Grid_Dim_y = (n - 1) / Block_Dim_y + 1;
  if (Grid_Dim_x * Block_Dim_x < m || Grid_Dim_y * Block_Dim_y < n) {
    printf("Error, number of threads in x/y dimensions less than number of array elements\n");
    exit(-1);
  }

  cudaSetDevice(gpunum);
  printf("Using device %d\n", gpunum);
  
  printf("Matrix Dimension = A (%d, %d), B (%d, %d), C (%d, %d) \n", n, p, p, m, n, m);
  printf("Block_Dim = (%d, %d), Grid_Dim = (%d, %d) \n", Block_Dim_x, Block_Dim_y, Grid_Dim_x, Grid_Dim_y);

  dim3 Grid(Grid_Dim_x, Grid_Dim_y); // Grid structure
  dim3 Block(Block_Dim_x, Block_Dim_y); // Block structure

  sizeA = n * p * sizeof(FP);
  sizeB = p * m * sizeof(FP);
  sizeC = n * m * sizeof(FP);

  a = (FP *) malloc(sizeA); // dynamically allocated memory for arrays on host
  b = (FP *) malloc(sizeB);
  c = (FP *) malloc(sizeC); // results from GPU

  srand(12345);
  for (i = 0; i < n; i++) {
    for (j = 0; j < p; j++) {
      a[i * p + j] = (FP) rand() / (FP) RAND_MAX;
    }
  }

  for (i = 0; i < p; i++) {
    for (j = 0; j < m; j++) {
      b[i * m + j] = (FP) rand() / (FP) RAND_MAX;
    }
  }

  // ------------- COMPUTATION DONE ON GPU ----------------------------

  cudaMalloc((void**) &dev_a, sizeA); // allocate memory on device
  cudaMalloc((void**) &dev_b, sizeB);
  cudaMalloc((void**) &dev_c, sizeC);

  cudaMemcpy(dev_a, a, sizeA, cudaMemcpyHostToDevice); // copy from CPU tp GPU
  cudaMemcpy(dev_b, b, sizeB, cudaMemcpyHostToDevice);

  cudaEventCreate(&start); // instrument code to measure start time
  cudaEventCreate(&stop);
  
  cudaEventRecord(start, 0);
  // cudaEventSynchronize(start); // not needed

  size_t Ns = 2 * TW * TW * sizeof(FP); // amount of shared momery
  gpu_matrixmult<<<Grid, Block, Ns>>>(dev_a, dev_b, dev_c, n, p, m, TW);

  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  cudaMemcpy(c, dev_c, sizeC, cudaMemcpyDeviceToHost); // copy from GPU to CPU

  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time

  // ----------------------------- clean up ------------------------------

  free(a);
  free(b);
  free(c);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}



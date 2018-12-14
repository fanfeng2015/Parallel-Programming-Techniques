#define FP float

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

// A: n * p, B: p * m, C: n * m
__global__ void gpu_matrixmult(FP *a, FP *b, FP *c, int n, int p, int m) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  int indexb = col;
  int index = row * m + col;
  
  if (col < m && row < n) {
    c[index] = 0.;
    for (int indexa = row * p; indexa < (row * p + p); indexa++, indexb += m) 
      c[index] += a[indexa] * b[indexb];
  }
}

// A: n * p, B: p * m, C: n * m
// Serial CPU code of the kij variant of matrix multiplication.
void cpu_matrixmult_kij(FP* a, FP* b, FP* c, int n, int p, int m) {
	for (int k = 0; k < p; k++) {
		for (int i = 0; i < n; i++) {
			FP r = a[i * p + k]; // A[i][k]
			for (int j = 0; j < m; j++) {
				c[i * m + j] += r * b[k * m + j]; // B[k][j], C[i][j]
			}
		}
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

  int n, p, m; // matrix dimension (A: n * p, B: p * m, C: n * m)
  FP *a, *b, *c, *kijc; // kijc: matrix C of the kij variant
  FP *dev_a, *dev_b, *dev_c;
  int sizeA, sizeB, sizeC; // number of bytes in arrays

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
  if (Block_Dim_x * Block_Dim_y > 1024) {
    printf("Error, too many threads in block\n");
    exit (-1);
  }

  Grid_Dim_x = m / Block_Dim_x;
  Grid_Dim_y = n / Block_Dim_y;
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
  kijc = (FP *) malloc(sizeC); // results from CPU

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

  gpu_matrixmult<<<Grid, Block>>>(dev_a, dev_b, dev_c, n, p, m);

  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  cudaMemcpy(c, dev_c, sizeC, cudaMemcpyDeviceToHost); // copy from GPU to CPU

  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time

  // ------------- COMPUTATION DONE ON HOST CPU (kij variant) ----------------------------

  cudaEventRecord(start, 0); // use same timing

  cpu_matrixmult_kij(a, b, kijc, n, p, m);

  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  printf("Time to calculate results on CPU (kij variant): %f ms.\n", elapsed_time_ms); // exec. time

  // ------------------- check device creates correct results -----------------

  double error, suma, sumb, sumc, ai, bi, ci;
  suma = 0.; sumb = 0; sumc = 0;
  for(i = 0; i < n * p; i++) {
    ai = (double) a[i];
    suma += ai * ai;
  }
  for(i = 0; i < p * m; i++) {
    bi = (double) b[i];
    sumb += bi * bi;
  }
  for(i = 0; i < n * m; i++) {
    ci = (double) (c[i] - kijc[i]);
    sumc += ci * ci;
  }

  suma = sqrt(suma);
  sumb = sqrt(sumb);
  sumc = sqrt(sumc);
  error = sumc / (n * suma * sumb);
  printf("Scaled error between GPU and CPU: %e\n", error);

  // ----------------------------- clean up ------------------------------

  free(a);
  free(b);
  free(c);
  free(kijc);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}



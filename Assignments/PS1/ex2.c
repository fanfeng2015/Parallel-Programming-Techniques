#include <stdio.h>
#include <math.h>
#include "timing.h"
#include "dummy.h"

// Author: Fan Feng
// NetId: ff242
// Email: fan.feng@yale.edu

double* initialize(int N) {
	double* array = (double *) malloc(N * sizeof(double));
	for (int i = 0; i < N; i++) {
		array[i] = 100 * (double) rand() / (double) RAND_MAX;	
	}
	return array;
}

int main() {
	int repeat;
	double wcs, wce, cputime, runtime;
	double *a, *b, *c, *d;
	for (int k = 3; k <= 25; k++) {
		long N = floor(pow(2.1, k));
		a = initialize(N), b = initialize(N), c = initialize(N), d = initialize(N);
		repeat = 1;
		runtime = 0.0;
		while (runtime < 1.0) {
			timing(&wcs, &cputime);
			for (int r = 0; r < repeat; r++) {
				// kernel benchmark loop
				for (int i = 0; i < N; i++) {
					a[i] = b[i] * c[i] + d[i];
				}
			}
			if (a[N >> 1] < 0) {
				dummy(); // fools the compiler
			}
			timing(&wce, &cputime);
			runtime = wce - wcs;
			repeat *= 2;
		}
		repeat / 2;
		printf("N = %ld, MFlops/s = %lf \n", N, (double) 2 * N * repeat / runtime / 1000000);
	}
	free(a);
	free(b);
	free(c);
	free(d);
	return 0;
}

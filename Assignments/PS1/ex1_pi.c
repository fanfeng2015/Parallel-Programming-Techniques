#include <stdio.h>
#include <math.h>
#include "timing.h"

// Author: Fan Feng
// NetId: ff242
// Email: fan.feng@yale.edu

int main() {
	double wcs, wce, cputime;
	int N = 1000000000;
	double xi = 1.0 / (2 * N);
	double dx = 1.0 / N;
	double sum = 0.0;
	timing(&wcs, &cputime);
	for (int i = 0; i < N; i++) { // 5 floating point operations per iteration
		sum += dx / (1 + xi * xi);
		xi += dx;
	}
	timing(&wce, &cputime);
	printf("Pi = %lf, sin(Pi) = %lf \n", 4 * sum, sin(4 * sum));
	printf("Time spent = %lf \n", wce - wcs);
	printf("MFlops/s = %lf \n", (double) 5 * N / 1000000 / (wce - wcs));
	return 0;
}

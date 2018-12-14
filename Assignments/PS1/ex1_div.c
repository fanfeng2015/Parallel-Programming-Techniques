#include <stdio.h>
#include "timing.h"

// Author: Fan Feng
// NetId: ff242
// Email: fan.feng@yale.edu

int main() {
	double wcs, wce, cputime;
	int N = 1000000000;
	double temp = 1.0;
	timing(&wcs, &cputime);
	for (int i = 0; i < N; i++) { // 1 division operation per iteration
		temp = 1.0 / i;
	}
	timing(&wce, &cputime);
	printf("Time spent = %lf \n", wce - wcs);
	printf("Number of division operations = %d \n", N);
	return 0;
}

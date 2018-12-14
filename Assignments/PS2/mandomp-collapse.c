#include <inttypes.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include "drand.h"
#include "timing.h"

int main() {
	dsrand(12345);
	double wcs, wce, cputime;
	timing(&wcs, &cputime);
	int N0 = 0, N1 = 0;
	
	int num_threads = (int) strtoimax(getenv("OMP_NUM_THREADS"), NULL, 10);
	omp_set_num_threads(num_threads);
	printf("Number of threads = %d \n", num_threads);

	#pragma omp parallel for default(none) schedule(runtime) collapse(2) reduction(+:N0) reduction(+:N1)
	for (int x = -2000; x < 500; x++) {
		for (int y = 0; y < 1250; y++) {
			double creal = ((double) x + drand()) / 1000;
			double cimgn = ((double) y + drand()) / 1000;
			double zreal = creal;
			double zimgn = cimgn;
			int i = 0;
			for (i = 0; i < 20000; i++) {
				double next_zreal = zreal * zreal - zimgn * zimgn + creal;
				double next_zimgn = 2 * zreal * zimgn + cimgn;
				zreal = next_zreal;
				zimgn = next_zimgn;
				if (zreal * zreal + zimgn * zimgn > 4) {
					break;
				}
			}
			if (i < 20000) {
				N0++;
			}
			else {
				N1++;
			}
		}
	}
	timing(&wce, &cputime);
	printf("Time spent = %lf. ", wce - wcs);
	printf("Area = %lf. \n", 2 * 3.125 * N1 / (N0 + N1));
	return 0;
}

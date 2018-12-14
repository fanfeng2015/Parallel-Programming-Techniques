#include "drand.h"

void dsrand(unsigned s) {
	seed = s - 1;
    // printf("Seed = %lu. RAND_MAX = %d.\n", seed, RAND_MAX);
}

double drand(void) {
	seed = 6364136223846793005ULL * seed + 1;
    return((double) (seed >> 33) / (double) RAND_MAX);
}

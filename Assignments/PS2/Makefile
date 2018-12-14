# This makes executables and a random number generator for use in Assignment 2
# Use module load Langs/Intel/15.0.2

CC = icc
CFLAGS = -g -O3 -xHost -fno-alias -std=c99
EXECUTABLES = mandseq mandomp mandomp-schedule mandomp-collapse mandomp-tasks mandomp-tasks-row mandomp-tasks-allthreads
TIMING = /home/fas/cpsc424/ahs3/utils/timing/timing.o 

all: $(EXECUTABLES)

mandseq: mandseq.o drand.o $(TIMING)  
	$(CC) $(CFLAGS) -o $@ $^ -lm
mandomp: mandomp.o drand.o $(TIMING)
	$(CC) $(CFLAGS) -openmp -o $@ $^ -lm
mandomp-schedule: mandomp-schedule.o drand.o $(TIMING)
	$(CC) $(CFLAGS) -openmp -o $@ $^ -lm
mandomp-collapse: mandomp-collapse.o drand.o $(TIMING)
	$(CC) $(CFLAGS) -openmp -o $@ $^ -lm
mandomp-tasks: mandomp-tasks.o drand.o $(TIMING)
	$(CC) $(CFLAGS) -openmp -o $@ $^ -lm
mandomp-tasks-row: mandomp-tasks-row.o drand.o $(TIMING)
	$(CC) $(CFLAGS) -openmp -o $@ $^ -lm
mandomp-tasks-allthreads: mandomp-tasks-allthreads.o drand.o $(TIMING)
	$(CC) $(CFLAGS) -openmp -o $@ $^ -lm

mandseq.o: mandseq.c
	$(CC) $(CFLAGS) -c $<
drand.o: drand.c
	$(CC) $(CFLAGS) -c $<
.c.o:
	$(CC) $(CFLAGS) -openmp -c $<

clean:
	rm -f $(EXECUTABLES) *.o

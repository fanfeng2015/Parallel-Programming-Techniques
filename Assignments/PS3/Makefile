# NOTE: This Makefile presumes that the following command has been executed:

# module load Langs/Intel/15.0.2 MPI/OpenMPI/2.1.1-intel15

TIMINGDIR = /home/fas/cpsc424/ahs3/utils/timing
CC = mpicc
CFLAGS = -g -O3 -xHost -fno-alias -std=c99 -I $(TIMINGDIR)
EXECUTABLES = serial mpi-blocking mpi-load-balance

serial:	serial.o matmul.o $(TIMINGDIR)/timing.o
	$(CC) -o $@ $(CFLAGS) $^
mpi-blocking: mpi-blocking.o matmul-block.o $(TIMINGDIR)/timing.o
	$(CC) -o $@ $(CFLAGS) $^
mpi-non-blocking: mpi-non-blocking.o matmul-block.o $(TIMINGDIR)/timing.o
	$(CC) -o $@ $(CFLAGS) $^
mpi-load-balance: mpi-load-balance.o matmul-block-load-balance.o $(TIMINGDIR)/timing.o
	$(CC) -o $@ $(CFLAGS) $^

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(EXECUTABLES) *.o


CC = icc
CFLAGS1 = -g -O0 -fno-alias -std=c99
CFLAGS2 = -g -O1 -fno-alias -std=c99
CFLAGS3 = -g -O3 -no-vec -no-simd -fno-alias -std=c99
CFLAGS4 = -g -O3 -xHost -fno-alias -std=c99

all: ex1_pi ex1_div ex2

ex1_pi: ex1_pi.c timing.o
	${CC} ${CFLAGS1} -o ex1_a ex1_pi.c timing.o
	${CC} ${CFLAGS2} -o ex1_b ex1_pi.c timing.o
	${CC} ${CFLAGS3} -o ex1_c ex1_pi.c timing.o
	${CC} ${CFLAGS4} -o ex1_d ex1_pi.c timing.o

ex1_div: ex1_div.c timing.o
	${CC} ${CFLAGS1} -o ex1_div ex1_div.c timing.o

ex2: ex2.c dummy.c timing.o
	${CC} ${CFLAGS4} -o ex2 ex2.c dummy.c timing.o
	
clean:
	rm ex1_a ex1_b ex1_c ex1_d ex1_div ex2

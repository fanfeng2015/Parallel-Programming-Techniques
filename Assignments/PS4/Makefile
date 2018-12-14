# This Makefile assumes the following module files are loaded:
#
# Langs/Intel/15
# GPU/Cuda/8.0
#
# This Makefile will only work if executed on a GPU node.
#

CUDAPATH = /gpfs/apps/hpc/GPU/Cuda/8.0
NVCC = $(CUDAPATH)/bin/nvcc
NVCCFLAGS = -I$(CUDAPATH)/include -O3
LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm
# Compiler-specific flags (by default, we always use sm_35)
GENCODE_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE = $(GENCODE_SM35)
BINARIES = matmul matmul-1a matmul-1b matmul-1c matmul-2a matmul-2b matmul-2c matmul-3

.SUFFIXES: .cu .ptx

matmul: matmul.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<
matmul-1a: matmul-1a.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<
matmul-1b: matmul-1b.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<
matmul-1c: matmul-1c.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<
matmul-2a: matmul-2a.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<
matmul-2b: matmul-2b.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<
matmul-2c: matmul-2c.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<
matmul-3: matmul-3.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

.cu.o:
	$(NVCC) $(GENCODE) $(NVCCFLAGS) -o $@ -c $<

clean:	
	rm -f *.o $(BINARIES)

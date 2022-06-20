SHELL:=/bin/bash
CC=dpcpp
CFLAGS=-O2 -Wall
LDFLAGS=
INCLUDES=-Iincludes
SRC_FILES=main.c dw_soa_sycl.cpp read_in.c
EXEC_BASENAME=main

oneapi_intel_cpu: CC = dpcpp

oneapi_cuda_gpu: CC = clang++
oneapi_cuda_gpu: CFLAGS += -fsycl -fsycl-targets=nvptx64-cuda-cuda

hip_omp_cpu: CC = syclcc
#hip_omp_cpu: CFLAGS += --hipsycl-targets=omp

hip_cuda_gpu: CC = syclcc
#hip_cuda_gpu: CFLAGS += --hipsycl-targets=cuda:sm_80

.PHONY: all
all: oneapi_targets hip_targets

.PHONY: oneapi_targets
oneapi_targets: oneapi_intel_cpu oneapi_cuda_gpu

.PHONY: hip_targets
hip_targets: hip_omp_cpu hip_cuda_gpu

# Ensure the compiler is in $PATH and correctly set up before compiling, for example:
# source setvars.sh script (from oneapi installation) before making the 'oneapi_intel_cpu' target
oneapi_intel_cpu oneapi_cuda_gpu hip_omp_cpu hip_cuda_gpu: $(SRC_FILES)
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o $(EXEC_BASENAME).$@

.PHONY: clean
clean:
	rm -f *.o
	rm -f $(EXEC_BASENAME).oneapi_intel_cpu
	rm -f $(EXEC_BASENAME).oneapi_cuda_gpu
	rm -f $(EXEC_BASENAME).hip_omp_cpu
	rm -f $(EXEC_BASENAME).hip_cuda_gpu

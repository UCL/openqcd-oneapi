SHELL:=/bin/bash
CC=dpcpp
CFLAGS=-O2 -Wall
LDFLAGS=
INCLUDES=-Iincludes

oneapi_intel_cpu: CC = dpcpp

oneapi_cuda_gpu: CC = clang++
oneapi_cuda_gpu: CFLAGS += -fsycl -fsycl-targets=nvptx64-cuda-cuda

hip_omp_cpu: CC = syclcc
#hip_omp_cpu: CFLAGS += --hipsycl-targets=omp

hip_cuda_gpu: CC = syclcc
#hip_cuda_gpu: CFLAGS += --hipsycl-targets=cuda:sm_80

all: oneapi_intel_cpu oneapi_cuda_gpu hip_intel_cpu hip_intel_gpu

oneapi_intel_cpu: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main.oneapi_intel_cpu

oneapi_cuda_gpu: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main.oneapi_cuda_gpu

hip_omp_cpu: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main.hip_omp_cpu

hip_cuda_gpu: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main.hip_cuda_gpu

clean:
	rm -f *.o main a.out 2>/dev/null

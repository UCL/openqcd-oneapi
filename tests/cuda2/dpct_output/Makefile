SHELL:=/bin/bash
CC=dpcpp
CFLAGS=-O2 -Wall
LDFLAGS=
INCLUDES=-Iincludes

intel_cpu: CC = dpcpp

nvidia_gpu: CC = clang++
nvidia_gpu: CFLAGS += -fsycl -fsycl-targets=nvptx64-nvidia-cuda

hip_gpu: CC = syclcc
hip_gpu: CFLAGS += -v

all: intel_cpu nvidia_gpu

intel_cpu: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main.intel_cpu

nvidia_gpu: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main.nvidia_gpu

hip_gpu:  main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main.hip_gpu

clean:
	rm -f *.o main a.out 2>/dev/null

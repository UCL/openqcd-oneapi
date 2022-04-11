SHELL:=/bin/bash
CC=dpcpp
CFLAGS=-O2 -Wall
LDFLAGS=
INCLUDES=-Iincludes

intel: CC = dpcpp

nvidia: CC = clang++
nvidia: CFLAGS += -fsycl -fsycl-targets=nvptx64-nvidia-cuda

all: intel_cpu nvidia_gpu

intel_cpu: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main.intel_cpu

nvidia_gpu: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main.nvidia_gpu

clean:
	rm -f *.o main a.out 2>/dev/null

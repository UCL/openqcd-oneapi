SHELL:=/bin/bash
CC=dpcpp
CFLAGS=-O2 -Wall
LDFLAGS=
INCLUDES=-Iincludes

intel: CC = dpcpp

nvidia: CC = clang++
nvidia: CFLAGS += -fsycl -fsycl-targets=nvptx64-nvidia-cuda

all: intel nvidia

intel: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main

nvidia: main.c dw_soa_sycl.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o main

clean:
	rm -f *.o main a.out 2>/dev/null

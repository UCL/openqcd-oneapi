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
	module purge;                                                                                     \
	source /usr/local/software/intel/oneapi/2022.1/setvars.sh >/dev/null 2>&1;                        \
    $(CC) $(INCLUDES) $(CFLAGS) $^ -o main

nvidia: main.c dw_soa_sycl.cpp read_in.c
	module purge;                                                                                     \
    source /usr/local/software/intel/oneapi/2022.1/setvars.sh >/dev/null 2>&1;                        \
    module use /usr/local/software/spack/spack-modules/dpcpp-cuda-20220220/linux-centos8-x86_64_v3/;  \
    module load dpcpp;                                                                                \
    module load gcc/11.2.0;                                                                           \
    $(CC) $(INCLUDES) $(CFLAGS) $^ -o main


clean:
	rm -f *.o main a.out 2>/dev/null

logclean:
	rm -rf logs_of_job_output_and_error/*


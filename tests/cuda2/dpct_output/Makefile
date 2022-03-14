CC=dpcpp
CFLAGS=-O2 -Wall
LDFLAGS=

INCLUDES=-Iincludes

all: main

main: main.c dw_cuda_soa.dp.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o $@


clean:
	rm -f *.o main a.out 2>/dev/null

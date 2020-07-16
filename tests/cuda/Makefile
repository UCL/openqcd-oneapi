CC=gcc
CFLAGS=-c -std=c99 -Wall -gdwarf-2 -g3 -Og
LDFLAGS=-g

CUDAC=nvcc
CUDACFLAGS=-c -arch=sm_20 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -g# -std=c++11
CUDALFLAGS=-lcuda -lcudart -lm -lstdc++

INCLUDES=-Iincludes

OBJECTS=main.o
CUDAOBJECTS=dw_cuda.o

all: main

main: $(OBJECTS) $(CUDAOBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS) $(CUDALFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

%.o: %.cu
	$(CUDAC) $(CUDACFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f *.o main

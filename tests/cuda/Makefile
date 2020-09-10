CC=gcc
CFLAGS=-c -std=c99 -O2 -Wall
LDFLAGS=

CUDAC=nvcc
CUDACFLAGS=-c -arch=sm_70 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__
CUDALFLAGS=-lcuda -lcudart -lm -lstdc++

INCLUDES=-Iincludes

OBJECTS=main.o read_in.o
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

CC=dpcpp
CFLAGS=-c -O2 -Wall
LDFLAGS=

INCLUDES=-Iincludes

OBJECTS=main.o read_in.o
SYCLOBJECTS=dw_cuda_soa.dp.o

all: main

main: $(OBJECTS) $(SYCLOBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

%.o: %.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f *.o main

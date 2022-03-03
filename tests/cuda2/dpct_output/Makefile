CC=dpcpp
CFLAGS=-O2 -Wall
LDFLAGS=

INCLUDES=-Iincludes

all: main

main: main.c dw_cuda_soa.dp.cpp read_in.c
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o $@

# OBJECTS=main.o read_in.o
# SYCLOBJECTS=dw_cuda_soa.dp.o

# main: main.c dw_cuda_soa.dp.cpp read_in.c
# 	$(CC) $(INCLUDES) $(CFLAGS) $< -o $@

# main: $(OBJECTS) $(SYCLOBJECTS)
# 	$(CC) -o $@ $^ $(LDFLAGS)
#
# %.o: %.c
# 	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@
#
# %.o: %.cpp
# 	$(CC) $(CCFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f *.o main a.out 2>/dev/null

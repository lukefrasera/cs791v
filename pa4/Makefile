#Makefile

# all: main

# main: vector_test.cu vector_sum.cu:
# 	nvcc $^ -arch=sm_35 -rdc=true -o $@ -lcudadevrt
# -arch=sm_35 -rdc=true -lcudadevrt
CC=nvcc
CFLAGS=-arch=sm_35 -rdc=true -lcudadevrt
LDFLAGS=-arch=sm_35 -rdc=true -lcudadevrt
SOURCES=src/vector_tests.cu src/vector_sum.cu
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=sum

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

# Compiler
NVCC = nvcc

# Flags (add -O3 for optimization if needed)
CFLAGS = -O3

# Source files
SRC = main.cu particle_filter.cu

# Output executable name
TARGET = particle_filter

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

# Compiler and Linker
CC = g++
LD = g++

# Compiler Flags
CCFLAGS += -g -O3 -std=c++17 
CCFLAGS += -Wall -Wextra -Wpedantic

# Eigen
CCFLAGS += -I $(LIB)/Eigen/include

# Directories
SRC = $(wildcard src/*.cpp)
OBJ = $(SRC:.cpp=.o)
BIN = bin
LIB = lib

# Routines
all: main clean

main: $(OBJ)
	$(LD) -o $(BIN)/main.exe $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) -o $@ -c $< $(CCFLAGS)

clean:
	@powershell "rm src/*.o"

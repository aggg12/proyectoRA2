# Define paths
ALE_DIR := ./Arcade-Learning-Environment-0.6.1
SRC_DIR := ./src

# Compiler options
CXX := g++
# We add SDL include path and -D__USE_SDL
CXXFLAGS := -std=c++11 -O3 -Wall -D__USE_SDL \
            -I/usr/include/SDL \
            -I$(ALE_DIR)/src \
            -I$(ALE_DIR)/src/controllers \
            -I$(ALE_DIR)/src/os_dependent \
            -I$(ALE_DIR)/src/environment \
            -I$(ALE_DIR)/src/external \
            -I$(SRC_DIR)

# Linker options
# Added -lSDL and ensured -lz for compression
LDFLAGS := -L$(ALE_DIR) -lale -lSDL -lSDLmain -lz -Wl,-rpath=$(ALE_DIR)

# Sources and Targets
COMMON_SRCS := $(SRC_DIR)/RamExtractor.cpp
COMMON_OBJS := $(COMMON_SRCS:.cpp=.o)

TARGETS :=  manual_check

all: $(TARGETS)

manual_check: $(SRC_DIR)/manual_check.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)



%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

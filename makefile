CXX=g++
CXXFLAGS=-std=c++17 -O3 -DNDEBUG

all: cbow

cbow : src/cbow.cpp
	$(CXX) src/cbow.cpp -o cbow $(CXXFLAGS) -pthread

clean:
	rm -rf cbow

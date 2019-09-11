CXX=g++
CXXFLAGS=-std=c++17 -O3 -DNDEBUG

all: cbow skip-gram

cbow : src/cbow.cpp
	$(CXX) src/cbow.cpp -o cbow $(CXXFLAGS) -pthread
skip-gram : src/sg.cpp
	$(CXX) src/sg.cpp -o skip-gram $(CXXFLAGS) -pthread

clean:
	rm -rf cbow skip-gram

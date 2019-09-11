CXX=g++
CXXFLAGS=-std=c++17 -O3 -DNDEBUG

all: cbow skip-gram build_dict distance

cbow : src/cbow.cpp
	$(CXX) src/cbow.cpp -o cbow $(CXXFLAGS) -pthread
skip-gram : src/sg.cpp
	$(CXX) src/sg.cpp -o skip-gram $(CXXFLAGS) -pthread
build_dict : src/build_dict.cpp
	$(CXX) src/build_dict.cpp -o build_dict $(CXXFLAGS)
distance : src/distance.cpp
	$(CXX) src/distance.cpp -o distance $(CXXFLAGS)

clean:
	rm -rf cbow skip-gram build_dict distance

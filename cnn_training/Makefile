all: example1 example2 getParameters

example1:
	$(CXX) -g src/example1.cpp -I . -o example1 -O3

example2:
	$(CXX) -g src/example2.cpp -I . -o example2 -O3

getParameters:
	$(CXX) -g src/getParameters.cpp -I . -o getParameters -O3

clean:
	rm -f example1 example2 getParameters
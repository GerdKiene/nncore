# make the shit out of that c++

simulator: simulator.cc
	g++ -std=c++11 -o simulator simulator.cc #common_functions.cc

run:
	g++ -std=c++11 -o simulator simulator.cc
	python generate_config.py
	./simulator

clean:
	rm simulator

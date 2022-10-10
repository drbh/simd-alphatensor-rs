SHELL=/bin/bash

get-data:
	wget \
		-O data/factorizations_r.npz \
		https://raw.githubusercontent.com/deepmind/alphatensor/main/algorithms/factorizations_r.npz 

gen:
	python3 scripts/print_equations_v2.py > data/algorithms.txt
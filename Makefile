all: main

main: fft.c
	gcc -fopenmp fft.c -o fft -O3 -Wall -lm -march=native
	# gcc -fopt-info -fopenmp fft.c -o fft -O3 -Wall -lm

sequential: fft_initial_structure.c
	gcc fft_initial_structure.c -o fft -lm

parallel: fft_initial_structure.c
	gcc fft_initial_structure.c -o fft -fopenmp -lm -Wall

clean:
	rm -f fft *.wav

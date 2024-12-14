#!/bin/bash

# compile the code
mpicxx matrix-inverse.cpp

# create a random test case
python create.py

# for num in range 1 to 12, call ../test.sh 'mpiexec -n ${num} ./a.out inp1.txt' 'inp1-opt.txt'
for num in {1..12}
do
    echo -n "$num " 
    bash ../test-file.sh "mpiexec -n ${num} ./a.out random.txt" "random-opt.txt"
done

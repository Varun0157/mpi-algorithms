#!/bin/bash

# compile the code
mpicxx 1.cpp

# for num in range 1 to 12, call ../test.sh 'mpiexec -n ${num} ./a.out inp1.txt' 'inp1-opt.txt'
for num in {1..12}
do
    start=`date +%s%N`
    bash ../test-file.sh "mpiexec -n ${num} ./a.out inp1.txt" "inp1-opt.txt"
    end=`date +%s%N`
    echo execution time : `expr $end - $start` nanoseconds.
done

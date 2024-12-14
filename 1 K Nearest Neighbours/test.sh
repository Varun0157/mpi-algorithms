#!/bin/bash

echo "compiling the program ..."
mpicxx nearest-neighbours.cpp

echo && echo "create a random test case ..."
python create.py
if [ $? -ne 0 ]; then
    echo "failed to create test case"
    exit 1
fi

echo && echo "running the test case with a varying number of processes ... "
for num in {1..12}
do
    echo -n "$num " 
    bash ../test-file.sh "mpiexec -n ${num} ./a.out random.txt" "random-opt.txt"
done

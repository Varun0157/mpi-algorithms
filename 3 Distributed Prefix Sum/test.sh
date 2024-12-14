#!/bin/bash

echo "compiling the code..."
mpicxx prefix-sum.cpp

echo && echo "creating a random test case ... "
g++ create.cpp -o create
./create
if [ $? -ne 0 ]; then
    echo "failed to create test case"
    exit 1
fi


echo && echo "running the test case for a varying number of processes ... "
for num in {1..12}
do
    echo -n "$num " 
    bash ../test-file.sh "mpiexec -n ${num} ./a.out random.txt" "random-opt.txt"
done

#!/bin/bash

# compile the code
echo "compiling the code..."
mpicxx 3.cpp

# create a random test case
echo && echo "creating a random test case"
g++ create.cpp -o create
./create
if [ $? -ne 0 ]; then
    echo "failed to create test case"
    exit 1
fi


# for num in range 1 to 12, call ../test.sh 'mpiexec -n ${num} ./a.out inp1.txt' 'inp1-opt.txt'
echo && echo "running test cases for varying number of processes"
for num in {1..12}
do
    echo $num 
    bash ../test-file.sh "mpiexec -n ${num} ./a.out random.txt" "random-opt.txt"
done

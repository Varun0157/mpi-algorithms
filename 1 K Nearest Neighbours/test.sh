#!/bin/bash

echo "compiling the program"
# compile the code
mpicxx 1.cpp

# create a random test case
echo && echo "create a random test case"
python create.py
if [ $? -ne 0 ]; then
    echo "failed to create test case"
    exit 1
fi

echo && echo "test the program with a varying number of programs"
# for num in range 1 to 12, call ../test.sh 'mpiexec -n ${num} ./a.out inp1.txt' 'inp1-opt.txt'
for num in {1..12}
do
    echo $num 
    bash ../test-file.sh "mpiexec -n ${num} ./a.out random.txt" "random-opt.txt"
done

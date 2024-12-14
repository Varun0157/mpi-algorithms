clean:
    rm prefix-sum test-*.txt create-test

build:
    mpic++ prefix-sum.cpp -o prefix-sum

build-test:
    g++ create-test.cpp -o create-test
    ./create-test

test NUM: build build-test 
    bash ../test-file.sh "mpiexec -n {{NUM}} ./prefix-sum test-inp.txt" "test-opt.txt"

test-all: build build-test
    for i in $(seq 1 12); do echo -n "$i "; bash ../test-file.sh "mpiexec -n $i ./prefix-sum test-inp.txt" "test-opt.txt"; done

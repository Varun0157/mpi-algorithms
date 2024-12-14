clean:
    rm nearest-neighbours test-*.txt 

build:
    mpic++ nearest-neighbours.cpp -o nearest-neighbours

build-test:
    python create-test.py 

test NUM: build build-test 
    bash ../test-file.sh "mpiexec -n {{NUM}} ./nearest-neighbours test-inp.txt" "test-opt.txt"

test-all: build build-test
    for i in $(seq 1 12); do echo -n "$i "; bash ../test-file.sh "mpiexec -n $i ./nearest-neighbours test-inp.txt" "test-opt.txt"; done

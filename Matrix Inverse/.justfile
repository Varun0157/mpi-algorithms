clean:
    rm matrix-inverse test-*.txt 

build:
    mpic++ matrix-inverse.cpp -o matrix-inverse

build-test:
    python create-test.py 

test NUM: build build-test 
    bash ../test-file.sh "mpiexec -n {{NUM}} ./matrix-inverse test-inp.txt" "test-opt.txt"

test-all: build build-test
    for i in $(seq 1 12); do echo -n "$i "; bash ../test-file.sh "mpiexec -n $i ./matrix-inverse test-inp.txt" "test-opt.txt"; done

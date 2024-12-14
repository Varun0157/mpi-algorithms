# Parallel Algorithms using MPI

Distributed implementations using MPI of a multi-query K Nearest Neighbours, Prefix Sum calculation, and Matrix Inversion using Row Reduction. 

## Prerequisites
### required
- `mpic++`

There are a number of [reference guides](https://www.iitgoa.ac.in/hpcshiksha/HPC%20Shiksha%20-%20MPI%20Installation%20Guide.pdf) available. On linux, simply ensure `build-essential` and `mpich` are installed. 

### optional
- `just`

recipes in `justfiles` are used to simplify the build, clean and test calls. Install using the [programmer's manual](https://just.systems/man/en/installation.html). 

## Table of Contents
1. [Distributed K-Nearest Neighbours](#1-distributed-k-nearest-neighbours)
2. [Parallel Prefix Sum](#2-parallel-prefix-sum)
3. [Matrix Inversion](#3-matrix-inversion)

## 1. Distributed K-Nearest Neighbours

### Problem Statement
Given a set `P` of `N` points in a 2D plane and a set `Q` of `M` query points, find the `K` nearest neighbors from `P` for each query point in `Q`. The distance is calculated using the Euclidean distance formula.

### Methodology
Here, we simply distribute the queries to handle. This can be optimised further by distributing the computation of the nearest neighbours for each query as well - merging the intermediate nearest neighbours would be considerably faster. This will be explored later. 

### Input Format
- The first line contains three integers: `N`, `M`, and `K`.
- The next `N` lines contain two space-separated floating-point numbers representing the coordinates `(xi, yi)` of each point in `P`.
- The next `M` lines contain two space-separated floating-point numbers representing the coordinates `(xj, yj)` of each query point in `Q`.

### Output Format
- For each query point in `Q`, we write `K` lines of two space-separated integers representing the coordinates of the `K` nearest neighbors from `P` to a file. 

### Time Complexity
- **sequential running time**

$$
O(M \cdot N \cdot log(K))
$$
- **parallel running time** for *p processes*

$$
O(\frac{M}{p} \cdot N \cdot log(K))
$$

## 2. Parallel Prefix Sum

### Problem Statement
Compute the prefix sum of an array using parallel processing. The prefix sum array `p` for an array `a` is defined such that `p(i) = sum(a(1) to a(i))`.

### Methodology
- distribute `N/p` numbers to each of the `p` processes. 
- calculate the intermediate prefix sums within each process. 
- Send the last number in each intermediate prefix list back to the first process. Use this to calculate the value must be added by each of the processes to obtain the final list. 
- Send the values back to the processes, add this offset to each element in the list, and accumulate the global prefix sums. 

### Input Format
- The first line contains a single integer `N`, representing the size of the array.
- The second line contains `N` space-separated floating-point numbers representing the elements of the array `a`.

### Output Format
- A single line containing `N` space-separated floating-point numbers representing the prefix sum array `p`.

### Time Complexity 
- **sequential running time**

$$
O(N)
$$
- **parallel running time** for *p processes*

$$
O(\frac{N}{p})
$$

## 3. Matrix Inversion

### Problem Statement
Given a non-singular square matrix `A` of size `N × N`, compute its inverse using the row reduction method. The matrix is partitioned into rows and distributed across multiple processes.

### Methodology
Explore [this presentation](https://cse.buffalo.edu/faculty/miller/Courses/CSE633/thanigachalam-Spring-2014-CSE633.pdf) by Aravindhan Thanigachalam as part of the Parallel Algorithms course at University of Buffalo. 

### Input Format
- The first line contains a single integer `N`, representing the size of the matrix.
- The next `N` lines each contain `N` space-separated floating-point numbers representing the elements of the matrix `A`.

### Output Format
- Print `N` lines, each containing `N` floating-point numbers, representing the inverse matrix `A^(-1)`.

### Time Complexity
- **sequential running time**

$$
O(N^3)
$$
- **parallel running time** for *p processes*

$$
O(\frac{N^3}{p})
$$

## Execution
You can test each program by `cd`'ing into the required directory and running `just test-all`. See the available recipes to `clean`, `build`, or run with a specific number of processes. 

## Note
This was implemented as a part of *Assignment 3* of *Distributed Systems* in *IIIT-Hyderabad, Monsoon '24*. 

# MPI-Based Parallel Algorithms

This repository contains implementations of various parallel algorithms using MPI (Message Passing Interface). Each algorithm is designed to efficiently solve specific computational problems by leveraging parallel processing capabilities. Below are the details for each implemented algorithm.

## Table of Contents
1. [Distributed K-Nearest Neighbours](#1-distributed-k-nearest-neighbours)
2. [Julia Set Computation](#2-julia-set-computation)
3. [Parallel Prefix Sum](#3-parallel-prefix-sum)
4. [Matrix Inversion](#4-matrix-inversion)
5. [Parallel Matrix Chain Multiplication](#5-parallel-matrix-chain-multiplication)

## 1. Distributed K-Nearest Neighbours

### Problem Statement
Given a set `P` of `N` points in a 2D plane and a set `Q` of `M` query points, find the `K` nearest neighbors from `P` for each query point in `Q`. The distance is calculated using the Euclidean distance formula.

### Input Format
- The first line contains three integers: `N`, `M`, and `K`.
- The next `N` lines contain two space-separated floating-point numbers representing the coordinates `(xi, yi)` of each point in `P`.
- The next `M` lines contain two space-separated floating-point numbers representing the coordinates `(xj, yj)` of each query point in `Q`.

### Output Format
- For each query point in `Q`, print `K` lines of two space-separated integers representing the coordinates of the `K` nearest neighbors from `P`.

## 2. Julia Set Computation

### Problem Statement
Compute the Julia set for a given grid of complex numbers. The Julia set is defined by iterating over the function `z(n+1) = z(n)^2 + c` and checking how many iterations it takes for the magnitude of `z(n)` to exceed a given threshold `T`. 

### Input Format
- The first line contains three integers: `N`, `M`, and `K`, where `N × M` is the grid size and `K` is the maximum number of iterations.
- The second line contains two floating-point numbers representing the real and imaginary parts of the constant complex number `c`.

### Output Format
- Print an `N × M` grid where each element is `1` if the point is in the Julia set (does not exceed the threshold within `K` iterations), and `0` otherwise.

## 3. Parallel Prefix Sum

### Problem Statement
Compute the prefix sum of an array using parallel processing. The prefix sum array `p` for an array `a` is defined such that `p(i) = sum(a(1) to a(i))`.

### Input Format
- The first line contains a single integer `N`, representing the size of the array.
- The second line contains `N` space-separated floating-point numbers representing the elements of the array `a`.

### Output Format
- Print a single line containing `N` space-separated floating-point numbers representing the prefix sum array `p`.

## 4. Matrix Inversion

### Problem Statement
Given a non-singular square matrix `A` of size `N × N`, compute its inverse using the row reduction method. The matrix is partitioned into rows and distributed across multiple processes.

### Input Format
- The first line contains a single integer `N`, representing the size of the matrix.
- The next `N` lines each contain `N` space-separated floating-point numbers representing the elements of the matrix `A`.

### Output Format
- Print `N` lines, each containing `N` floating-point numbers, representing the inverse matrix `A^(-1)`.

## 5. Parallel Matrix Chain Multiplication

### Problem Statement
Given a sequence of matrices with specified dimensions, determine the optimal order of multiplication to minimize the total number of scalar multiplications needed. The problem leverages the associative property of matrix multiplication to reduce computational cost.

### Input Format
- The first line contains one integer `N`, representing the number of matrices.
- The second line contains `N + 1` integers representing the dimensions array `d`. The dimensions of the ith matrix are `d(i-1) × d(i)` for `1 ≤ i ≤ N`.

### Output Format
- Print one integer representing the minimum number of scalar multiplications needed to multiply the `N` matrices.

## Setup and Execution

### Prerequisites
- MPI Library (e.g., OpenMPI)
- C++ Compiler with MPI support (e.g., `mpic++`)

### Compilation
Each program can be compiled using the following command:
# Compilation Command

To compile the MPI program, use the following command:

```bash
mpic++ -o <output_executable_name> <source_code_file.cpp>
```



### Example
```bash
mpic++ -o julia_set julia_set.cpp
mpirun -np 4 ./julia_set input.txt

# Power, Inverse Power, Rayleigh Quotient, QR Hessenberg Decomposition, Arnoldi Krylov Subspace and Lanczos Krylov Subspace Eigenvalues

The power iteration, the inverse power iteration, and the Rayleigh quotient iteration algorithms compute the dominant eigenvalue of the linear system of equations AX = B. The QR Hessenberg decomposition, the Arnoldi Krylov subspace, and the Lanczos Krylov subspace algorithms compute all the eigenvalues and eigenvectors of the linear system of equations.

# Eigenvalue and Eigenvector Calculator

The C++ and python programs implement various algorithms to calculate eigenvalues and eigenvectors of matrices. The programs use the Eigen library for efficient matrix operations and provides multiple methods for eigenvalue computation.

# Features

1. Power Iteration Algorithm
2. Inverse Power Iteration Algorithm
3. Rayleigh Quotient Iteration Algorithm
4. QR Hessenberg Direct Algorithm
5. Arnoldi Krylov Subspace Algorithm
6. Lanczos Krylov Subspace Algorithm 

# Prerequisites

C++ compiler with C++11 support
Eigen library (version 3.3 or later)

# Installation

1. Clone the repository or download the source code.
2. Ensure you have the Eigen library installed and properly linked in your project.

# Usage

1. Compile the program using your C++ compiler.
2. Prepare input files:
  A.dat: Matrix Market format file containing the input matrix

  B.dat: Matrix Market format file containing the initial vector (for iterative methods)
  
4. Run the executable.

# Input File Format

The program uses the Matrix Market format for input files. The format is as follows:

%%MatrixMarket matrix coordinate real general

<num_rows> <num_cols> <num_entries>

<row> <col> <value>

<row> <col> <value>

# Output

The program will output:
The input matrix and vector

The calculated eigenvalues and eigenvectors

Execution time for each algorithm

Results are displayed in the console and also written to output files:

A_out.dat: Output matrix (if applicable)

X_out.dat: Output vector (eigenvector)

# Algorithms

1. Power Iteration: Calculates the dominant eigenvalue and corresponding eigenvector.
2. Inverse Power Iteration: Finds the smallest eigenvalue and its eigenvector.
3. Rayleigh Quotient Iteration: Uses the Rayleigh quotient to quickly converge to an eigenvalue.
4. QR Hessenberg: Direct method to calculate all eigenvalues and eigenvectors.
5. Arnoldi Krylov Subspace: Iterative method to calculate all eigenvalues and eigenvectors for a dense matrix.
6. Lanczos Krylov Subspace: Iterative method to calculate all eigenvalues and eigenvectors for a symmetric matrix.

# Functions

writeMatrixMarketMatrix: Writes a matrix to a file in Matrix Market format.

writeMatrixMarketVector: Writes a vector to a file in Matrix Market format.

readMatrixMarketMatrix: Reads a matrix from a file in Matrix Market format.

readMatrixMarketVector: Reads a vector from a file in Matrix Market format.

printExecutionTime: Displays the execution time of an algorithm.

printVector: Prints a vector to the console.

printMatrix: Prints a matrix to the console.

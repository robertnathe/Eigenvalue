import numpy as np
import scipy.io as spio
import time
import sys

def writeMatrixMarketMatrix(matrix, filename="A_out.dat"):
    with open(filename, 'w') as outfile:
        outfile.write("%%MatrixMarket matrix coordinate real general\n")
        rows, cols = matrix.shape
        non_zero = np.count_nonzero(matrix)
        outfile.write(f"{rows} {cols} {non_zero}\n")
        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] != 0.0:
                    outfile.write(f"{i+1} {j+1} {matrix[i, j]:.15f}\n")

def writeMatrixMarketVector(vector, filename="X_out.dat"):
    with open(filename, 'w') as outfile:
        outfile.write("%%MatrixMarket matrix coordinate real general\n")
        outfile.write(f"1 {vector.size} {vector.size}\n")
        for idx in range(vector.size):
            outfile.write(f"1 {idx+1} {vector[idx]:.15f}\n")

def readMatrixMarketMatrix(filename="A.dat"):
    try:
        matrix = spio.mmread(filename).toarray()
        return matrix
    except Exception as e:
        print(f"Failed to read matrix from {filename}: {e}")
        return None

def readMatrixMarketVector(filename="B.dat"):
    try:
        vector = spio.mmread(filename).toarray().flatten()
        return vector
    except Exception as e:
        print(f"Failed to read vector from {filename}: {e}")
        return None

def printExecutionTime(start, end):
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f}s\n")

def printVector(label, vector):
    print(f"{label}:\n[{', '.join(f'{v:.5f}' for v in vector)}]\n")

def printMatrix(label, matrix):
    print(f"{label}:")
    for row in matrix:
        print(' '.join(f'{val:10.5f}' for val in row))
    print()

def computeArnoldiEigenvalues():
    print("Arnoldi Krylov Subspace Algorithm:\n")
    A = readMatrixMarketMatrix("A.dat")
    if A is None:
        return -1
    v0 = readMatrixMarketVector("B.dat")
    if v0 is None:
        return -1
    if A.shape[0] != v0.size:
        print("Error: Matrix A and initial vector dimensions do not match.")
        return -1
    m = A.shape[0]
    V, H = Arnoldi(A, v0, m)
    Hm = H[:m, :m]
    eigvals = np.linalg.eigvals(Hm)
    printMatrix("Hessenberg Matrix H", Hm)
    print("Eigenvalues (Ritz values):")
    for eig in eigvals:
        print(f"{eig:.10f}")
    print()
    return 0

def Arnoldi(A, v0, m):
    n = A.shape[0]
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))
    V[:, 0] = v0 / np.linalg.norm(v0)
    for j in range(m):
        w = A @ V[:, j]
        h = V[:, :j+1].T @ w
        H[:j+1, j] = h
        w -= V[:, :j+1] @ h
        h_norm = np.linalg.norm(w)
        H[j+1, j] = h_norm
        if h_norm > 1e-12:
            V[:, j+1] = w / h_norm
    return V, H

def computeLanczosEigenvalues():
    print("Lanczos Algorithm:\n")
    A = readMatrixMarketMatrix("A.dat")
    if A is None:
        return -1
    v0 = readMatrixMarketVector("B.dat")
    if v0 is None:
        return -1
    if A.shape[0] != v0.size:
        print("Error: Matrix A and initial vector dimensions do not match.")
        return -1
    A_sym = (A + A.T) / 2
    printMatrix("Symmetric Matrix", A_sym)
    print("Lanczos Tridiagonalization Algorithm:\n")
    m = min(100, A.shape[0])
    V, T = Lanczos(A, v0, m)
    Tm = T[:m, :m]
    eigvals = np.linalg.eigvalsh(Tm)
    print("Tridiagonal Matrix T:")
    print(Tm)
    print("Eigenvalues:")
    print(eigvals)
    return 0

def Lanczos(A, v0, m):
    n = A.shape[0]
    if not np.allclose(A, A.T, atol=1e-8):
        print("Matrix must be symmetric for Lanczos")
        return None, None
    V = np.zeros((n, m+1))
    T = np.zeros((m+1, m))
    v_current = v0 / np.linalg.norm(v0)
    V[:, 0] = v_current
    v_prev = np.zeros(n)
    beta_prev = 0.0
    for j in range(m):
        w = A @ v_current
        alpha = v_current @ w
        T[j, j] = alpha
        w -= alpha * v_current
        if j > 0:
            w -= beta_prev * v_prev
        beta = np.linalg.norm(w)
        T[j+1, j] = beta
        if j < m-1:
            T[j, j+1] = beta
        if beta < 1e-12:
            print(f"Lanczos breakdown at step {j+1}")
            return V, T
        if j < m-1:
            v_next = w / beta
            V[:, j+1] = v_next
            v_prev, v_current = v_current, v_next
            beta_prev = beta
    return V, T

def computeDominantEigenvalue():
    print("Power iteration algorithm:\n")
    inputMatrix = readMatrixMarketMatrix()
    if inputMatrix is None:
        return -1
    inputVector = readMatrixMarketVector()
    if inputVector is None:
        return -1
    if inputMatrix.shape[0] != inputVector.size:
        print("Error: Matrix and vector dimensions do not match.")
        return -1
    resultVector = np.zeros(inputMatrix.shape[0])
    printMatrix("Input Matrix", inputMatrix)
    printVector("Input Vector", inputVector)
    startTime = time.time()
    dominantEigenvalue, resultVector = calculateDominantEigenvalue(inputMatrix, inputVector, resultVector)
    endTime = time.time()
    if dominantEigenvalue is None:
        return -1
    print(f"Dominant Eigenvalue: {dominantEigenvalue:.10f}")
    printVector("Result Vector", resultVector)
    writeMatrixMarketVector(resultVector)
    printExecutionTime(startTime, endTime)
    return 0

def inversePowerEigenvalue():
    print("Inverse power iteration algorithm:\n")
    inputMatrix = readMatrixMarketMatrix()
    if inputMatrix is None:
        return -1
    inputVector = readMatrixMarketVector()
    if inputVector is None:
        return -1
    if inputMatrix.shape[0] != inputVector.size:
        print("Error: Matrix and vector dimensions do not match.")
        return -1
    resultVector = np.zeros(inputMatrix.shape[0])
    printMatrix("Input Matrix", inputMatrix)
    printVector("Input Vector", inputVector)
    startTime = time.time()
    pseudoInverseMatrix = np.linalg.pinv(inputMatrix)
    inverseEigenvalue, resultVector = calculateDominantEigenvalue(pseudoInverseMatrix, inputVector, resultVector)
    endTime = time.time()
    if inverseEigenvalue is None or inverseEigenvalue == 0.0:
        print("Error: Eigenvalue is zero. Cannot compute inverse.")
        return -1
    print(f"Inverse Dominant Eigenvalue: {1.0 / inverseEigenvalue:.10f}")
    printVector("Result Vector", resultVector)
    writeMatrixMarketVector(resultVector)
    printExecutionTime(startTime, endTime)
    return 0

def computeRayleighEigenvalue():
    print("Rayleigh iteration algorithm:\n")
    inputMatrix = readMatrixMarketMatrix()
    if inputMatrix is None:
        return -1
    initialVector = readMatrixMarketVector()
    if initialVector is None:
        return -1
    if inputMatrix.shape[0] != initialVector.size:
        print("Error: Input matrix and vector dimensions do not match.")
        return -1
    printMatrix("Input Matrix", inputMatrix)
    printVector("Initial Vector", initialVector)
    startTime = time.time()
    updatedVector = rayleigh_quotient_iteration(inputMatrix, initialVector, 1000, 1e-8)
    dominantRayleighEigenvalue = rayleigh_quotient(inputMatrix, updatedVector)
    endTime = time.time()
    print(f"Dominant Rayleigh Eigenvalue: {dominantRayleighEigenvalue:.10f}")
    printVector("Updated Vector", updatedVector)
    writeMatrixMarketVector(updatedVector)
    printExecutionTime(startTime, endTime)
    return 0

def qrEigenvalue():
    print("QR Hessenberg direct algorithm:\n")
    inputMatrix = readMatrixMarketMatrix()
    if inputMatrix is None:
        return -1
    printMatrix("Input Matrix", inputMatrix)
    startTime = time.time()
    eigvals, eigvecs = np.linalg.eig(inputMatrix)
    endTime = time.time()
    print("Eigenvalues:")
    for eig in eigvals:
        print(f"{eig:.10f}")
    print("\nEigenvectors:")
    for idx in range(eigvecs.shape[1]):
        print(f"Eigenvector {idx+1}:")
        for val in eigvecs[:, idx]:
            print(f"{val:.10f}")
        print()
    printExecutionTime(startTime, endTime)
    return 0

def rayleigh_quotient(A, x):
    numerator = x.T @ A @ x
    denominator = x.T @ x
    return numerator / denominator

def rayleigh_quotient_iteration(A, x, max_iter=100, tolerance=1e-6):
    prev_eigenvalue = 0.0
    for i in range(max_iter):
        eigenvalue = rayleigh_quotient(A, x)
        if abs(eigenvalue - prev_eigenvalue) < tolerance:
            break
        prev_eigenvalue = eigenvalue
        I = np.eye(A.shape[0])
        w = np.linalg.solve(A - eigenvalue * I, x)
        x = w / np.linalg.norm(w)
    return x

def calculateDominantEigenvalue(matrix, vector, resultVector, tolerance=1e-5, maxIterations=1000):
    return eigenvaluePowerMethod(matrix, vector, tolerance, maxIterations)

def eigenvaluePowerMethod(matrix, initialVector, tolerance, maxIterations):
    resultVector = initialVector.copy()
    previousEigenvalue = 0.0
    for iteration in range(maxIterations):
        matrixVectorMultiply(matrix, resultVector, resultVector)
        eigenvalue = calculateVectorNorm(resultVector)
        if eigenvalue == 0.0:
            return None, None
        scaleVector(resultVector, eigenvalue)
        if iteration > 0 and abs((eigenvalue - previousEigenvalue) / eigenvalue) < tolerance:
            return eigenvalue, resultVector
        previousEigenvalue = eigenvalue
    return None, None

def matrixVectorMultiply(matrix, vector, result):
    result[:] = matrix @ vector

def calculateVectorNorm(vector):
    return np.linalg.norm(vector)

def calculateInnerProduct(firstVector, secondVector):
    return np.dot(firstVector, secondVector)

def scaleVector(vector, scalar):
    vector /= scalar

if __name__ == "__main__":
    computeDominantEigenvalue()
    inversePowerEigenvalue()
    computeRayleighEigenvalue()
    qrEigenvalue()
    computeArnoldiEigenvalues()
    computeLanczosEigenvalues()

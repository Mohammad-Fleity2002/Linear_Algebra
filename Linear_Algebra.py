import numpy as np
import sympy as sp


A = np.random.randint(0, 10, size=(4, 4))
B = np.random.randint(0, 10, size=(4, 4))
C = np.array([[2, 3, 4], [3, 1, 5], [4, 5, 3]])
D = np.array([[0, 1, 2], [-1, 0, 3], [-2, -3, 0]])
E = np.array([[1, 1, 2, 4], [0, 1, 3, 1], [0, 0, 1, 2]])
F = np.array([[1, -2, 3, 9], [-1, 3, 0, -4], [2, -5, 5, 17]])
G = sp.Matrix([[1, -2, 3, 9], [-1, 3, 0, -4], [2, -5, 5, 17]])
K = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
L = np.array([[1, 0, 0, 3],
              [0, 1, 0, 3],
              [0, 0, 1, 3]])


def Matrix_Add(A, B):
    if A.shape != B.shape:
        print("Addition impossible, Not same shape")
        return None
    return np.add(A, B)


def Matrix_Equal(A, B):
    return np.array_equal(A, B)


def Matrix_Multiplication(A, B):
    if A.shape[1] != B.shape[0]:
        print("Dot Product impossible, Shape errors")
        return None

    return np.dot(A, B)


def Matrix_isSquare(A):
    return A.shape[0] == A.shape[1]


def Matrix_isUpperTriangle(A):
    return np.allclose(A, np.triu(A))


def Matrix_isLowerTriangle(A):
    return np.allclose(A, np.tril(A))


def Matrix_isDiagonal(A):
    return np.allclose(A, np.diag(np.diag(A)))


def Identity_Matrix_For(A):
    if Matrix_isSquare == False:
        print("Matrix isn't Square, No Identity Matrix")
        return None
    return np.identity(A.shape[0])


def Matrix_Transpose(A):
    return np.transpose(A)


def Matrix_isSymetric(A):
    return Matrix_Equal(A, Matrix_Transpose(A))


def Matrix_isSkewSymetric(A):
    return Matrix_Equal(A, (-1*Matrix_Transpose(A)))


def Matrix_isREF(A):
    rows, cols = A.shape
    last_pivot = -1

    for i in range(rows):
        # Find the first nonzero entry in the row
        row = A[i]
        pivot_indices = np.where(row != 0)[0]

        # Skip zero rows
        if pivot_indices.size == 0:
            continue

        # Check if pivot is to the right
        pivot = pivot_indices[0]
        if pivot <= last_pivot:
            return False

        # Check if below pivot elements are zero
        if not np.all(A[i+1:, pivot] == 0):
            return False

        last_pivot = pivot

    return True


def Matrix_isRREF(A):
    if not Matrix_isREF(A):  # Must first satisfy REF
        return False

    rows, cols = A.shape

    for i in range(rows):
        # Find the pivot in the current row
        row = A[i]
        pivot_indices = np.where(row == 1)[0]

        if pivot_indices.size == 0:  # Skip zero rows
            continue

        pivot = pivot_indices[0]

        # Check if the pivot is the only nonzero element in its column
        if not np.allclose(A[:, pivot], np.eye(1, rows, i).T.flatten()):
            return False

    return True


def Matrix_ToREF(A):

    A = A.astype(float)
    row, column = A.shape
    # this for loop organize the matrix considering the max values for each pivot is on top of the column
    # 1 4 8                     3 7 9                                         3 7 9
    # 2 6 0 => first iteration: 2 6 0 => 2nd iteration 2nd column from row 1: 2 6 0
    # 3 7 9                     1 4 8                                         1 4 8
    # since it's 3x3 matrix there are no more iterations.

    for i in range(min(row, column)):  # if n < m or m < n i always should be the min between them
        # find the pivot
        max_row = i+np.argmax(abs(A[i:, i]))
        if A[max_row, i] == 0:
            continue  # The pivot is Zero then continue

        # Swap rows
        A[[i, max_row]] = A[[max_row, i]]

        # Normalize the pivot to 1
        A[i] = A[i]/A[i, i]

        # Eliminate the below value in the column
        for j in range(i+1, row):
            A[j] -= A[j, i]*A[i]
    return A


def Matrix_ToRREF(A):
    A = sp.Matrix(A)
    ref_Matrix, pivots = A.rref()
    return ref_Matrix


# print("A: ", A)
# print("B: ", B)
# print("A+B: \n", Matrix_Add(A, B))
# print("A==B: \n", Matrix_Equal(A, B))
# print("A*B: \n", Matrix_Multiplication(A, B))
# print("A is square? \n", Matrix_isSquare(A))
# print("Identity matrix of A: \n", Identity_Matrix_For(A))
# print("Upper Triangel A? \n", Matrix_isUpperTriangle(A))
# print("Lower Triangel A? \n", Matrix_isLowerTriangle(A))
# print("Diagonal Matrix A? \n", Matrix_isDiagonal(A))
# print("Diagonal Matrix IA? \n", Matrix_isDiagonal(Identity_Matrix_For(A)))
# print("Upper Triangel IA? \n", Matrix_isUpperTriangle(Identity_Matrix_For(A)))
# print("Lower Triangel IA? \n", Matrix_isLowerTriangle(Identity_Matrix_For(A)))
# print("Transpose of A: \n", Matrix_Transpose(A))
# print("C: \n", C)
# print("Is C Symetric? \n", Matrix_isSymetric(C))
# print("B: \n", B)
# print("Is B Skew-Symetric? \n", Matrix_isSkewSymetric(B))
# print("D: \n", D)
# print("Is D Skew-Symetric? \n", Matrix_isSkewSymetric(D))
print("F: \n", F)
print("Echlon Form of F: \n", Matrix_ToREF(F))
print("G: \n", G)
print("Reduced Echlon Form of G: \n", Matrix_ToRREF(G))
print("E: \n", E)
print("Is E REF? \n", Matrix_isREF(E))
print("Is E RREF? \n", Matrix_isRREF(E))
print("k: \n", K)
print("Is K RREF?\n", Matrix_isRREF(K))
print("Is K REF?\n", Matrix_isREF(K))
print("L: \n", L)
print("Is L RREF?\n", Matrix_isRREF(L))
print("Is L REF?\n", Matrix_isREF(L))

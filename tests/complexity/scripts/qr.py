# pylint: disable=missing-function-docstring, missing-module-docstring/
# ===================================================
def qr(A: 'double[:,:]',
       L: 'double[:,:]'):

    from numpy import shape
    from numpy import sqrt

    n = shape(L)[0]

    for i in range(n):
        s = 0.
        for k in range(i):
            s = s + L[i,k]**2
        L[i,i] = sqrt(A[i,i] - s)

        for j in range(i):
            s = 0.
            for k in range(j):
                s = s + L[i,k] * L[j,k]
            L[i,j] = A[i,j] - s
            L[i,j] = L[i,j] / L[j,j]

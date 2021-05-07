# pylint: disable=missing-function-docstring, missing-module-docstring/
# ===================================================
def f1(L: 'double[:,:]',
       b: 'double[:,:]',
       x: 'double[:,:]'):

    from numpy import shape
    n = shape(L)[0]
    for i in range(n):
        for j in range(n):
            s = 0.
            for k in range(j):
                s = s + L[j,k] * x[i,k]
            x[i,j] = b[i,j] - s
            x[i,j] = x[i,j] / L[j,j]

# ===================================================
def f2(L: 'double[:,:]',
       b: 'double[:,:]',
       x: 'double[:,:]'):

    from numpy import shape
    n = shape(L)[0]
    for i in range(n):
        for j in range(n):
            s = 0.
            for k in range(j):
                s = s + L[j,k] * x[i,k] / L[j,j]
            x[i,j] = b[i,j] / L[j,j] - s

# ===================================================
def f3(A: 'double[:,:]',
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

# ===================================================
from numpy import zeros

n = 5

x = zeros((n,n))
b = zeros((n,n))
L = zeros((n,n))

f1(L,b,x)
f2(L,b,x)
f3(L,b)

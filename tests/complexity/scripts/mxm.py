# pylint: disable=missing-function-docstring, missing-module-docstring/
# ===================================================
def mxm(x: 'double[:,:]',
        y: 'double[:,:]',
        z: 'double[:,:]'):

    from numpy import shape

    n = shape(x)[0]

    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                z[i,j] = z[i,j] + x[i,k]*y[k,j]

# ===================================================
def mxm_block(x: 'double[:,:]',
              y: 'double[:,:]',
              z: 'double[:,:]',
              b: 'int'):

    from numpy import zeros
    from numpy import shape

    n = shape(x)[0]

    r = zeros((b,b))
    u = zeros((b,b))
    v = zeros((b,b))

    for i in range(0, n, b):
        for j in range(0, n, b):
            for k1 in range(0, b):
                for k2 in range(0, b):
                    r[k1,k2] = z[i+k1,j+k2]

            for k in range(0, n, b):
                for k1 in range(0, b):
                    for k2 in range(0, b):
                        u[k1,k2] = x[i+k1,k+k2]
                        v[k1,k2] = y[k+k1,j+k2]

                for ii in range(0, b):
                    for jj in range(0, b):
                        for kk in range(0, b):
                            r[ii,jj] = r[ii,jj] + u[ii,kk]*v[kk,jj]

            for k1 in range(0, b):
                for k2 in range(0, b):
                    z[i+k1,j+k2] = r[k1,k2]

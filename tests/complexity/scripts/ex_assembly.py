# pylint: disable=missing-function-docstring, missing-module-docstring/
#==============================================================================
def assemble_matrix_ex01(ne1: 'int', ne2: 'int',
                        p1:  'int', p2:  'int',
                        spans_1:          'int[:]', spans_2:          'int[:]',
                        basis_1: 'double[:,:,:,:]', basis_2: 'double[:,:,:,:]',
                        weights_1:   'double[:,:]', weights_2:   'double[:,:]',
                        points_1:    'double[:,:]', points_2:    'double[:,:]',
                        vector_u:    'double[:,:]',
                        matrix: 'double[:,:,:,:]'):

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    from numpy import zeros

    lcoeffs_u = zeros((p1+1,p2+1))
    lvalues_u = zeros((k1, k2))

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u[ : , : ] = 0.0
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]
                    for g1 in range(0, k1):
                        b1 = basis_1[ie1,il_1,0,g1]
                        for g2 in range(0, k2):
                            b2 = basis_2[ie2,il_2,0,g2]

                            lvalues_u[g1,g2] += coeff_u*b1*b2

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2+1):
                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_0 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_y = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    u = lvalues_u[g1,g2]

                                    v += (1. + u**2)*(bi_x * bj_x + bi_y * bj_y) * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...

# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np

from pyccel.decorators import types, stack_array, allow_negative_index

a_1d   = np.array([1 << i for i in range(21)], dtype=np.int)
a_2d_f = np.array([[1 << j for j in range(21)] for i in range(21)], dtype=np.int, order='F')
a_2d_c = np.array([[1 << j for j in range(21)] for i in range(21)], dtype=np.int)

#==============================================================================
# 1D ARRAYS OF INT-32
#==============================================================================

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_add( x, a ):
    x[:] += a

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_sub( x, a ):
    x[:] -= a

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_mul( x, a ):
    x[:] *= a

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_div( x, a ):
    x[:] = x / a

@types( 'int32[:]', 'int32' )
def array_int32_1d_scalar_idiv( x, a ):
    x[:] = x // a

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_add( x, y ):
    x[:] += y

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_sub( x, y ):
    x[:] -= y

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_mul( x, y ):
    x[:] *= y

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_idiv( x, y ):
    x[:] = x // y

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_add_augassign( x, y ):
    x += y

@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_sub_augassign( x, y ):
    x -= y

def array_int_1d_initialization_1():
    import numpy as np
    a = np.array([1, 2, 4, 8, 16])
    b = np.array(a)
    return np.sum(b), b[0], b[-1]

def array_int_1d_initialization_2():
    import numpy as np
    a = [1, 2, 4, 8, 16]
    b = np.array(a)
    return np.sum(b), b[0], b[-1]

def array_int_1d_initialization_3():
    import numpy as np
    a = (1, 2, 4, 8, 16)
    b = np.array(a)
    return np.sum(b), b[0], b[-1]

#==============================================================================
# 2D ARRAYS OF INT-32 WITH C ORDERING
#==============================================================================

@types( 'int32[:,:]', 'int32' )
def array_int32_2d_C_scalar_add( x, a ):
    x[:,:] += a

@types( 'int32[:,:]', 'int32' )
def array_int32_2d_C_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'int32[:,:]', 'int32' )
def array_int32_2d_C_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'int32[:,:]', 'int32' )
def array_int32_2d_C_scalar_idiv( x, a ):
    x[:,:] = x // a

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_add( x, y ):
    x[:,:] += y

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_sub( x, y ):
    x[:,:] -= y

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_mul( x, y ):
    x[:,:] *= y

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_idiv( x, y ):
    x[:,:] = x // y

#==============================================================================
# 2D ARRAYS OF INT-32 WITH F ORDERING
#==============================================================================

@types( 'int32[:,:](order=F)', 'int32' )
def array_int32_2d_F_scalar_add( x, a ):
    x[:,:] += a

@types( 'int32[:,:](order=F)', 'int32' )
def array_int32_2d_F_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'int32[:,:](order=F)', 'int32' )
def array_int32_2d_F_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'int32[:,:](order=F)', 'int32' )
def array_int32_2d_F_scalar_idiv( x, a ):
    x[:,:] = x // a

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_add( x, y ):
    x[:,:] += y

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_sub( x, y ):
    x[:,:] -= y

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_mul( x, y ):
    x[:,:] *= y

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_idiv( x, y ):
    x[:,:] = x // y


#==============================================================================
# 1D ARRAYS OF INT-64
#==============================================================================

@types( 'int[:]', 'int' )
def array_int_1d_scalar_add( x, a ):
    x[:] += a

@types( 'int[:]', 'int' )
def array_int_1d_scalar_sub( x, a ):
    x[:] -= a

@types( 'int[:]', 'int' )
def array_int_1d_scalar_mul( x, a ):
    x[:] *= a

@types( 'int[:]', 'int' )
def array_int_1d_scalar_idiv( x, a ):
    x[:] = x // a

@types( 'int[:]', 'int[:]' )
def array_int_1d_add( x, y ):
    x[:] += y

@types( 'int[:]', 'int[:]' )
def array_int_1d_sub( x, y ):
    x[:] -= y

@types( 'int[:]', 'int[:]' )
def array_int_1d_mul( x, y ):
    x[:] *= y

@types( 'int[:]', 'int[:]' )
def array_int_1d_idiv( x, y ):
    x[:] = x // y

#==============================================================================
# 2D ARRAYS OF INT-64 WITH C ORDERING
#==============================================================================

@types( 'int[:,:]', 'int' )
def array_int_2d_C_scalar_add( x, a ):
    x[:,:] += a

@types( 'int[:,:]', 'int' )
def array_int_2d_C_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'int[:,:]', 'int' )
def array_int_2d_C_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'int[:,:]', 'int' )
def array_int_2d_C_scalar_idiv( x, a ):
    x[:,:] = x // a

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_add( x, y ):
    x[:,:] += y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_sub( x, y ):
    x[:,:] -= y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_mul( x, y ):
    x[:,:] *= y

@types( 'int[:,:]', 'int[:,:]' )
def array_int_2d_C_idiv( x, y ):
    x[:,:] = x // y

@types('int[:,:]')
def array_int_2d_C_initialization(a):
    from numpy import array
    tmp = array([[1, 2, 3], [4, 5, 6]])
    a[:,:] = tmp[:,:]

#==============================================================================
# 2D ARRAYS OF INT-64 WITH F ORDERING
#==============================================================================

@types( 'int[:,:](order=F)', 'int' )
def array_int_2d_F_scalar_add( x, a ):
    x[:,:] += a

@types( 'int[:,:](order=F)', 'int' )
def array_int_2d_F_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'int[:,:](order=F)', 'int' )
def array_int_2d_F_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'int[:,:](order=F)', 'int' )
def array_int_2d_F_scalar_idiv( x, a ):
    x[:,:] = x // a

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_add( x, y ):
    x[:,:] += y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_sub( x, y ):
    x[:,:] -= y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_mul( x, y ):
    x[:,:] *= y

@types( 'int[:,:](order=F)', 'int[:,:](order=F)' )
def array_int_2d_F_idiv( x, y ):
    x[:,:] = x // y

@types('int[:,:](order=F)')
def array_int_2d_F_initialization(a):
    from numpy import array
    tmp = array([[1, 2, 3], [4, 5, 6]], dtype='int', order='F')
    a[:,:] = tmp[:,:]


#==============================================================================
# 1D ARRAYS OF REAL
#==============================================================================

@types( 'real[:]', 'real' )
def array_real_1d_scalar_add( x, a ):
    x[:] += a

@types( 'real[:]', 'real' )
def array_real_1d_scalar_sub( x, a ):
    x[:] -= a

@types( 'real[:]', 'real' )
def array_real_1d_scalar_mul( x, a ):
    x[:] *= a

@types( 'real[:]', 'real' )
def array_real_1d_scalar_div( x, a ):
    x[:] /= a

@types( 'real[:]', 'real' )
def array_real_1d_scalar_idiv( x, a ):
    x[:] = x // a

@types( 'real[:]', 'real[:]' )
def array_real_1d_add( x, y ):
    x[:] += y

@types( 'real[:]', 'real[:]' )
def array_real_1d_sub( x, y ):
    x[:] -= y

@types( 'real[:]', 'real[:]' )
def array_real_1d_mul( x, y ):
    x[:] *= y

@types( 'real[:]', 'real[:]' )
def array_real_1d_div( x, y ):
    x[:] /= y

@types( 'real[:]', 'real[:]' )
def array_real_1d_idiv( x, y ):
    x[:] = x // y

#==============================================================================
# 2D ARRAYS OF REAL WITH C ORDERING
#==============================================================================

@types( 'real[:,:]', 'real' )
def array_real_2d_C_scalar_add( x, a ):
    x[:,:] += a

@types( 'real[:,:]', 'real' )
def array_real_2d_C_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'real[:,:]', 'real' )
def array_real_2d_C_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'real[:,:]', 'real' )
def array_real_2d_C_scalar_div( x, a ):
    x[:,:] /= a

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_add( x, y ):
    x[:,:] += y

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_sub( x, y ):
    x[:,:] -= y

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_mul( x, y ):
    x[:,:] *= y

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_div( x, y ):
    x[:,:] /= y

@types('real[:,:]')
def array_real_2d_C_array_initialization(a):
    from numpy import array
    tmp = array([[1, 2, 3], [4, 5, 6]], dtype='float')
    a[:,:] = tmp[:,:]

@types('real[:,:]','real[:,:]', 'real[:,:,:]')
def array_real_3d_C_array_initialization_1(x, y, a):
    from numpy import array
    tmp      = array([x, y], dtype='float')
    a[:,:,:] = tmp[:,:,:]

@types('real[:,:,:]')
def array_real_3d_C_array_initialization_2(a):
    from numpy import array
    x = array([[[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
              [[12., 13., 14., 15.], [16., 17., 18., 19.], [20., 21., 22., 23.]]], order='C')
    a[:,:,:] = x[:,:,:]

@types('real[:,:,:]','real[:,:,:]', 'real[:,:,:,:]')
def array_real_4d_C_array_initialization(x, y, a):
    from numpy import array
    tmp      = array([x, y], dtype='float')
    a[:,:,:,:] = tmp[:,:,:,:]


#==============================================================================
# 2D ARRAYS OF REAL WITH F ORDERING
#==============================================================================

@types( 'real[:,:](order=F)', 'real' )
def array_real_2d_F_scalar_add( x, a ):
    x[:,:] += a

@types( 'real[:,:](order=F)', 'real' )
def array_real_2d_F_scalar_sub( x, a ):
    x[:,:] -= a

@types( 'real[:,:](order=F)', 'real' )
def array_real_2d_F_scalar_mul( x, a ):
    x[:,:] *= a

@types( 'real[:,:](order=F)', 'real' )
def array_real_2d_F_scalar_div( x, a ):
    x[:,:] /= a

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_add( x, y ):
    x[:,:] += y

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_sub( x, y ):
    x[:,:] -= y

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_mul( x, y ):
    x[:,:] *= y

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_div( x, y ):
    x[:,:] /= y

@types('real[:,:](order=F)')
def array_real_2d_F_array_initialization(a):
    from numpy import array
    tmp = array([[1, 2, 3], [4, 5, 6]], dtype='float', order='F')
    a[:,:] = tmp[:,:]

@types('real[:,:](order=F)','real[:,:](order=F)', 'real[:,:,:](order=F)')
def array_real_3d_F_array_initialization_1(x, y, a):
    from numpy import array
    tmp      = array([x, y], dtype='float', order='F')
    a[:,:,:] = tmp[:,:,:]

@types('real[:,:,:](order=F)')
def array_real_3d_F_array_initialization_2(a):
    from numpy import array
    x = array([[[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
                 [[12., 13., 14., 15.], [16., 17., 18., 19.], [20., 21., 22., 23.]]], order='F')
    a[:,:,:] = x[:,:,:]

@types('real[:,:,:](order=F)','real[:,:,:](order=F)', 'real[:,:,:,:](order=F)')
def array_real_4d_F_array_initialization(x, y, a):
    from numpy import array
    tmp      = array([x, y], dtype='float', order='F')
    a[:,:,:,:] = tmp[:,:,:,:]

@types('real[:,:](order=F)', 'real[:,:,:,:](order=F)')
def array_real_4d_F_array_initialization_mixed_ordering(x, a):
    import numpy as np
    tmp      = np.array(((((0., 1.), (2., 3.)),
                          ((4., 5.), (6., 7.)),
                          ((8., 9.), (10., 11.))),
                          (((12., 13.), (14., 15.)),
                          x,
                          ((20., 21.), (22., 23.)))),
                          dtype='float', order='F')

    a[:,:,:,:] = tmp[:,:,:,:]

#==============================================================================
# COMPLEX EXPRESSIONS IN 3D : TEST CONSTANT AND UNKNOWN SHAPES
#==============================================================================


@types( 'int32[:]', 'int32[:]' )
def array_int32_1d_complex_3d_expr( x, y ):
    from numpy import full, int32
    z = full(3,5, dtype=int32)
    x[:] = (x // y) * x + z

@types( 'int32[:,:]', 'int32[:,:]' )
def array_int32_2d_C_complex_3d_expr( x, y ):
    from numpy import full, int32
    z = full((2,3),5, dtype=int32)
    x[:] = (x // y) * x + z

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)' )
def array_int32_2d_F_complex_3d_expr( x, y ):
    from numpy import full, int32
    z = full((2,3),5,order='F', dtype=int32)
    x[:] = (x // y) * x + z

@types( 'real[:]', 'real[:]' )
def array_real_1d_complex_3d_expr( x, y ):
    from numpy import full
    z = full(3,5)
    x[:] = (x // y) * x + z

@types( 'real[:,:]', 'real[:,:]' )
def array_real_2d_C_complex_3d_expr( x, y ):
    from numpy import full
    z = full((2,3),5)
    x[:] = (x // y) * x + z

@types( 'real[:,:](order=F)', 'real[:,:](order=F)' )
def array_real_2d_F_complex_3d_expr( x, y ):
    from numpy import full
    z = full((2,3),5,order='F')
    x[:] = (x // y) * x + z

@types( 'int32[:]', 'int32[:]', 'bool[:]' )
def array_int32_in_bool_out_1d_complex_3d_expr( x, y, ri ):
    from numpy import full, int32, empty
    z = full(3,5, dtype=int32)
    ri[:] = (x // y) * x > z

@types( 'int32[:,:]', 'int32[:,:]', 'bool[:,:]' )
def array_int32_in_bool_out_2d_C_complex_3d_expr( x, y, ri ):
    from numpy import full, int32
    z = full((2,3),5, dtype=int32)
    ri[:] = (x // y) * x > z

@types( 'int32[:,:](order=F)', 'int32[:,:](order=F)', 'bool[:,:](order=F)' )
def array_int32_in_bool_out_2d_F_complex_3d_expr( x, y, ri ):
    from numpy import full, int32
    z = full((2,3),5,order='F', dtype=int32)
    ri[:] = (x // y) * x > z

#==============================================================================
# 1D STACK ARRAYS OF REAL
#==============================================================================

@stack_array('a')
def array_real_1d_sum_stack_array():
    from numpy import zeros
    a = zeros(10)
    s = 0.
    for i in range(10):
        s += a[i]
    return s

@stack_array('a')
def array_real_1d_div_stack_array():
    from numpy import ones
    a = ones(10)
    s = 0.
    for i in range(10):
        s += 1.0 / a[i]
    return s

@stack_array('a')
@stack_array('b')
def multiple_stack_array_1():
    from numpy import ones, array
    a = ones(5)
    b = array([1, 3, 5, 7, 9])
    s = 0.0
    for i in range(5):
        s += a[i] / b[i]
    return s

@stack_array('a')
@stack_array('b', 'c')
def multiple_stack_array_2():
    from numpy import ones, array
    a = ones(5)
    b = array([2, 4, 6, 8, 10])
    c = array([1, 3, 5, 7, 9])
    s = 0.0
    for i in range(5):
        s = s + b[i] - a[i] / c[i]
    return s

#==============================================================================
# 2D STACK ARRAYS OF REAL
#==============================================================================

@stack_array('a')
def array_real_2d_sum_stack_array():
    from numpy import zeros
    a = zeros((10, 10))
    s = 0.
    for i in range(10):
        for j in range(10):
            s += a[i][j]
    return s

@stack_array('a')
def array_real_2d_div_stack_array():
    from numpy import full
    a = full((10, 10), 2)
    s = 1.
    for i in range(10):
        for j in range(10):
            s /= a[i][j]
    return s

@stack_array('a')
@stack_array('b')
def multiple_2d_stack_array_1():
    from numpy import ones, array
    a = ones((2, 5))
    b = array([[1, 3, 5, 7, 9], [11, 13, 17, 19, 23]])
    s = 0.0
    j = 0
    for i in range(2):
        for j in range(5):
            s += a[i][j] / b[i][j]
    return s

@stack_array('a')
@stack_array('b', 'c')
def multiple_2d_stack_array_2():
    from numpy import ones, array
    a = ones(5)
    b = array([[2, 4, 6, 8, 10], [1, 3, 5, 7, 9]])
    c = array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    s = 0.0
    for i in range(2):
        for j in range(5):
            s = s + b[i][j] - a[j] / c[i][j]
    return s

#==============================================================================
# TEST: Product and matrix multiplication
#==============================================================================

@types('real[:], real[:]')
def array_real_1d_1d_prod(x, out):
    from numpy import prod
    out[:] = prod(x)

@types('real[:,:], real[:], real[:]')
def array_real_2d_1d_matmul(A, x, out):
    from numpy import matmul
    out[:] = matmul(A, x)

@types('real[:,:](order=F), real[:], real[:]')
def array_real_2d_1d_matmul_order_F(A, x, out):
    from numpy import matmul
    out[:] = matmul(A, x)

@types('real[:], real[:,:], real[:]')
def array_real_1d_2d_matmul(x, A, out):
    from numpy import matmul
    out[:] = matmul(x, A)

@types('real[:,:], real[:,:], real[:,:]')
def array_real_2d_2d_matmul(A, B, out):
    from numpy import matmul
    out[:,:] = matmul(A, B)

@types('real[:,:](order=F), real[:,:](order=F), real[:,:](order=F)')
def array_real_2d_2d_matmul_F_F(A, B, out):
    from numpy import matmul
    out[:,:] = matmul(A, B)

# Mixed order, not supported currently, see #244
@types('real[:,:], real[:,:](order=F), real[:,:]')
def array_real_2d_2d_matmul_mixorder(A, B, out):
    from numpy import matmul
    out[:,:] = matmul(A, B)

@types('real[:], real[:], real[:]')
def array_real_loopdiff(x, y, out):
    dxy = x - y
    for k in range(len(x)):
        out[k] = dxy[k]

#==============================================================================
# KEYWORD ARGUMENTS
#==============================================================================

def array_kwargs_full():
    """ full(shape, fill_value, dtype=None, order='C')
    """

    from numpy import sum as np_sum
    from numpy import full

    n = 3

    a = full((n, n-1), 0.5, 'float', 'C')
    b = full((n+1, 2*n), 2.0, order='F')
    c = full((1, n), 3)
    d = full(2+n, order='F', fill_value=5)
    e = full(dtype=int, fill_value=1.0, shape=2*n)

    return np_sum(a) + np_sum(b) + np_sum(c) + np_sum(d) + np_sum(e)

def array_kwargs_ones():
    """ ones(shape, dtype=float, order='C')
    """

    from numpy import sum as np_sum
    from numpy import ones

    n = 4

    a = ones((n, n-1), 'float', 'C')
    b = ones((n+1, 2*n), float, order='F')
    c = ones((1, n), complex)
    d = ones(dtype=int, shape=2+n)

    return np_sum(a) + np_sum(b) + np_sum(c) + np_sum(d)


#==============================================================================
# NEGATIVE INDEXES
#==============================================================================

@types('int')
def constant_negative_index(n):
    import numpy as np
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i

    return a[-1], a[-2]

@types('int')
def almost_negative_index(n):
    import numpy as np
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i
    j = -1

    return a[-j]

@allow_negative_index('a')
@types('int', 'int')
def var_negative_index(n, idx):
    import numpy as np
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i

    return a[idx]

@allow_negative_index('a')
@types('int', 'int', 'int')
def expr_negative_index(n, idx_1, idx_2):
    import numpy as np
    a = np.empty(n, dtype=int)

    for i in range(n):
        a[i] = i

    return a[idx_1-idx_2]

@allow_negative_index('a')
@allow_negative_index('b')
@types('int', 'int')
def test_multiple_negative_index(c, d):
    import numpy as np
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([1, 2, 3])
    x = a[c]
    y = b[d]

    return x, y

@allow_negative_index('a', 'b')
@types('int', 'int')
def test_multiple_negative_index_2(c, d):
    import numpy as np
    a = np.array([1.2, 2.2, 3.2, 4.2])
    b = np.array([1, 5, 9, 13])

    x = a[c] * d
    y = b[d] * c

    return x, y

@allow_negative_index('a')
@allow_negative_index('b', 'c')
@types('int', 'int', 'int')
def test_multiple_negative_index_3(d, e, f):
    import numpy as np
    a = np.array([1.2, 2.2, 3.2, 4.2])
    b = np.array([1])
    c = np.array([1, 2, 3])

    return a[d], b[e], c[f]

@allow_negative_index('a')
@types('int[:]')
def test_argument_negative_index_1(a):
    c = -2
    d = 5
    return a[c], a[d]

@allow_negative_index('a', 'b')
@types('int[:]', 'int[:]')
def test_argument_negative_index_2(a, b):
    c = -2
    d = 3
    return a[c], a[d], b[c], b[d]

#==============================================================================
# SHAPE INITIALISATION
#==============================================================================

def array_random_size():
    import numpy as np
    a = np.zeros(np.random.randint(23))
    c = np.zeros_like(a)
    return np.shape(a)[0], np.shape(c)[0]

@types('int','int')
def array_variable_size(n,m):
    import numpy as np
    s = n
    a = np.zeros(s)
    s = m
    c = np.zeros_like(a)
    return np.shape(a)[0], np.shape(c)[0]

#==============================================================================
# 1D ARRAY SLICING
#==============================================================================

@types('int[:]')
def array_1d_slice_1(a):
    import numpy as np
    b = a[:]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_2(a):
    import numpy as np
    b = a[5:]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_3(a):
    import numpy as np
    b = a[:5]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_4(a):
    import numpy as np
    b = a[5:15]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_5(a):
    import numpy as np
    b = a[:-5]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_6(a):
    import numpy as np
    b = a[-5:]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_7(a):
    import numpy as np
    b = a[-15:-5]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_8(a):
    import numpy as np
    b = a[5:-5]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_9(a):
    import numpy as np
    b = a[-15:15]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
@types('int[:]')
def array_1d_slice_10(a):
    import numpy as np
    c = -15
    b = a[c:]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
@types('int[:]')
def array_1d_slice_11(a):
    import numpy as np
    c = -5
    b = a[:c]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
@types('int[:]')
def array_1d_slice_12(a):
    import numpy as np
    c = -15
    d = -5
    b = a[c:d]
    return np.sum(b), b[0], b[-1], len(b)

#==============================================================================
# 2D ARRAY SLICE ORDER F
#==============================================================================

@types('int[:,:](order=F)')
def array_2d_F_slice_1(a):
    import numpy as np
    b = a[:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_2(a):
    import numpy as np
    b = a[5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_3(a):
    import numpy as np
    b = a[:5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_4(a):
    import numpy as np
    b = a[-15:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_5(a):
    import numpy as np
    b = a[:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_6(a):
    import numpy as np
    b = a[5:15]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_7(a):
    import numpy as np
    b = a[-15:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_8(a):
    import numpy as np
    b = a[::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_9(a):
    import numpy as np
    b = a[5:, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_10(a):
    import numpy as np
    b = a[:5, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_11(a):
    import numpy as np
    b = a[:, 5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_12(a):
    import numpy as np
    b = a[:, :5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_13(a):
    import numpy as np
    b = a[:-5, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_14(a):
    import numpy as np
    b = a[-5:, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_15(a):
    import numpy as np
    b = a[:, -5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_16(a):
    import numpy as np
    b = a[:, :-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_17(a):
    import numpy as np
    b = a[:, 5:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_18(a):
    import numpy as np
    b = a[5:15, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])


@types('int[:,:](order=F)')
def array_2d_F_slice_19(a):
    import numpy as np
    b = a[5:15, -5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_20(a):
    import numpy as np
    b = a[5:15, 5:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:](order=F)')
def array_2d_F_slice_21(a):
    import numpy as np
    c = -5
    d = 5
    b = a[d:15, 5:c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:](order=F)')
def array_2d_F_slice_22(a):
    import numpy as np
    c = -5
    d = -15
    b = a[d:15, 5:c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:](order=F)')
def array_2d_F_slice_23(a):
    import numpy as np
    c = -5
    b = a[:c, :c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

#==============================================================================
# 2D ARRAY SLICE ORDER C
#==============================================================================
@types('int[:,:]')
def array_2d_C_slice_1(a):
    import numpy as np
    b = a[:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_2(a):
    import numpy as np
    b = a[5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_3(a):
    import numpy as np
    b = a[:5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_4(a):
    import numpy as np
    b = a[-15:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_5(a):
    import numpy as np
    b = a[:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_6(a):
    import numpy as np
    b = a[5:15]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_7(a):
    import numpy as np
    b = a[-15:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_8(a):
    import numpy as np
    b = a[::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_9(a):
    import numpy as np
    b = a[5:, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_10(a):
    import numpy as np
    b = a[:5, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_11(a):
    import numpy as np
    b = a[:, 5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_12(a):
    import numpy as np
    b = a[:, :5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_13(a):
    import numpy as np
    b = a[:-5, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_14(a):
    import numpy as np
    b = a[-5:, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_15(a):
    import numpy as np
    b = a[:, -5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_16(a):
    import numpy as np
    b = a[:, :-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_17(a):
    import numpy as np
    b = a[:, 5:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_18(a):
    import numpy as np
    b = a[5:15, :]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])


@types('int[:,:]')
def array_2d_C_slice_19(a):
    import numpy as np
    b = a[5:15, -5:]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_20(a):
    import numpy as np
    b = a[5:15, 5:-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:]')
def array_2d_C_slice_21(a):
    import numpy as np
    c = -5
    d = 5
    b = a[d:15, 5:c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:]')
def array_2d_C_slice_22(a):
    import numpy as np
    c = -5
    d = -15
    b = a[d:15, 5:c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:]')
def array_2d_C_slice_23(a):
    import numpy as np
    c = -5
    b = a[:c, :c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

#==============================================================================
# 1D ARRAY SLICE STRIDE
#==============================================================================
@types('int[:]')
def array_1d_slice_stride_1(a):
    import numpy as np
    b = a[::1]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_2(a):
    import numpy as np
    b = a[::-1]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_3(a):
    import numpy as np
    b = a[::2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_4(a):
    import numpy as np
    b = a[::-2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_5(a):
    import numpy as np
    b = a[5::2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_6(a):
    import numpy as np
    b = a[5::-2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_7(a):
    import numpy as np
    b = a[:15:2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_8(a):
    import numpy as np
    b = a[:15:-2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_9(a):
    import numpy as np
    b = a[5:15:2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_10(a):
    import numpy as np
    b = a[15:5:-2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_11(a):
    import numpy as np
    b = a[-15:-5:2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_12(a):
    import numpy as np
    b = a[-5:-15:-2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_13(a):
    import numpy as np
    b = a[-5::2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_14(a):
    import numpy as np
    b = a[:-5:-2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_15(a):
    import numpy as np
    b = a[::-5]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_16(a):
    import numpy as np
    b = a[-15::2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_17(a):
    import numpy as np
    b = a[:-15:-2]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_18(a):
    import numpy as np
    b = a[5::-5]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_19(a):
    import numpy as np
    b = a[5:-5:5]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_20(a):
    import numpy as np
    b = a[-5:5:-5]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
@types('int[:]')
def array_1d_slice_stride_21(a):
    import numpy as np
    c = -5
    b = a[-5:5:c]
    return np.sum(b), b[0], b[-1], len(b)

@types('int[:]')
def array_1d_slice_stride_22(a):
    import numpy as np
    c = 5
    b = a[5:-5:c]
    return np.sum(b), b[0], b[-1], len(b)

@allow_negative_index('a')
@types('int[:]')
def array_1d_slice_stride_23(a):
    import numpy as np
    c = -5
    b = a[::c]
    return np.sum(b), b[0], b[-1], len(b)

#==============================================================================
# 2D ARRAY SLICE STRIDE ORDER F
#==============================================================================

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_1(a):
    import numpy as np
    b = a[::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_2(a):
    import numpy as np
    b = a[::-1]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_3(a):
    import numpy as np
    b = a[::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_4(a):
    import numpy as np
    b = a[::, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_5(a):
    import numpy as np
    b = a[::, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_6(a):
    import numpy as np
    b = a[::2, ::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_7(a):
    import numpy as np
    b = a[::-2, ::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_8(a):
    import numpy as np
    b = a[::2, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_9(a):
    import numpy as np
    b = a[::-2, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_10(a):
    import numpy as np
    b = a[::2, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_11(a):
    import numpy as np
    b = a[::-2, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_12(a):
    import numpy as np
    b = a[5:15:2, 15:5:-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_13(a):
    import numpy as np
    b = a[15:5:-2, 5:15]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_14(a):
    import numpy as np
    b = a[-15:-5:2, -5:-15:-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_15(a):
    import numpy as np
    b = a[-5:-15:-2, -15:-5:2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_16(a):
    import numpy as np
    b = a[::-5, ::5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_17(a):
    import numpy as np
    b = a[::5, ::-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_18(a):
    import numpy as np
    b = a[::-1, ::-1]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_19(a):
    import numpy as np
    b = a[5:15:3, 15:5:-3]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:](order=F)')
def array_2d_F_slice_stride_20(a):
    import numpy as np
    b = a[::-10, ::-10]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:](order=F)')
def array_2d_F_slice_stride_21(a):
    import numpy as np
    c = -5
    b = a[::c, ::c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:](order=F)')
def array_2d_F_slice_stride_22(a):
    import numpy as np
    c = 5
    d = -10
    b = a[::c, ::d]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:](order=F)')
def array_2d_F_slice_stride_23(a):
    import numpy as np
    c = 10
    d = -5
    b = a[::d, ::c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

#==============================================================================
# 2D ARRAY SLICE STRIDE ORDER C
#==============================================================================

@types('int[:,:]')
def array_2d_C_slice_stride_1(a):
    import numpy as np
    b = a[::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_2(a):
    import numpy as np
    b = a[::-1]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_3(a):
    import numpy as np
    b = a[::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_4(a):
    import numpy as np
    b = a[::, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_5(a):
    import numpy as np
    b = a[::, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_6(a):
    import numpy as np
    b = a[::2, ::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_7(a):
    import numpy as np
    b = a[::-2, ::]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_8(a):
    import numpy as np
    b = a[::2, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_9(a):
    import numpy as np
    b = a[::-2, ::2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_10(a):
    import numpy as np
    b = a[::2, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_11(a):
    import numpy as np
    b = a[::-2, ::-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_12(a):
    import numpy as np
    b = a[5:15:2, 15:5:-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_13(a):
    import numpy as np
    b = a[15:5:-2, 5:15]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_14(a):
    import numpy as np
    b = a[-15:-5:2, -5:-15:-2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_15(a):
    import numpy as np
    b = a[-5:-15:-2, -15:-5:2]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_16(a):
    import numpy as np
    b = a[::-5, ::5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_17(a):
    import numpy as np
    b = a[::5, ::-5]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_18(a):
    import numpy as np
    b = a[::-1, ::-1]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_19(a):
    import numpy as np
    b = a[5:15:3, 15:5:-3]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@types('int[:,:]')
def array_2d_C_slice_stride_20(a):
    import numpy as np
    b = a[::-10, ::-10]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:]')
def array_2d_C_slice_stride_21(a):
    import numpy as np
    c = -5
    b = a[::c, ::c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:]')
def array_2d_C_slice_stride_22(a):
    import numpy as np
    c = -5
    d = 10
    b = a[::c, ::d]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

@allow_negative_index('a')
@types('int[:,:]')
def array_2d_C_slice_stride_23(a):
    import numpy as np
    c = -10
    d = 5
    b = a[::d, ::c]
    return np.sum(b), b[0][0], b[-1][-1], len(b), len(b[0])

#==============================================================================
# ARITHMETIC OPERATIONS
#==============================================================================

def arrs_similar_shapes_0():
    import numpy as np
    a = np.zeros(10)
    b = a[2:4]+a[4:6]
    return np.shape(b)[0]

def arrs_similar_shapes_1():
    import numpy as np
    i = 4
    a = np.zeros(10)
    b = a[2:i]+a[4:i + 2]
    return np.shape(b)[0]

def arrs_different_shapes_0():
    import numpy as np
    i = 5
    a = np.zeros(10)
    b = a[2:4]+a[4:i]
    return np.shape(b)[0]

def arrs_uncertain_shape_1():
    import numpy as np
    i = 4
    j = 6
    a = np.zeros(10)
    b = a[2:i]+a[4:j]
    return np.shape(b)[0]

def arrs_2d_similar_shapes_0():
    import numpy as np
    from numpy import shape
    dy = 4
    dx = 2
    pn = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x = ((dy**2 * (pn[1:shape(pn)[0]-1, 2:] + pn[1:shape(pn)[0]-1, 0:shape(pn)[1]-2]) +
        dx**2 *(pn[2:, 1:shape(pn)[1]-1] + pn[0:shape(pn)[0]-2, 1:shape(pn)[1]-1])) / (2 * (dx**2 + dy**2)))
    return np.shape(x)[0], np.shape(x)[1]

def arrs_2d_different_shapes_0():
    import numpy as np
    pn = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    pm = np.array([[1, 1, 1]])
    x = pn + pm
    return np.shape(x)[0], np.shape(x)[1]
def arrs_1d_negative_index_1():
    import numpy as np
    a = np.zeros(10)
    b = a[:-1]+a[-9:]
    return np.shape(b)[0], np.sum(b)

def arrs_1d_negative_index_2():
    import numpy as np
    a = np.ones(10)
    b = a[1:-1] + a[2:]
    return np.shape(b)[0], np.sum(b)

def arrs_1d_int32_index():
    import numpy as np
    i = np.int32(1)
    a = np.ones(10)
    b = a[i] + a[i + 2]
    return b

def arrs_1d_int64_index():
    import numpy as np
    i = np.int64(1)
    a = np.ones(10)
    b = a[i] + a[i + 2]
    return b

def arrs_1d_negative_index_negative_step():
    import numpy as np
    a = np.ones(10)
    b = a[-1:1:-2] + a[:2:-2]
    return np.shape(b)[0], np.sum(b)

def arrs_1d_negative_step_positive_step():
    import numpy as np
    a = np.ones(10)
    b = a[1:-1: 3] + a[2::3]
    return np.shape(b)[0], np.sum(b)

def arrs_2d_negative_index():
    import numpy as np
    a = np.ones((10, 10))
    b = a[1:-1, :-1] + a[2:, -9:]
    return np.shape(b)[0], np.shape(b)[1], np.sum(b)

#==============================================================================
# NUMPY ARANGE
#==============================================================================

def arr_arange_1():
    import numpy as np
    a = np.arange(6)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_2():
    import numpy as np
    a = np.arange(1, 7)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_3():
    import numpy as np
    a = np.arange(0, 10, 0.3)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_4():
    import numpy as np
    a = np.arange(1, 28, 3, dtype=np.float)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_5():
    import numpy as np
    a = np.arange(20, 2.2, -2)
    return np.shape(a)[0], a[0], a[-1]

def arr_arange_6():
    import numpy as np
    a = np.arange(20, 1, -1.1)
    return np.shape(a)[0], a[0], a[-1]

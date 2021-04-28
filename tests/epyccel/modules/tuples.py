# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types, pure

__all__ = [
        'homogenous_tuple_int',
        'homogenous_tuple_bool',
        'homogenous_tuple_float',
        'homogenous_tuple_string',
        'homogenous_tuple_math',
        'inhomogenous_tuple_1',
        'inhomogenous_tuple_2',
        'inhomogenous_tuple_3',
        'inhomogenous_tuple_2_levels_1',
        'inhomogenous_tuple_2_levels_2',
        'homogeneous_tuple_2_levels',
        'tuple_unpacking_1',
        'tuple_unpacking_2',
        'tuple_name_clash',
        'tuples_as_indexing_basic',
        'tuples_as_indexing_var',
        'tuple_multi_indexing_1',
        'tuple_multi_indexing_2',
        'tuple_inhomogeneous_return',
        'tuple_homogeneous_return',
        'tuple_arg_unpacking',
        'tuple_indexing_basic',
        'tuple_indexing_2d',
        'tuple_visitation_homogeneous',
        'tuple_visitation_inhomogeneous',
        'tuples_homogeneous_have_pointers',
        'tuples_inhomogeneous_have_pointers',
        'tuples_homogeneous_copies_have_pointers',
        'tuples_inhomogeneous_copies_have_pointers',
        'tuples_mul_homogeneous',
        'tuples_mul_homogeneous2',
        'tuples_mul_homogeneous3',
        'tuples_mul_homogeneous4',
        'tuples_mul_inhomogeneous',
        'tuples_mul_inhomogeneous2',
        'tuples_mul_homogeneous_2d',
        'tuples_mul_mixed_homogeneous_2d',
        'tuples_mul_inhomogeneous_2d',
        'tuples_add_homogeneous',
        'tuples_add_homogeneous_variables',
        'tuples_add_homogeneous_with_variables',
        'tuples_add_homogeneous_with_variables2',
        'tuples_add_inhomogeneous',
        'tuples_add_inhomogeneous_variables',
        'tuples_add_inhomogeneous_with_variables',
        'tuples_add_inhomogeneous_with_variables2',
        'tuples_add_mixed_homogeneous',
        'tuples_add_mixed_homogeneous_variables',
        'tuples_add_mixed_homogeneous_with_variables',
        'tuples_2d_sum',
        ]

def homogenous_tuple_int():
    ai = (1,4,5)
    return ai[0], ai[1], ai[2]

def homogenous_tuple_bool():
    ai = (False, True)
    return ai[0], ai[1]

def homogenous_tuple_float():
    ai = (1.5, 4.3, 5.2, 7.2, 9.999)
    return ai[0], ai[1], ai[2], ai[3], ai[4]

def homogenous_tuple_string():
    ai = ('hello', 'tuple', 'world', '!!')
    return ai[0], ai[1], ai[2], ai[3]

def homogenous_tuple_math():
    ai = (4+5,3*9, 2**3)
    return ai[0], ai[1], ai[2]

def inhomogenous_tuple_1():
    ai = (0, False, 3+1j)
    return ai[0], ai[1], ai[2]

def inhomogenous_tuple_2():
    ai = (0, False, 3)
    return ai[0], ai[1], ai[2]

def inhomogenous_tuple_3():
    ai = (0, 1.0, 3)
    return ai[0], ai[1], ai[2]

def inhomogenous_tuple_2_levels_1():
    ai = ((1,2), (4,False), (3.0, 'boo'))
    return ai[0][0], ai[0][1], ai[1][0], ai[1][1], ai[2][0]

def inhomogenous_tuple_2_levels_2():
    ai = ((0,1,2), (True,False,True))
    return ai[0][0], ai[0][1] ,ai[0][2], ai[1][0], ai[1][1], ai[1][2]

def homogeneous_tuple_2_levels():
    ai = ((0,1,2), (3,4,5))
    return ai[0][0], ai[0][1] ,ai[0][2], ai[1][0], ai[1][1], ai[1][2]

def tuple_unpacking_1():
    ai = (1,False,3.5)
    a,b,c = ai
    return a,b,c

def tuple_unpacking_2():
    a,b,c = 1,False,3.5
    return a,b,c

def tuple_unpacking_3(x : 'int[:,:]'):
    x[0,0], x[1,0] = 2, 2

def tuple_unpacking_4(x : 'int[:,:]'):
    x[:,0], x[0,:] = 2, 3

def tuple_name_clash():
    ai = (1+2j, False, 10.4)
    ai_0 = 44
    return ai_0, ai[0], ai[1], ai[2]

def tuples_as_indexing_basic():
    from numpy import ones
    x = ones((2,3,2))
    for z in range(2):
        for y in range(3):
            for w in range(2):
                x[z,y,w] = w+y*2+z*6
    idx = (1,1,0)
    return x[idx]

def tuples_as_indexing_var():
    from numpy import ones
    x = ones((2,3,2))
    for z in range(2):
        for y in range(3):
            for w in range(2):
                x[z,y,w] = w+y*2+z*6
    idx_0 = 1
    idx = (1,idx_0,0)
    return x[idx]

def tuple_multi_indexing_1():
    from numpy import ones
    x = ones((2,3,2))
    for z in range(2):
        for y in range(3):
            for w in range(2):
                x[z,y,w] = w+y*2+z*6
    idx = (1,1,0)
    ai = x[idx,0,1]
    return ai[0], ai[1], ai[2]

def tuple_multi_indexing_2():
    from numpy import ones
    x = ones((2,3,2))
    for z in range(2):
        for y in range(3):
            for w in range(2):
                x[z,y,w] = w+y*2+z*6
    idx = (1,1,0)
    idx_2 = (0,1,2)
    ai = x[idx,idx_2,1]
    return ai[0], ai[1], ai[2]

def tuple_inhomogeneous_return():
    ai = (7.5, False, 8)
    return ai

def tuple_homogeneous_return():
    ai = (7.5, 4.2, 8)
    return ai

def tuple_arg_unpacking():
    @pure
    @types('int','int')
    def add2(x, y):
        return x+y

    args = (3,4)
    z = add2(*args)
    return z

def tuple_indexing_basic():
    ai = (1,2,3,4)
    z = 0
    for i in range(4):
        z += ai[i]
    return z

def tuple_indexing_2d():
    ai = ((0,1,2), (True,False,True))
    z = 0
    for i in range(3):
        if ai[1][i]:
            z += ai[0][i]
    return z

def tuple_visitation_inhomogeneous():
    ai = (1,3.5, False)
    for a in ai:
        print(a)

def tuple_visitation_homogeneous():
    ai = (1,5, 4)
    for a in ai:
        print(a)

def tuples_homogeneous_have_pointers():
    from numpy import zeros
    a = zeros(2)
    b = zeros(2)
    c = (a,b)
    a[1] = 4
    return c[0][0], c[0][1], c[1][0], c[1][1]

def tuples_inhomogeneous_have_pointers():
    from numpy import zeros
    a = zeros(2)
    b = zeros(3)
    c = (a,b)
    a[1] = 4
    return c[0][0], c[0][1], c[1][0], c[1][1], c[1][2]

def tuples_homogeneous_copies_have_pointers():
    from numpy import zeros
    a = zeros(2)
    b = zeros(2)
    c = (a,b)
    d = c
    a[1] = 4
    return d[0][0], d[0][1], d[1][0], d[1][1]

def tuples_inhomogeneous_copies_have_pointers():
    from numpy import zeros
    a = zeros(2)
    b = zeros(3)
    c = (a,b)
    d = c
    a[1] = 4
    return d[0][0], d[0][1], d[1][0], d[1][1], d[1][2]

def tuples_mul_homogeneous():
    a = (1,2,3)
    b = a*2
    return b[0], b[1], b[2], b[3], b[4], b[5]

def tuples_mul_homogeneous2():
    a = (1,2,3)
    b = 2*a
    return b[0], b[1], b[2], b[3], b[4], b[5]

def tuples_mul_homogeneous3():
    a = (1,2,3)
    s = 2
    b = a*s
    return b[0], b[1], b[2], b[3], b[4], b[5]

def tuples_mul_homogeneous4():
    s = 2
    b = (1,2,3)*s
    return b[0], b[1], b[2], b[3], b[4], b[5]

def tuples_mul_inhomogeneous():
    a = (1,False)
    b = a*3
    return b[0], b[1], b[2], b[3], b[4], b[5]

def tuples_mul_inhomogeneous2():
    a = (1,False)
    b = 3*a
    return b[0], b[1], b[2], b[3], b[4], b[5]

def tuples_mul_homogeneous_2d():
    a= ((1,2), (3,4), (5,6))
    b = a*2
    return b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], \
            b[3][0], b[3][1], b[4][0], b[4][1], b[5][0], b[5][1]

def tuples_mul_mixed_homogeneous_2d():
    a= ((1,2), (True,False), (5,6))
    b = a*2
    return b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], \
            b[3][0], b[3][1], b[4][0], b[4][1], b[5][0], b[5][1]

def tuples_mul_inhomogeneous_2d():
    a= ((1,False), (3.0,False), (True,6))
    b = a*2
    return b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], \
            b[3][0], b[3][1], b[4][0], b[4][1], b[5][0], b[5][1]

def tuples_add_homogeneous():
    a = (1,2,3) + (4,5,6)
    return a[0], a[1], a[2], a[3], a[4], a[5]

def tuples_add_homogeneous_variables():
    a = (1,2,3)
    b = (4,5,6)
    c = a + b
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_homogeneous_with_variables():
    a = (1,2,3)
    c = a + (4,5,6)
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_homogeneous_with_variables2():
    a = (1,2,3)
    c = (4,5,6) + a
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_inhomogeneous():
    a = (1,2,True) + (False,5,6)
    return a[0], a[1], a[2], a[3], a[4], a[5]

def tuples_add_inhomogeneous_variables():
    a = (1,2,False)
    b = (4,5,True)
    c = a + b
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_inhomogeneous_with_variables():
    a = (1,2,True)
    c = a + (4,False,6)
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_inhomogeneous_with_variables2():
    a = (1,2,True)
    c = (4,False,6) + a
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_mixed_homogeneous():
    a = (1,2,3) + (False,5,6)
    return a[0], a[1], a[2], a[3], a[4], a[5]

def tuples_add_mixed_homogeneous_variables():
    a = (1,2,3)
    b = (4,5,True)
    c = a + b
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_add_mixed_homogeneous_with_variables():
    a = (1,2,3)
    c = a + (4,False,6)
    return c[0], c[1], c[2], c[3], c[4], c[5]

def tuples_2d_sum():
    a = ((1,2), (3,4))
    b = a + ((5,6),)
    return b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1]

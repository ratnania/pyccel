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
    ai = ((1,2), (4,False))
    return ai[0][0], ai[0][1], ai[1][0], ai[1][1]

def inhomogenous_tuple_2_levels_2():
    ai = ((0,1,2), (True,False,True))
    return ai[0][0], ai[0][1] ,ai[0][2], ai[1][0], ai[1][1], ai[1][2]

def homogeneous_tuple_2_levels():
    ai = ((0,1,2), (3,4,5))
    return ai[0][0], ai[0][1] ,ai[0][2], ai[1][0], ai[1][1], ai[1][2]

def tuple_unpacking_1():
    ai = (1,False,3.0)
    a,b,c = ai
    return a,b,c

def tuple_unpacking_2():
    a,b,c = 1,False,3.0
    return a,b,c

def tuple_name_clash():
    ai = (1+2j, False, 10.0)
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

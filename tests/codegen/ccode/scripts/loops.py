from pyccel.decorators import types

#==============================================================================

@types( int )
def sum_natural_numbers( n ):
    x = 0
    for i in range( 1, n+1 ):
        x += i
    return x

# ...
@types( int )
def factorial( n ):
    x = 1
    for i in range( 2, n+1 ):
        x *= i
    return x

# ...
@types( int )
def fibonacci( n ):
    x = 0
    y = 1
    for i in range( n ):
        z = x+y
        x = y
        y = z
    return x

# ...
@types( int )
def double_loop( n ):
    x = 0
    for i in range( 3, 10 ):
        x += 1
        y  = n*x
        for j in range( 4, 15 ):
            z = x-y
    return z

# ...
#@types( 'int[:,:](order=C)' )
#def double_loop_on_2d_array_C( z ):
#
#    from numpy import shape
#
#    s = shape( z )
#    m = s[0]
#    n = s[1]
#
#    for i in range( m ):
#        for j in range( n ):
#            z[i,j] = i-j


# ...
#@types( 'int[:,:](order=F)' )
#def double_loop_on_2d_array_F( z ):
#
#    from numpy import shape
#
#    s = shape( z )
#    m = s[0]
#    n = s[1]
#
#    for i in range( m ):
#        for j in range( n ):
#            z[i,j] = i-j

# ...
#@types( 'real[:], real[:]' )
#def product_loop_on_real_array( z, out ):
#
#    from numpy     import shape
#
#    s = shape( z )
#    n = s[0]
#
#    for i in range(n):
#        out[i] = z[i]**2

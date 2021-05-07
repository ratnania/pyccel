# pylint: disable=missing-function-docstring, missing-module-docstring/
# ...
def add( x: 'int', y: 'int' ):
    return x+y

# ...
def sub( x: 'int', y: 'int' ):
    return x-y

# ...
def mul( x: 'int', y: 'int' ):
    return x*y

# ...
def div( x: 'int', y: 'int' ):
    return x/y

# ...
def idiv( x: 'int', y: 'int' ):
    return x//y

# ...
def aug_add( x: 'int', y: 'int' ):
    x += y

# ...
def aug_sub( x: 'int', y: 'int' ):
    x -= y

# ...
def aug_mul( x: 'int', y: 'int' ):
    x *= y

# ...
def aug_div( x: 'int', y: 'int' ):
    x /= y

# ...
def sum_natural_numbers( n: 'int' ):
    x = 0
    for i in range( 1, n+1 ):
        x += i
    return x

# ...
def factorial( n: 'int' ):
    x = 1
    for i in range( 2, n+1 ):
        x *= i
    return x

# ...
def fibonacci( n: 'int' ):
    x = 0
    y = 1
    for i in range( n ): # pylint: disable=unused-variable
        z = x+y
        x = y
        y = z
    return x

# ...
def double_loop( n: 'int' ):
    x = 0
    for i in range( 3, 10 ): # pylint: disable=unused-variable
        x += 1
        y  = n*x
        for j in range( 4, 15 ): # pylint: disable=unused-variable
            z = x-y
    return z

# ...
def double_loop_on_2d_array_C( z: 'int[:,:](order=C)' ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j

# ...
def double_loop_on_2d_array_F( z: 'int[:,:](order=F)' ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j

# ...
def array_int32_1d_scalar_add( x: 'int32[:]', a: 'int32' ):
    x[:] += a

# ...
def array_int32_1d_scalar_sub( x: 'int32[:]', a: 'int32' ):
    x[:] -= a

# ...
def array_int32_1d_scalar_mul( x: 'int32[:]', a: 'int32' ):
    x[:] *= a

# ...
def array_int32_1d_scalar_div( x: 'int32[:]', a: 'int32' ):
    x[:] = x / a

# ...
def array_int32_1d_scalar_idiv( x: 'int32[:]', a: 'int32' ):
    x[:] = x // a

# ...
def array_int32_2d_scalar_add( x: 'int32[:,:]', a: 'int32' ):
    x[:,:] += a

# ...
def array_int32_2d_scalar_sub( x: 'int32[:,:]', a: 'int32' ):
    x[:,:] -= a

# ...
def array_int32_2d_scalar_mul( x: 'int32[:,:]', a: 'int32' ):
    x[:,:] *= a

# ...
def array_int32_2d_scalar_div( x: 'int32[:,:]', a: 'int32' ):
    x[:,:] = x / a

# ...
def array_int32_2d_scalar_idiv( x: 'int32[:,:]', a: 'int32' ):
    x[:,:] = x // a

# ...
def array_int32_1d_add( x: 'int32[:]', y: 'int32[:]' ):
    x[:] += y

# ...
def array_int32_1d_sub( x: 'int32[:]', y: 'int32[:]' ):
    x[:] -= y

# ...
def array_int32_1d_mul( x: 'int32[:]', y: 'int32[:]' ):
    x[:] *= y

# ...
def array_int32_1d_idiv( x: 'int32[:]', y: 'int32[:]' ):
    x[:] = x // y

# ...
def array_int32_2d_add( x: 'int32[:,:]', y: 'int32[:,:]' ):
    x[:,:] += y

# ...
def array_int32_2d_sub( x: 'int32[:,:]', y: 'int32[:,:]' ):
    x[:,:] -= y

# ...
def array_int32_2d_mul( x: 'int32[:,:]', y: 'int32[:,:]' ):
    x[:,:] *= y

# ...
def array_int32_2d_idiv( x: 'int32[:,:]', y: 'int32[:,:]' ):
    x[:,:] = x // y

# ...
def array_int32_1d_scalar_add_stride1( x: 'int32[:,:]', a: 'int32' ):
    x[1:10] += a

# ...
def array_int32_1d_scalar_add_stride2( x: 'int32[:,:]', a: 'int32' ):
    x[1:10, 2:5] += a

# ...
def array_int32_1d_scalar_add_stride3( x: 'int32[:,:]', a: 'int32' ):
    x[:5, 2:5] += a

# ...
def array_int32_1d_scalar_add_stride4( x: 'int32[:,:]', a: 'int32' ):
    x[:5, 2:] += a

# ...
def sum_natural_numbers_range_step_int( n: 'int' ):
    x = 0
    for i in range( 1, n+1, 5 ):
        x += i
    return x

# ...
def sum_natural_numbers_range_step_variable( n: 'int', b: 'int' ):
    x = 0
    for i in range( 1, n+1, b ):
        x += i
    return x

# ...
def abs_real_scalar( x: 'real' ):
    return abs(x)

# ...
def floor_real_scalar( x: 'real' ):
    from numpy import floor
    return floor(x)

# ...
def exp_real_scalar( x: 'real' ):
    from numpy import exp
    return exp(x)

# ...
def log_real_scalar( x: 'real' ):
    from numpy import log
    return log(x)

# ...
def sqrt_real_scalar( x: 'real' ):
    from numpy import sqrt
    return sqrt(x)

# ...
def sin_real_scalar( x: 'real' ):
    from numpy import sin
    return sin(x)

# ...
def cos_real_scalar( x: 'real' ):
    from numpy import cos
    return cos(x)

# ...
def tan_real_scalar( x: 'real' ):
    from numpy import tan
    return tan(x)

# ...
def arcsin_real_scalar( x: 'real' ):
    from numpy import arcsin
    return arcsin(x)

# ...
def arccos_real_scalar( x: 'real' ):
    from numpy import arccos
    return arccos(x)

# ...
def arctan_real_scalar( x: 'real' ):
    from numpy import arctan
    return arctan(x)

# ...
def sinh_real_scalar( x: 'real' ):
    from numpy import sinh
    return sinh(x)

# ...
def cosh_real_scalar( x: 'real' ):
    from numpy import cosh
    return cosh(x)

# ...
def tanh_real_scalar( x: 'real' ):
    from numpy import tanh
    return tanh(x)
# ...
def arcsinh_real_scalar( x: 'real' ):
    from numpy import arcsinh
    return arcsinh(x)

# ...
def arccosh_real_scalar( x: 'real' ):
    from numpy import arccosh
    return arccosh(x)

# ...
def arctanh_real_scalar( x: 'real' ):
    from numpy import arctanh
    return arctanh(x)

# ...
def arctan2_real_scalar( y: 'real', x: 'real' ):
    from numpy import arctan2
    return arctan2(y, x)

# ...
def sin_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import sin
    out[:] = sin(x)

# ...
def cos_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import cos
    out[:] = cos(x)

# ...
def tan_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import tan
    out[:] = tan(x)

# ...
def arcsin_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import arcsin
    out[:] = arcsin(x)

# ...
def arccos_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import arccos
    out[:] = arccos(x)

# ...
def arctan_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import arctan
    out[:] = arctan(x)

# ...
def sinh_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import sinh
    out[:] = sinh(x)

# ...
def cosh_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import cosh
    out[:] = cosh(x)

# ...
def tanh_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import tanh
    out[:] = tanh(x)
# ...
def arcsinh_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import arcsinh
    out[:] = arcsinh(x)

# ...
def arccosh_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import arccosh
    out[:] = arccosh(x)

# ...
def arctanh_real_array_1d( x: 'real[:]', out: 'real[:]' ):
    from numpy import arctanh
    out[:] = arctanh(x)

# ...
def arctan2_real_array_1d( x: 'real[:]', y: 'real[:]', out: 'real[:]' ):
    from numpy import arctan2
    out[:] = arctan2(y, x)

# ...
def numpy_math_expr_real_scalar( x: 'real', y: 'real', z: 'real' ):
    from numpy import sin
    return sin(x*2+y/z)

# ...
def numpy_math_expr_real_array_1d( x: 'real[:]', y: 'real[:]', z: 'real[:]', out: 'real[:]' ):
    from numpy import sin
    out[:] = sin(x*2+y/z)

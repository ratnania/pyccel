# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
from numpy.random import rand, randint, uniform
from numpy import isclose

from pyccel.decorators import types
from pyccel.epyccel import epyccel

import sys

max_float = 3.40282e5        # maximum positive float
min_float = sys.float_info.min  # Minimum positive float

def test_fabs_call(language):
    @types('real')
    def fabs_call(x):
        from math import fabs
        return fabs(x)

    f1 = epyccel(fabs_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  fabs_call(x), rtol=1e-14, atol=1e-15))

def test_fabs_phrase(language):
    @types('real','real')
    def fabs_phrase(x,y):
        from math import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  fabs_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_fabs_return_type(language):
    @types('int')
    def fabs_return_type(x):
        from math import fabs
        a = fabs(x)
        return a

    f1 = epyccel(fabs_return_type, language = language)
    x = randint(100)
    assert(isclose(f1(x) ,  fabs_return_type(x), rtol=1e-14, atol=1e-15))
    assert(type(f1(x))  == type(fabs_return_type(x))) # pylint: disable=unidiomatic-typecheck

def test_sqrt_call(language):
    @types('real')
    def sqrt_call(x):
        from math import sqrt
        return sqrt(x)

    f1 = epyccel(sqrt_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sqrt_call(x), rtol=1e-14, atol=1e-15))

def test_sqrt_phrase(language):
    @types('real','real')
    def sqrt_phrase(x,y):
        from math import sqrt
        a = sqrt(x)*sqrt(y)
        return a

    f2 = epyccel(sqrt_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  sqrt_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_sqrt_return_type(language):
    @types('real')
    def sqrt_return_type_real(x):
        from math import sqrt
        a = sqrt(x)
        return a

    f1 = epyccel(sqrt_return_type_real, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sqrt_return_type_real(x), rtol=1e-14, atol=1e-15))
    assert(type(f1(x))  == type(sqrt_return_type_real(x))) # pylint: disable=unidiomatic-typecheck

def test_sin_call(language):
    @types('real')
    def sin_call(x):
        from math import sin
        return sin(x)

    f1 = epyccel(sin_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sin_call(x), rtol=1e-14, atol=1e-15))

def test_sin_phrase(language):
    @types('real','real')
    def sin_phrase(x,y):
        from math import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  sin_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_cos_call(language):
    @types('real')
    def cos_call(x):
        from math import cos
        return cos(x)

    f1 = epyccel(cos_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  cos_call(x), rtol=1e-14, atol=1e-15))

def test_cos_phrase(language):
    @types('real','real')
    def cos_phrase(x,y):
        from math import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  cos_phrase(x,y), rtol=1e-14, atol=1e-15))


def test_tan_call(language):
    @types('real')
    def tan_call(x):
        from math import tan
        return tan(x)

    f1 = epyccel(tan_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  tan_call(x), rtol=1e-14, atol=1e-15))


def test_tan_phrase(language):
    @types('real','real')
    def tan_phrase(x,y):
        from math import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  tan_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_exp_call(language):
    @types('real')
    def exp_call(x):
        from math import exp
        return exp(x)

    f1 = epyccel(exp_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  exp_call(x), rtol=1e-14, atol=1e-15))

def test_exp_phrase(language):
    @types('real','real')
    def exp_phrase(x,y):
        from math import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  exp_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_log_call(language):
    @types('real')
    def log_call(x):
        from math import log
        return log(x)

    f1 = epyccel(log_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  log_call(x), rtol=1e-14, atol=1e-15))

def test_log_phrase(language):
    @types('real','real')
    def log_phrase(x,y):
        from math import log
        a = log(x)+log(y)
        return a

    f2 = epyccel(log_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  log_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_asin_call(language):
    @types('real')
    def asin_call(x):
        from math import asin
        return asin(x)

    f1 = epyccel(asin_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  asin_call(x), rtol=1e-14, atol=1e-15))

def test_asin_phrase(language):
    @types('real','real')
    def asin_phrase(x,y):
        from math import asin
        a = asin(x)+asin(y)
        return a

    f2 = epyccel(asin_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  asin_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_acos_call(language):
    @types('real')
    def acos_call(x):
        from math import acos
        return acos(x)

    f1 = epyccel(acos_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  acos_call(x), rtol=1e-14, atol=1e-15))

def test_acos_phrase(language):
    @types('real','real')
    def acos_phrase(x,y):
        from math import acos
        a = acos(x)+acos(y)
        return a

    f2 = epyccel(acos_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  acos_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_atan_call(language):
    @types('real')
    def atan_call(x):
        from math import atan
        return atan(x)

    f1 = epyccel(atan_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  atan_call(x), rtol=1e-14, atol=1e-15))

def test_atan_phrase(language):
    @types('real','real')
    def atan_phrase(x,y):
        from math import atan
        a = atan(x)+atan(y)
        return a

    f2 = epyccel(atan_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  atan_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_sinh_call(language):
    @types('real')
    def sinh_call(x):
        from math import sinh
        return sinh(x)

    f1 = epyccel(sinh_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sinh_call(x), rtol=1e-14, atol=1e-15))

def test_sinh_phrase(language):
    @types('real','real')
    def sinh_phrase(x,y):
        from math import sinh
        a = sinh(x)+sinh(y)
        return a

    f2 = epyccel(sinh_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  sinh_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_cosh_call(language):
    @types('real')
    def cosh_call(x):
        from math import cosh
        return cosh(x)

    f1 = epyccel(cosh_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  cosh_call(x), rtol=1e-14, atol=1e-15))

def test_cosh_phrase(language):
    @types('real','real')
    def cosh_phrase(x,y):
        from math import cosh
        a = cosh(x)+cosh(y)
        return a

    f2 = epyccel(cosh_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  cosh_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_tanh_call(language):
    @types('real')
    def tanh_call(x):
        from math import tanh
        return tanh(x)

    f1 = epyccel(tanh_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  tanh_call(x), rtol=1e-14, atol=1e-15))

def test_tanh_phrase(language):
    @types('real','real')
    def tanh_phrase(x,y):
        from math import tanh
        a = tanh(x)+tanh(y)
        return a

    f2 = epyccel(tanh_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  tanh_phrase(x,y), rtol=1e-14, atol=1e-15))

def test_atan2_call(language):
    @types('real', 'real')
    def atan2_call(x, y):
        from math import atan2
        return atan2(x, y)

    f1 = epyccel(atan2_call, language = language)
    x = rand()
    y = rand()
    assert(isclose(f1(x, y), atan2_call(x, y), rtol=1e-14, atol=1e-15))

def test_atan2_phrase(language):
    @types('real', 'real', 'real')
    def atan2_phrase(x, y, z):
        from math import atan2
        a = atan2(x, y)+atan2(y, z)
        return a

    f2 = epyccel(atan2_phrase, language = language)
    x = rand()
    y = rand()
    z = rand()
    assert(isclose(f2(x, y, z), atan2_phrase(x, y, z), rtol=1e-14, atol=1e-15))

#------------------------------- Floor function ------------------------------#
def test_floor_call(language):
    @types('real')
    def floor_call(x):
        from math import floor
        return floor(x)

    fflags = "-Werror -Wconversion"
    f1 = epyccel(floor_call, language = language, fflags=fflags)
    x = rand()
    assert(isclose(f1(x) ,  floor_call(x), rtol=1e-14, atol=1e-15))
    assert(isclose(f1(-x) ,  floor_call(-x), rtol=1e-14, atol=1e-15))

def test_floor_phrase(language):
    @types('real','real')
    def floor_phrase(x,y):
        from math import floor
        a = floor(x)*floor(y)
        return a

    fflags = "-Werror -Wconversion"
    f2 = epyccel(floor_phrase, language = language, fflags=fflags)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  floor_phrase(x,y), rtol=1e-14, atol=1e-15))
    assert(isclose(f2(-x,y) ,  floor_phrase(-x,y), rtol=1e-14, atol=1e-15))
    assert(isclose(f2(x,-y) ,  floor_phrase(x,-y), rtol=1e-14, atol=1e-15))
    assert(isclose(f2(-x,-y) ,  floor_phrase(-x,-y), rtol=1e-14, atol=1e-15))

def test_floor_return_type(language):
    @types('int')
    def floor_return_type_int(x):
        from math import floor
        a = floor(x)
        return a

    @types('real')
    def floor_return_type_real(x):
        from math import floor
        a = floor(x)
        return a

    fflags = "-Werror -Wconversion"
    f1 = epyccel(floor_return_type_int, language = language, fflags=fflags)

    x = randint(100)
    assert(isclose(f1(x) ,  floor_return_type_int(x), rtol=1e-14, atol=1e-15))
    assert(isclose(f1(-x) ,  floor_return_type_int(-x), rtol=1e-14, atol=1e-15))
    assert(type(f1(x))  == type(floor_return_type_int(x))) # pylint: disable=unidiomatic-typecheck

    fflags = "-Werror -Wconversion"
    f1 = epyccel(floor_return_type_real, language = language, fflags=fflags)

    x = randint(100)
    assert(isclose(f1(x) ,  floor_return_type_real(x), rtol=1e-14, atol=1e-15))
    assert(isclose(f1(-x) ,  floor_return_type_real(-x), rtol=1e-14, atol=1e-15))
    assert(type(f1(x))  == type(floor_return_type_real(x))) # pylint: disable=unidiomatic-typecheck

#------------------------------- Ceil function -------------------------------#
def test_ceil_call_r(language):
    @types('real')
    def ceil_call(x):
        from math import ceil
        return ceil(x)

    fflags = "-Werror -Wconversion"
    f1 = epyccel(ceil_call, language = language, fflags=fflags)

    x = rand()
    assert(ceil_call(x) == f1(x))
    assert(ceil_call(-x) == f1(-x))

    assert isinstance(ceil_call(x), type(f1(x)))

def test_ceil_call_i(language):
    @types('int')
    def ceil_call(x):
        from math import ceil
        return ceil(x)

    fflags = "-Werror -Wconversion"
    f1 = epyccel(ceil_call, language = language, fflags=fflags)

    x = randint(10)
    assert(ceil_call(x) == f1(x))
    assert(ceil_call(-x) == f1(-x))

    assert isinstance(ceil_call(x), type(f1(x)))

def test_ceil_phrase(language):
    @types('real','real')
    def ceil_phrase(x,y):
        from math import ceil
        a = ceil(x)*ceil(y)
        return a

    fflags = "-Werror -Wconversion"
    f2 = epyccel(ceil_phrase, language = language, fflags=fflags)

    x = rand()
    y = rand()
    assert(isclose(ceil_phrase(x, y), f2(x, y), rtol=1e-14, atol=1e-15))
    assert(isclose(ceil_phrase(-x, y), f2(-x, y), rtol=1e-14, atol=1e-15))
    assert(isclose(ceil_phrase(x, -y), f2(x, -y), rtol=1e-14, atol=1e-15))
    assert(isclose(ceil_phrase(-x, -y), f2(-x, -y), rtol=1e-14, atol=1e-15))
#------------------------------- copysign function -------------------------------#

def test_copysign_call(language):
    @types('real', 'real')
    def copysign_call(x, y):
        from math import copysign
        return copysign(x, y)

    f1 = epyccel(copysign_call, language = language)
    x = rand()
    y = rand()
    # Same sign
    assert(isclose(copysign_call(x, y), f1(x, y), rtol=1e-14, atol=1e-15))
    assert(isclose(copysign_call(-x, -y), f1(-x, -y), rtol=1e-14, atol=1e-15))
    # Different sign
    assert(isclose(copysign_call(-x, y), f1(-x, y), rtol=1e-14, atol=1e-15))
    assert(isclose(copysign_call(x, -y), f1(x, -y), rtol=1e-14, atol=1e-15))
    # x =/= 0, y = 0 and x = 0, y =/= 0
    assert(isclose(copysign_call(x, 0), f1(x, 0), rtol=1e-14, atol=1e-15))
    assert(isclose(copysign_call(0, y), f1(0, y), rtol=1e-14, atol=1e-15))

def test_copysign_call_zero_case(language):
    @types('int', 'int')
    def copysign_zero_case(x, y):
        from math import copysign
        return copysign(x, y)

    f1 = epyccel(copysign_zero_case, language = language)
    x = 0
    y = 0
    # Same sign
    assert(isclose(copysign_zero_case(x, y), f1(x, y), rtol=1e-14, atol=1e-15))
    assert(isclose(copysign_zero_case(-x, -y), f1(-x, -y), rtol=1e-14, atol=1e-15))
    # Different sign
    assert(isclose(copysign_zero_case(-x, y), f1(-x, y), rtol=1e-14, atol=1e-15))
    assert(isclose(copysign_zero_case(x, -y), f1(x, -y), rtol=1e-14, atol=1e-15))

def test_copysign_return_type_1(language): # copysign
    '''test type copysign(real, real) => should return real number'''
    @types('real', 'real')
    def copysign_return_type(x, y):
        from math import copysign
        a = copysign(x, y)
        return a

    f1 = epyccel(copysign_return_type, language = language)
    x = rand() # real
    y = rand() # real

    # Same sign
    assert(type(f1(x, y)) == type(copysign_return_type(x, y)))
    assert(type(f1(-x, -y)) == type(copysign_return_type(-x, -y)))
    # Different sign
    assert(type(f1(-x, y)) == type(copysign_return_type(-x, y)))
    assert(type(f1(x, -y)) == type(copysign_return_type(x, -y)))

def test_copysign_return_type_2(language): # copysign
    '''test type copysign(int, int) => should return real type'''
    @types('int', 'int')
    def copysign_return_type(x, y):
        from math import copysign
        a = copysign(x, y)
        return a

    f1 = epyccel(copysign_return_type, language = language)
    high = 10000000
    x = randint(high)   # int
    y = randint(high)   # int

    # Same sign
    assert(type(f1(x, y)) == type(copysign_return_type(x, y)))
    assert(type(f1(-x, -y)) == type(copysign_return_type(-x, -y)))
    # Different sign
    assert(type(f1(-x, y)) == type(copysign_return_type(-x, y)))
    assert(type(f1(x, -y)) == type(copysign_return_type(x, -y)))

def test_copysign_return_type_3(language): # copysign
    '''test type copysign(int, real) => should return real type'''
    @types('int', 'real')
    def copysign_return_type(x, y):
        from math import copysign
        a = copysign(x, y)
        return a

    f1 = epyccel(copysign_return_type, language = language)
    high = 10000000
    x = randint(high)   # int
    y = rand()          # real

    # Same sign
    assert(type(f1(x, y)) == type(copysign_return_type(x, y)))
    assert(type(f1(-x, -y)) == type(copysign_return_type(-x, -y)))
    # Different sign
    assert(type(f1(-x, y)) == type(copysign_return_type(-x, y)))
    assert(type(f1(x, -y)) == type(copysign_return_type(x, -y)))

def test_copysign_return_type_4(language): # copysign
    '''test type copysign(real, int) => should return real type'''
    @types('real', 'int')
    def copysign_return_type(x, y):
        from math import copysign
        a = copysign(x, y)
        return a

    f1 = epyccel(copysign_return_type, language = language)
    high = 10000000
    x = rand()          # real
    y = randint(high)   # int

    # Same sign
    assert(type(f1(x, y)) == type(copysign_return_type(x, y)))
    assert(type(f1(-x, -y)) == type(copysign_return_type(-x, -y)))
    # Different sign
    assert(type(f1(-x, y)) == type(copysign_return_type(-x, y)))
    assert(type(f1(x, -y)) == type(copysign_return_type(x, -y)))


#----------------------------- isfinite function -----------------------------#
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="isfinite not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_isfinite_call(language): # isfinite
    @types('real')
    def isfinite_call(x):
        from math import isfinite
        return isfinite(x)

    f1 = epyccel(isfinite_call, language = language)
    x = rand()

    assert(isfinite_call(x) == f1(x))

    from math import nan, inf
    # Test not a number
    assert(isfinite_call(nan) == f1(nan))
    # Test infinite number
    assert(isfinite_call(inf) == f1(inf))
    # Test negative infinite number
    assert(isfinite_call(-inf) == f1(-inf))

#------------------------------- isinf function ------------------------------#
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="isinf not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_isinf_call(language): # isinf
    @types('real')
    def isinf_call(x):
        from math import isinf
        return isinf(x)

    f1 = epyccel(isinf_call, language = language)
    x = rand()

    assert(isinf_call(x) == f1(x))

    from math import nan, inf
    # Test not a number
    assert(isinf_call(nan) == f1(nan))
    # Test infinite number
    assert(isinf_call(inf) == f1(inf))
    # Test negative infinite number
    assert(isinf_call(-inf) == f1(-inf))

#------------------------------- isnan function ------------------------------#

def test_isnan_call(language): # isnan
    @types('real')
    def isnan_call(x):
        from math import isnan
        return isnan(x)

    f1 = epyccel(isnan_call, language = language)
    x = rand()

    assert(isnan_call(x) == f1(x))

    from math import nan, inf
    # Test not a number
    assert(isnan_call(nan) == f1(nan))
    # Test infinite number
    assert(isnan_call(inf) == f1(inf))
    # Test negative infinite number
    assert(isnan_call(-inf) == f1(-inf))

#------------------------------- ldexp function ------------------------------#
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="ldexp not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_ldexp_call(language): # ldexp
    @types('real', 'int')
    def ldexp_call(x, exp):
        from math import ldexp
        return ldexp(x, exp)

    f1 = epyccel(ldexp_call, language = language)
    high = 100
    x = rand()
    exp = randint(high)

    assert(isclose(ldexp_call(x, exp), f1(x, exp), rtol=1e-14, atol=1e-15))
    # Negative exponent
    assert(isclose(ldexp_call(x, -exp), f1(x, -exp), rtol=1e-14, atol=1e-15))
    # Negative value
    assert(isclose(ldexp_call(-x, exp), f1(-x, exp), rtol=1e-14, atol=1e-15))
    # Negative value and negative exponent
    assert(isclose(ldexp_call(-x, -exp), f1(-x, -exp), rtol=1e-14, atol=1e-15))

@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="ldexp not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_ldexp_return_type(language): # ldexp
    @types('real', 'int')
    def ldexp_type(x, exp):
        from math import ldexp
        return ldexp(x, exp)

    f1 = epyccel(ldexp_type, language = language)
    high = 100
    x = rand()
    exp = randint(high)

    assert(type(ldexp_type(x, exp)) == type(f1(x, exp)))
    # Negative exponent
    assert(type(ldexp_type(x, -exp)) == type(f1(x, -exp)))
    # Negative value
    assert(type(ldexp_type(-x, exp)) == type(f1(-x, exp)))
    # Negative value and negative exponent
    assert(type(ldexp_type(-x, -exp)) == type(f1(-x, -exp)))

#--------------------------- remainder function ------------------------------#

@pytest.mark.skipif(sys.version_info < (3, 7), reason="requires python3.7 or higher")
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="remainder not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_remainder_call(language): # remainder
    @types('real', 'real')
    def remainder_call(x, y):
        from math import remainder
        return remainder(x, y)

    f1 = epyccel(remainder_call, language = language)
    x = rand()
    y = rand() + 1
    # Same sign
    assert(isclose(remainder_call(x, y), f1(x, y), rtol=1e-14, atol=1e-15))
    assert(isclose(remainder_call(-x, -y), f1(-x, -y), rtol=1e-14, atol=1e-15))

    # Different sign
    assert(isclose(remainder_call(x, -y), f1(x, -y), rtol=1e-14, atol=1e-15))
    assert(isclose(remainder_call(-x, y), f1(-x, y), rtol=1e-14, atol=1e-15))

@pytest.mark.skipif(sys.version_info < (3, 7), reason="requires python3.7 or higher")
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="remainder not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_remainder_return_type(language): # remainder
    @types('real', 'real')
    def remainder_type(x, y):
        from math import remainder
        return remainder(x, y)

    f1 = epyccel(remainder_type, language = language)
    x = rand()
    y = rand()

    # Same sign
    assert(type(remainder_type(x, y)) == type(f1(x, y)))
    assert(type(remainder_type(-x, -y)) == type(f1(-x, -y)))

    # Different sign
    assert(type(remainder_type(x, -y)) == type(f1(x, -y)))
    assert(type(remainder_type(-x, y)) == type(f1(-x, y)))

#----------------------------- trunc function --------------------------------#

def test_trunc_call(language): # trunc
    @types('real')
    def trunc_call(x):
        from math import trunc
        return trunc(x)

    f1 = epyccel(trunc_call, language = language)
    x = uniform(high = 10000.0)

    # positive number
    assert(trunc_call(x) == f1(x))
    # Negative number
    assert(trunc_call(-x) == f1(-x))

def test_trunc_call_int(language): # trunc
    @types('int')
    def trunc_call(x):
        from math import trunc
        return trunc((x))

    f1 = epyccel(trunc_call, language = language)
    high = 10000
    x = randint(high)

    # positive number
    assert(trunc_call(x) == f1(x))
    # Negative number
    assert(trunc_call(-x) == f1(-x))

def test_trunc_return_type(language): # trunc
    @types('real')
    def trunc_type(x):
        from math import trunc
        return trunc(x)

    f1 = epyccel(trunc_type, language = language)
    x = uniform(high = 10000.0)

    assert(type(trunc_type((x))) == type(f1((x))))
    assert(type(trunc_type(-x)) == type(f1(-x)))

#--------------------------- expm1 function ------------------------------#
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="expm1 not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_expm1_call(language): # expm1
    @types('real')
    def expm1_call(x):
        from math import expm1
        return expm1(x)

    f1 = epyccel(expm1_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  expm1_call(x), rtol=1e-14, atol=1e-15))

@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="expm1 not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_expm1_call_special_case(language): # expm1
    @types('real')
    def expm1_call(x):
        from math import expm1
        return expm1(x)
    # should give result accurate to full precision better than exp()
    x = 1e-5
    f1 = epyccel(expm1_call, language = language)
    assert(isclose(f1(x), expm1_call(x), rtol=1e-14, atol=1e-15))

@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="expm1 not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_expm1_phrase(language): # expm1
    @types('real','real')
    def expm1_phrase(x,y):
        from math import expm1
        a = expm1(x)+expm1(y)
        return a

    f2 = epyccel(expm1_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  expm1_phrase(x,y), rtol=1e-14, atol=1e-15))

@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="expm1 not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_expm1_return_type(language): # expm1 # expm1
    @types('real')
    def expm1_type(x):
        from math import expm1
        return expm1(x)

    f1 = epyccel(expm1_type, language = language)
    x = uniform(high = 700.0)

    assert(type(expm1_type(x)) == type(f1(x)))
    assert(type(expm1_type(-x)) == type(f1(-x)))

#--------------------------- log1p function ------------------------------#

@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="log1p not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_log1p_call(language):
    @types('real')
    def log1p_call(x):
        from math import log1p
        return log1p(x)

    f1 = epyccel(log1p_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  log1p_call(x), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x), type(log1p_call(x)))

@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="log1p not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_log1p_phrase(language):
    @types('real','real')
    def log1p_phrase(x,y):
        from math import log1p
        a = log1p(x)+log1p(y)
        return a

    f2 = epyccel(log1p_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  log1p_phrase(x,y), rtol=1e-14, atol=1e-15))

#--------------------------- log2 function ------------------------------#
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="log2 not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_log2_call(language):
    @types('real')
    def log2_call(x):
        from math import log2
        return log2(x)

    f1 = epyccel(log2_call, language = language)
    low = min_float
    high = max_float
    x = uniform(low=low, high=high)
    assert(isclose(f1(x) ,  log2_call(x), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x), type(log2_call(x)))

@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="log2 not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_log2_phrase(language):
    @types('real','real')
    def log2_phrase(x,y):
        from math import log2
        a = log2(x)+log2(y)
        return a

    f2 = epyccel(log2_phrase, language = language)
    low = min_float
    high = max_float
    x = uniform(low=low, high=high)
    y = uniform(low=low, high=high)
    assert(isclose(f2(x,y) ,  log2_phrase(x,y), rtol=1e-14, atol=1e-15))

#--------------------------- log10 function ------------------------------#

def test_log10_call(language):
    @types('real')
    def log10_call(x):
        from math import log10
        return log10(x)

    f1 = epyccel(log10_call, language = language)
    low = min_float
    high = max_float
    x = uniform(low=low, high=high)
    assert(isclose(f1(x) ,  log10_call(x), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x), type(log10_call(x)))

def test_log10_phrase(language):
    @types('real','real')
    def log10_phrase(x,y):
        from math import log10
        a = log10(x)+log10(y)
        return a

    f2 = epyccel(log10_phrase, language = language)
    low = min_float
    high = max_float
    x = uniform(low=low, high=high)
    y = uniform(low=low, high=high)
    assert(isclose(f2(x,y) ,  log10_phrase(x,y), rtol=1e-14, atol=1e-15))

#--------------------------------- Pow function ------------------------------#

def test_pow_call(language):
    @types('real', 'real')
    def pow_call(x, y):
        from math import pow as my_pow
        return my_pow(x, y)

    f1 = epyccel(pow_call, language = language)
    high = 10
    # case 1: x > 0
    x = uniform(low=min_float)
    y = uniform(low=-high, high=high)
    assert(isclose(f1(x, y) , pow_call(x, y), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x, y), type(pow_call(x, y)))

    # case 2: x = 0 and y > 0
    x = 0.0
    y = uniform(high=high)
    assert(isclose(f1(x, y), pow_call(x, y), rtol=1e-14, atol=1e-15))

    # case 3: x < 0 and y is integer
    x = uniform(low=-high, high=0)
    y = randint(high)
    assert(isclose(f1(x, y), pow_call(x, y), rtol=1e-14, atol=1e-15))

#------------------------------- Hypot function ------------------------------#

def test_hypot_call(language):
    @types('real', 'real')
    def hypot_call(x, y):
        from math import hypot
        return hypot(x, y)

    f1 = epyccel(hypot_call, language = language)
    high = 10
    x = uniform(low=-high, high=high)
    y = uniform(low=-high, high=high)
    assert(isclose(f1(x, y), hypot_call(x, y), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x, y), type(hypot_call(x, y)))

#------------------------------- Acosh function ------------------------------#

def test_acosh_call(language):
    @types('real')
    def acosh_call(x):
        from math import acosh
        return acosh(x)

    f1 = epyccel(acosh_call, language = language)

    x = uniform(low=1, high=max_float)
    assert(isclose(f1(x) ,  acosh_call(x), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x), type(acosh_call(x)))

def test_acosh_phrase(language):
    @types('real','real')
    def acosh_phrase(x,y):
        from math import acosh
        a = acosh(x) + acosh(y)
        return a

    f2 = epyccel(acosh_phrase, language = language)

    x = uniform(low=1, high=max_float)
    y = uniform(low=1, high=max_float)
    assert(isclose(f2(x,y) , acosh_phrase(x,y), rtol=1e-14, atol=1e-15))


#------------------------------- Asinh function ------------------------------#

def test_asinh_call(language):
    @types('real')
    def asinh_call(x):
        from math import asinh
        return asinh(x)

    f1 = epyccel(asinh_call, language = language)

    x = uniform(high=max_float)
    assert(isclose(f1(x) , asinh_call(x), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x), type(asinh_call(x)))

    # Negative value
    assert(isclose(f1(-x) , asinh_call(-x), rtol=1e-14, atol=1e-15))

def test_asinh_phrase(language):
    @types('real','real')
    def asinh_phrase(x,y):
        from math import asinh
        a = asinh(x)+ asinh(y)
        return a

    f2 = epyccel(asinh_phrase, language = language)
    x = uniform(high=max_float)
    y = uniform(high=max_float)
    assert(isclose(f2(x,y), asinh_phrase(x,y), rtol=1e-14, atol=1e-15))
    # Negative value
    assert(isclose(f2(-x,-y), asinh_phrase(-x,-y), rtol=1e-14, atol=1e-15))

#------------------------------- Atanh function ------------------------------#

def test_atanh_call(language):
    @types('real')
    def atanh_call(x):
        from math import atanh
        return atanh(x)

    f1 = epyccel(atanh_call, language = language)
    low = -1 + min_float
    high = 1 - min_float
    x = uniform(low=low, high=high)
    assert(isclose(f1(x) , atanh_call(x), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x), type(atanh_call(x)))

def test_atanh_phrase(language):
    @types('real','real')
    def atanh_phrase(x, y):
        from math import atanh
        a = atanh(x)+ atanh(y)
        return a

    f2 = epyccel(atanh_phrase, language = language)

    # Domain ]-1, 1[
    low = -1 + min_float
    high = 1 - min_float
    x = uniform(low=low, high=high)
    y = uniform(low=low, high=high)
    assert(isclose(f2(x, y), atanh_phrase(x, y), rtol=1e-14, atol=1e-15))

#--------------------------------- Erf function ------------------------------#

def test_erf_call(language):
    @types('real')
    def erf_call(x):
        from math import erf
        return erf(x)

    f1 = epyccel(erf_call, language = language)

    # Domain ]-inf, +inf[
    x = uniform(high=max_float)
    assert(isclose(f1(x) , erf_call(x), rtol=1e-14, atol=1e-15))
    assert(isclose(f1(-x) , erf_call(-x), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x), type(erf_call(x)))

def test_erf_phrase(language):
    @types('real','real')
    def erf_phrase(x, y):
        from math import erf
        a = erf(x)+ erf(y)
        return a

    f2 = epyccel(erf_phrase, language = language)

    # Domain ]-inf, +inf[
    x = uniform(high=max_float)
    y = uniform(high=max_float)
    assert(isclose(f2(x, y), erf_phrase(x, y), rtol=1e-14, atol=1e-15))
    assert(isclose(f2(-x, -y), erf_phrase(-x, -y), rtol=1e-14, atol=1e-15))

#-------------------------------- Erfc function ------------------------------#

def test_erfc_call(language):
    @types('real')
    def erfc_call(x):
        from math import erfc
        return erfc(x)

    f1 = epyccel(erfc_call, language = language)

    # Domain ]-inf, +inf[
    x = uniform(high=max_float)
    assert(isclose(f1(x) , erfc_call(x), rtol=1e-14, atol=1e-15))
    assert(isclose(f1(-x) , erfc_call(-x), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x), type(erfc_call(x)))

def test_erfc_phrase(language):
    @types('real','real')
    def erfc_phrase(x, y):
        from math import erfc
        a = erfc(x)+ erfc(y)
        return a

    f2 = epyccel(erfc_phrase, language = language)

    # Domain ]-inf, +inf[
    x = uniform(high=max_float)
    y = uniform(high=max_float)
    assert(isclose(f2(x, y), erfc_phrase(x, y), rtol=1e-14, atol=1e-15))
    assert(isclose(f2(-x, -y), erfc_phrase(-x, -y), rtol=1e-14, atol=1e-15))

#-------------------------------- gamma function -----------------------------#

def test_gamma_call(language):
    @types('real')
    def gamma_call(x):
        from math import gamma
        return gamma(x)

    f1 = epyccel(gamma_call, language = language)

    # Domain ]0, +inf[ || (x < 0 and x.fraction not null)
    x = uniform(low=min_float)
    assert(isclose(f1(x) , gamma_call(x), rtol=1e-14, atol=1e-15))
    from math import modf
    # make fractional part different from zero to test negative case
    if modf(x)[0] == 0:
        x += - 0.1
    assert(isclose(f1(-x) , gamma_call(-x), rtol=1e-14, atol=1e-15))

    assert isinstance(f1(x), type(gamma_call(x)))

def test_gamma_phrase(language):
    @types('real','real')
    def gamma_phrase(x, y):
        from math import gamma
        a = gamma(x)+ gamma(y)
        return a

    f2 = epyccel(gamma_phrase, language = language)

    # Domain ]0, +inf[ || (x < 0 and fractional part of x not null)
    x = uniform(low=min_float)
    y = uniform(low=min_float)
    assert(isclose(f2(x, y), gamma_phrase(x, y), rtol=1e-14, atol=1e-15))

#------------------------------- lgamma function -----------------------------#

def test_lgamma_call(language):
    @types('real')
    def lgamma_call(x):
        from math import lgamma
        return lgamma(x)

    f1 = epyccel(lgamma_call, language = language)

    # Domain ]0, +inf[ || (x < 0 and x.fraction not null)
    x = uniform(low=min_float)
    assert(isclose(f1(x) , lgamma_call(x), rtol=1e-14, atol=1e-15))
    from math import modf
    _, f = modf(x)
    # make fractional part different from zero to test negative case
    if f == 0:
        x += - 0.1
    assert(isclose(f1(-x) , lgamma_call(-x), rtol=1e-14, atol=1e-15))
    assert isinstance(f1(x), type(lgamma_call(x)))

def test_lgamma_phrase(language):
    @types('real','real')
    def lgamma_phrase(x, y):
        from math import lgamma
        a = lgamma(x)+ lgamma(y)
        return a

    f2 = epyccel(lgamma_phrase, language = language)

    # Domain ]0, +inf[ || (x < 0 and fractional part of x not null)
    x = uniform(low=min_float)
    y = uniform(low=min_float)
    assert(isclose(f2(x, y), lgamma_phrase(x, y), rtol=1e-14, atol=1e-15))

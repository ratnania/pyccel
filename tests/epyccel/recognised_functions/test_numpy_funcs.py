# pylint: disable=missing-function-docstring, missing-module-docstring, unidiomatic-typecheck/
import sys
import pytest
from numpy.random import rand, randint, uniform
from numpy import isclose

from pyccel.decorators import types
from pyccel.epyccel import epyccel

min_float = sys.float_info.min  # Minimum positive float

# Functions still to be tested:
#    array
#    # ...
#    norm
#    int
#    real
#    imag
#    float
#    double
#    mod
#    float32
#    float64
#    int32
#    int64
#    complex128
#    complex64
#    matmul
#    prod
#    product
#    linspace
#    diag
#    where
#    cross
#    # ---

# Relative and absolute tolerances for array comparisons in the form
# numpy.isclose(a, b, rtol, atol). Windows has larger round-off errors.
if sys.platform == 'win32':
    RTOL = 1e-13
    ATOL = 1e-14
else:
    RTOL = 1e-14
    ATOL = 1e-15

#-------------------------------- Fabs function ------------------------------#
def test_fabs_call_r(language):
    @types('real')
    def fabs_call_r(x):
        from numpy import fabs
        return fabs(x)

    f1 = epyccel(fabs_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), fabs_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), fabs_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(fabs_call_r(x).item()))

def test_fabs_call_i(language):
    @types('int')
    def fabs_call_i(x):
        from numpy import fabs
        return fabs(x)

    f1 = epyccel(fabs_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), fabs_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), fabs_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(fabs_call_i(x).item()))

def test_fabs_phrase_r_r(language):
    @types('real','real')
    def fabs_phrase_r_r(x,y):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), fabs_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), fabs_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), fabs_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), fabs_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))

def test_fabs_phrase_i_i(language):
    @types('int','int')
    def fabs_phrase_i_i(x,y):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase_i_i, language = language)
    x = randint(1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), fabs_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), fabs_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), fabs_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), fabs_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))

def test_fabs_phrase_r_i(language):
    @types('real','int')
    def fabs_phrase_r_i(x,y):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), fabs_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), fabs_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), fabs_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), fabs_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))

def test_fabs_phrase_i_r(language):
    @types('int','real')
    def fabs_phrase_r_i(x,y):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase_r_i, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), fabs_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), fabs_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), fabs_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), fabs_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))

#------------------------------ absolute function ----------------------------#
def test_absolute_call_r(language):
    @types('real')
    def absolute_call_r(x):
        from numpy import absolute
        return absolute(x)

    f1 = epyccel(absolute_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), absolute_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), absolute_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(absolute_call_r(x).item()))

def test_absolute_call_i(language):
    @types('int')
    def absolute_call_i(x):
        from numpy import absolute
        return absolute(x)

    f1 = epyccel(absolute_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), absolute_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), absolute_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(absolute_call_i(x).item()))

def test_absolute_phrase_r_r(language):
    @types('real','real')
    def absolute_phrase_r_r(x,y):
        from numpy import absolute
        a = absolute(x)*absolute(y)
        return a

    f2 = epyccel(absolute_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), absolute_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), absolute_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), absolute_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), absolute_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_absolute_phrase_i_r(language):
    @types('int','real')
    def absolute_phrase_i_r(x,y):
        from numpy import absolute
        a = absolute(x)*absolute(y)
        return a

    f2 = epyccel(absolute_phrase_i_r, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), absolute_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), absolute_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), absolute_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), absolute_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_absolute_phrase_r_i(language):
    @types('real','int')
    def absolute_phrase_r_i(x,y):
        from numpy import absolute
        a = absolute(x)*absolute(y)
        return a

    f2 = epyccel(absolute_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), absolute_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), absolute_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), absolute_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), absolute_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- sin function ------------------------------#
def test_sin_call_r(language):
    @types('real')
    def sin_call_r(x):
        from numpy import sin
        return sin(x)

    f1 = epyccel(sin_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), sin_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), sin_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(sin_call_r(x).item()))

def test_sin_call_i(language):
    @types('int')
    def sin_call_i(x):
        from numpy import sin
        return sin(x)

    f1 = epyccel(sin_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), sin_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), sin_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(sin_call_i(x).item()))

def test_sin_phrase_r_r(language):
    @types('real','real')
    def sin_phrase_r_r(x,y):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), sin_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sin_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sin_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sin_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_sin_phrase_i_i(language):
    @types('int','int')
    def sin_phrase_i_i(x,y):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase_i_i, language = language)
    x = randint(1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), sin_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sin_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sin_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sin_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))

def test_sin_phrase_i_r(language):
    @types('int','real')
    def sin_phrase_i_r(x,y):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase_i_r, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), sin_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sin_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sin_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sin_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_sin_phrase_r_i(language):
    @types('real','int')
    def sin_phrase_r_i(x,y):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), sin_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sin_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sin_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sin_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- cos function ------------------------------#
def test_cos_call_i(language):
    @types('int')
    def cos_call_i(x):
        from numpy import cos
        return cos(x)

    f1 = epyccel(cos_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), cos_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), cos_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(cos_call_i(x).item()))

def test_cos_call_r(language):
    @types('real')
    def cos_call_r(x):
        from numpy import cos
        return cos(x)

    f1 = epyccel(cos_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), cos_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), cos_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(cos_call_r(x).item()))


def test_cos_phrase_i_i(language):
    @types('int','int')
    def cos_phrase_i_i(x,y):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase_i_i, language = language)
    x = randint(1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), cos_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cos_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cos_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cos_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))

def test_cos_phrase_r_r(language):
    @types('real','real')
    def cos_phrase_r_r(x,y):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), cos_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cos_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cos_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cos_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_cos_phrase_i_r(language):
    @types('int','real')
    def cos_phrase_i_r(x,y):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase_i_r, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), cos_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cos_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cos_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cos_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_cos_phrase_r_i(language):
    @types('real','int')
    def cos_phrase_r_i(x,y):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), cos_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cos_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cos_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cos_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- tan function ------------------------------#
def test_tan_call_i(language):
    @types('int')
    def tan_call_i(x):
        from numpy import tan
        return tan(x)

    f1 = epyccel(tan_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), tan_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), tan_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(tan_call_i(x).item()))

def test_tan_call_r(language):
    @types('real')
    def tan_call_r(x):
        from numpy import tan
        return tan(x)

    f1 = epyccel(tan_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), tan_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), tan_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(tan_call_r(x).item()))

def test_tan_phrase_i_i(language):
    @types('int','int')
    def tan_phrase_i_i(x,y):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase_i_i, language = language)
    x = randint(1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), tan_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tan_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tan_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tan_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))

def test_tan_phrase_r_r(language):
    @types('real','real')
    def tan_phrase_r_r(x,y):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), tan_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tan_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tan_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tan_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_tan_phrase_i_r(language):
    @types('int','real')
    def tan_phrase_i_r(x,y):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase_i_r, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), tan_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tan_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tan_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tan_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_tan_phrase_r_i(language):
    @types('real','int')
    def tan_phrase_r_i(x,y):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), tan_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tan_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tan_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tan_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- exp function ------------------------------#
def test_exp_call_i(language):
    @types('int')
    def exp_call_i(x):
        from numpy import exp
        return exp(x)

    f1 = epyccel(exp_call_i, language = language)
    x = randint(1e2)
    assert(isclose(f1(x), exp_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), exp_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(exp_call_i(x).item()))

def test_exp_call_r(language):
    @types('real')
    def exp_call_r(x):
        from numpy import exp
        return exp(x)

    f1 = epyccel(exp_call_r, language = language)
    x = uniform(high=1e2)
    assert(isclose(f1(x), exp_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), exp_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(exp_call_r(x).item()))

def test_exp_phrase_i_i(language):
    @types('int','int')
    def exp_phrase_i_i(x,y):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase_i_i, language = language)
    x = randint(1e2)
    y = randint(1e2)
    assert(isclose(f2(x,y), exp_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), exp_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), exp_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), exp_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))

def test_exp_phrase_r_r(language):
    @types('real','real')
    def exp_phrase_r_r(x,y):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase_r_r, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), exp_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), exp_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), exp_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), exp_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_exp_phrase_i_r(language):
    @types('int','real')
    def exp_phrase_i_r(x,y):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase_i_r, language = language)
    x = randint(1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), exp_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), exp_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), exp_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), exp_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_exp_phrase_r_i(language):
    @types('real','int')
    def exp_phrase_r_i(x,y):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase_r_i, language = language)
    x = uniform(high=1e2)
    y = randint(1e2)
    assert(isclose(f2(x,y), exp_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,y), exp_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,y), exp_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), exp_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- log function ------------------------------#
def test_log_call_i(language):
    @types('int')
    def log_call_i(x):
        from numpy import log
        return log(x)

    f1 = epyccel(log_call_i, language = language)
    x = randint(low=min_float, high=1e6)
    assert(isclose(f1(x), log_call_i(x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(log_call_i(x).item()))

def test_log_call_r(language):
    @types('real')
    def log_call_r(x):
        from numpy import log
        return log(x)

    f1 = epyccel(log_call_r, language = language)
    x = uniform(low=min_float, high=1e6)
    assert(isclose(f1(x), log_call_r(x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(log_call_r(x).item()))

def test_log_phrase(language):
    @types('real','real')
    def log_phrase(x,y):
        from numpy import log
        a = log(x)+log(y)
        return a

    f2 = epyccel(log_phrase, language = language)
    x = uniform(low=min_float, high=1e6)
    y = uniform(low=min_float, high=1e6)
    assert(isclose(f2(x,y), log_phrase(x,y), rtol=RTOL, atol=ATOL))

#----------------------------- arcsin function -------------------------------#
def test_arcsin_call_i(language):
    @types('int')
    def arcsin_call_i(x):
        from numpy import arcsin
        return arcsin(x)

    f1 = epyccel(arcsin_call_i, language = language)
    x = randint(2)
    assert(isclose(f1(x), arcsin_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arcsin_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(arcsin_call_i(x).item()))

def test_arcsin_call_r(language):
    @types('real')
    def arcsin_call_r(x):
        from numpy import arcsin
        return arcsin(x)

    f1 = epyccel(arcsin_call_r, language = language)
    x = rand()
    assert(isclose(f1(x), arcsin_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arcsin_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(arcsin_call_r(x).item()))

def test_arcsin_phrase(language):
    @types('real','real')
    def arcsin_phrase(x,y):
        from numpy import arcsin
        a = arcsin(x)+arcsin(y)
        return a

    f2 = epyccel(arcsin_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), arcsin_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), arcsin_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), arcsin_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), arcsin_phrase(x,-y), rtol=RTOL, atol=ATOL))

#----------------------------- arccos function -------------------------------#

def test_arccos_call_i(language):
    @types('int')
    def arccos_call_i(x):
        from numpy import arccos
        return arccos(x)

    f1 = epyccel(arccos_call_i, language = language)
    x = randint(2)
    assert(isclose(f1(x), arccos_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arccos_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(arccos_call_i(x).item()))

def test_arccos_call_r(language):
    @types('real')
    def arccos_call_r(x):
        from numpy import arccos
        return arccos(x)

    f1 = epyccel(arccos_call_r, language = language)
    x = rand()
    assert(isclose(f1(x), arccos_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arccos_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(arccos_call_r(x).item()))

def test_arccos_phrase(language):
    @types('real','real')
    def arccos_phrase(x,y):
        from numpy import arccos
        a = arccos(x)+arccos(y)
        return a

    f2 = epyccel(arccos_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), arccos_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), arccos_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), arccos_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), arccos_phrase(x,-y), rtol=RTOL, atol=ATOL))

#----------------------------- arctan function -------------------------------#
def test_arctan_call_i(language):
    @types('int')
    def arctan_call_i(x):
        from numpy import arctan
        return arctan(x)

    f1 = epyccel(arctan_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), arctan_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arctan_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(arctan_call_i(x).item()))

def test_arctan_call_r(language):
    @types('real')
    def arctan_call_r(x):
        from numpy import arctan
        return arctan(x)

    f1 = epyccel(arctan_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), arctan_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arctan_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(arctan_call_r(x).item()))

def test_arctan_phrase(language):
    @types('real','real')
    def arctan_phrase(x,y):
        from numpy import arctan
        a = arctan(x)+arctan(y)
        return a

    f2 = epyccel(arctan_phrase, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), arctan_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), arctan_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), arctan_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), arctan_phrase(x,-y), rtol=RTOL, atol=ATOL))

#------------------------------- sinh function -------------------------------#
def test_sinh_call_i(language):
    @types('int')
    def sinh_call_i(x):
        from numpy import sinh
        return sinh(x)

    f1 = epyccel(sinh_call_i, language = language)
    x = randint(100)
    assert(isclose(f1(x), sinh_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), sinh_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(sinh_call_i(x).item()))

def test_sinh_call_r(language):
    @types('real')
    def sinh_call_r(x):
        from numpy import sinh
        return sinh(x)

    f1 = epyccel(sinh_call_r, language = language)
    x = uniform(high=1e2)
    assert(isclose(f1(x), sinh_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), sinh_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(sinh_call_r(x).item()))

def test_sinh_phrase(language):
    @types('real','real')
    def sinh_phrase(x,y):
        from numpy import sinh
        a = sinh(x)+sinh(y)
        return a

    f2 = epyccel(sinh_phrase, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), sinh_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sinh_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sinh_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sinh_phrase(x,-y), rtol=RTOL, atol=ATOL))

#------------------------------- sinh function -------------------------------#
def test_cosh_call_i(language):
    @types('int')
    def cosh_call_i(x):
        from numpy import cosh
        return cosh(x)

    f1 = epyccel(cosh_call_i, language = language)
    x = randint(100)
    assert(isclose(f1(x), cosh_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), cosh_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(cosh_call_i(x).item()))

def test_cosh_call_r(language):
    @types('real')
    def cosh_call_r(x):
        from numpy import cosh
        return cosh(x)

    f1 = epyccel(cosh_call_r, language = language)
    x = uniform(high=1e2)
    assert(isclose(f1(x), cosh_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), cosh_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(cosh_call_r(x).item()))

def test_cosh_phrase(language):
    @types('real','real')
    def cosh_phrase(x,y):
        from numpy import cosh
        a = cosh(x)+cosh(y)
        return a

    f2 = epyccel(cosh_phrase, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), cosh_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cosh_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cosh_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cosh_phrase(x,-y), rtol=RTOL, atol=ATOL))

#------------------------------- sinh function -------------------------------#
def test_tanh_call_i(language):
    @types('int')
    def tanh_call_i(x):
        from numpy import tanh
        return tanh(x)

    f1 = epyccel(tanh_call_i, language = language)
    x = randint(100)
    assert(isclose(f1(x), tanh_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), tanh_call_i(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(tanh_call_i(x).item()))

def test_tanh_call_r(language):
    @types('real')
    def tanh_call_r(x):
        from numpy import tanh
        return tanh(x)

    f1 = epyccel(tanh_call_r, language = language)
    x = uniform(high=1e2)
    assert(isclose(f1(x), tanh_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), tanh_call_r(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(tanh_call_r(x).item()))

def test_tanh_phrase(language):
    @types('real','real')
    def tanh_phrase(x,y):
        from numpy import tanh
        a = tanh(x)+tanh(y)
        return a

    f2 = epyccel(tanh_phrase, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), tanh_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tanh_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tanh_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tanh_phrase(x,-y), rtol=RTOL, atol=ATOL))

#------------------------------ arctan2 function -----------------------------#
def test_arctan2_call_i_i(language):
    @types('int','int')
    def arctan2_call(x,y):
        from numpy import arctan2
        return arctan2(x,y)

    f1 = epyccel(arctan2_call, language = language)
    x = randint(100)
    y = randint(100)
    assert(isclose(f1(x,y), arctan2_call(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,-y), arctan2_call(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,y), arctan2_call(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(x,-y), arctan2_call(x,-y), rtol=RTOL, atol=ATOL))
    assert(type(f1(x, y)) == type(arctan2_call(x, y).item()))

def test_arctan2_call_i_r(language):
    @types('int','real')
    def arctan2_call(x,y):
        from numpy import arctan2
        return arctan2(x,y)

    f1 = epyccel(arctan2_call, language = language)
    x = randint(100)
    y = uniform(high=1e2)
    assert(isclose(f1(x,y), arctan2_call(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,-y), arctan2_call(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,y), arctan2_call(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(x,-y), arctan2_call(x,-y), rtol=RTOL, atol=ATOL))
    assert(type(f1(x, y)) == type(arctan2_call(x, y).item()))

def test_arctan2_call_r_i(language):
    @types('real','int')
    def arctan2_call(x,y):
        from numpy import arctan2
        return arctan2(x,y)

    f1 = epyccel(arctan2_call, language = language)
    x = uniform(high=1e2)
    y = randint(100)
    assert(isclose(f1(x,y), arctan2_call(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,-y), arctan2_call(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,y), arctan2_call(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(x,-y), arctan2_call(x,-y), rtol=RTOL, atol=ATOL))
    assert(type(f1(x, y)) == type(arctan2_call(x, y).item()))

def test_arctan2_call_r_r(language):
    @types('real','real')
    def arctan2_call(x,y):
        from numpy import arctan2
        return arctan2(x,y)

    f1 = epyccel(arctan2_call, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f1(x,y), arctan2_call(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,-y), arctan2_call(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,y), arctan2_call(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(x,-y), arctan2_call(x,-y), rtol=RTOL, atol=ATOL))
    assert(type(f1(x, y)) == type(arctan2_call(x, y).item()))

def test_arctan2_phrase(language):
    @types('real','real','real')
    def arctan2_phrase(x,y,z):
        from numpy import arctan2
        a = arctan2(x,y)+arctan2(x,z)
        return a

    f2 = epyccel(arctan2_phrase, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    z = uniform(high=1e2)
    assert(isclose(f2(x,y,z), arctan2_phrase(x,y,z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y,z), arctan2_phrase(-x,y,z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y,z), arctan2_phrase(-x,-y,z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y,-z), arctan2_phrase(-x,y,-z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y,z), arctan2_phrase(x,-y,z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y,-z), arctan2_phrase(x,-y,-z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,y,-z), arctan2_phrase(x,y,-z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y,-z), arctan2_phrase(-x,-y,-z), rtol=RTOL, atol=ATOL))

#-------------------------------- sqrt function ------------------------------#
def test_sqrt_call(language):
    @types('real')
    def sqrt_call(x):
        from numpy import sqrt
        return sqrt(x)

    f1 = epyccel(sqrt_call, language = language)
    x = rand()
    assert(isclose(f1(x), sqrt_call(x), rtol=RTOL, atol=ATOL))

def test_sqrt_phrase(language):
    @types('real','real')
    def sqrt_phrase(x,y):
        from numpy import sqrt
        a = sqrt(x)*sqrt(y)
        return a

    f2 = epyccel(sqrt_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), sqrt_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_sqrt_return_type_r(language):
    @types('real')
    def sqrt_return_type_real(x):
        from numpy import sqrt
        a = sqrt(x)
        return a

    f1 = epyccel(sqrt_return_type_real, language = language)
    x = rand()
    assert(isclose(f1(x), sqrt_return_type_real(x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(sqrt_return_type_real(x).item()))

def test_sqrt_return_type_c(language):
    @types('complex')
    def sqrt_return_type_comp(x):
        from numpy import sqrt
        a = sqrt(x)
        return a

    f1 = epyccel(sqrt_return_type_comp, language = language)
    x = rand() + 1j * rand()
    assert(isclose(f1(x), sqrt_return_type_comp(x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(sqrt_return_type_comp(x).item()))

#-------------------------------- floor function -----------------------------#
def test_floor_call_i(language):
    @types('int')
    def floor_call(x):
        from numpy import floor
        return floor(x)

    f1 = epyccel(floor_call, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), floor_call(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), floor_call(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(floor_call(x).item()))

def test_floor_call_r(language):
    @types('real')
    def floor_call(x):
        from numpy import floor
        return floor(x)

    f1 = epyccel(floor_call, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), floor_call(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), floor_call(-x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x)) == type(floor_call(x).item()))

def test_floor_phrase(language):
    @types('real','real')
    def floor_phrase(x,y):
        from numpy import floor
        a = floor(x)*floor(y)
        return a

    f2 = epyccel(floor_phrase, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), floor_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), floor_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), floor_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), floor_phrase(x,-y), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_shape_indexed(language):
    @types('int[:]')
    def test_shape_1d(f):
        from numpy import shape
        return shape(f)[0]

    @types('int[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_shape_property(language):
    @types('int[:]')
    def test_shape_1d(f):
        return f.shape[0]

    @types('int[:,:]')
    def test_shape_2d(f):
        a = f.shape
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    assert(f1(x1) == test_shape_1d(x1))
    assert(all(isclose(f2(x2), test_shape_2d(x2))))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_shape_tuple_output(language):
    @types('int[:]')
    def test_shape_1d(f):
        from numpy import shape
        s = shape(f)
        return s[0]

    @types('int[:]')
    def test_shape_1d_tuple(f):
        from numpy import shape
        s, = shape(f)
        return s

    @types('int[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a, b = shape(f)
        return a, b

    from numpy import empty
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    f1 = epyccel(test_shape_1d, language = language)
    assert(f1(x1)   == test_shape_1d(x1))
    f1_t = epyccel(test_shape_1d_tuple, language = language)
    assert(f1_t(x1) == test_shape_1d_tuple(x1))
    f2 = epyccel(test_shape_2d, language = language)
    assert(f2(x2)   == test_shape_2d(x2))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_shape_real(language):
    @types('real[:]')
    def test_shape_1d(f):
        from numpy import shape
        b = shape(f)
        return b[0]

    @types('real[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = float)
    x2 = empty((n2,n3), dtype = float)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_shape_int(language):
    @types('int[:]')
    def test_shape_1d(f):
        from numpy import shape
        b = shape(f)
        return b[0]

    @types('int[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)

    from numpy import empty
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_shape_bool(language):
    @types('bool[:]')
    def test_shape_1d(f):
        from numpy import shape
        b = shape(f)
        return b[0]

    @types('bool[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = bool)
    x2 = empty((n2,n3), dtype = bool)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_full_basic_int(language):
    @types('int')
    def create_full_shape_1d(n):
        from numpy import full, shape
        a = full(n,4)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_full_shape_2d(n):
        from numpy import full, shape
        a = full((n,n),4)
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int')
    def create_full_val(val):
        from numpy import full
        a = full(3,val)
        return a[0],a[1],a[2]
    @types('int')
    def create_full_arg_names(val):
        from numpy import full
        a = full(fill_value = val, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)

    f_shape_1d  = epyccel(create_full_shape_1d, language = language)
    assert(f_shape_1d(size) == create_full_shape_1d(size))

    f_shape_2d  = epyccel(create_full_shape_2d, language = language)
    assert(f_shape_2d(size) == create_full_shape_2d(size))

    f_val       = epyccel(create_full_val, language = language)
    assert(f_val(size)      == create_full_val(size))
    assert(type(f_val(size)[0])       == type(create_full_val(size)[0].item()))

    f_arg_names = epyccel(create_full_arg_names, language = language)
    assert(f_arg_names(size) == create_full_arg_names(size))
    assert(type(f_arg_names(size)[0]) == type(create_full_arg_names(size)[0].item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_full_basic_real(language):
    @types('int')
    def create_full_shape_1d(n):
        from numpy import full, shape
        a = full(n,4)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_full_shape_2d(n):
        from numpy import full, shape
        a = full((n,n),4)
        s = shape(a)
        return len(s),s[0], s[1]
    @types('real')
    def create_full_val(val):
        from numpy import full
        a = full(3,val)
        return a[0],a[1],a[2]
    @types('real')
    def create_full_arg_names(val):
        from numpy import full
        a = full(fill_value = val, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)
    val  = rand()*5

    f_shape_1d  = epyccel(create_full_shape_1d, language = language)
    assert(f_shape_1d(size)     == create_full_shape_1d(size))

    f_shape_2d  = epyccel(create_full_shape_2d, language = language)
    assert(f_shape_2d(size)     == create_full_shape_2d(size))

    f_val       = epyccel(create_full_val, language = language)
    assert(f_val(val)           == create_full_val(val))
    assert(type(f_val(val)[0])       == type(create_full_val(val)[0].item()))

    f_arg_names = epyccel(create_full_arg_names, language = language)
    assert(f_arg_names(val)     == create_full_arg_names(val))
    assert(type(f_arg_names(val)[0]) == type(create_full_arg_names(val)[0].item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_full_basic_bool(language):
    @types('int')
    def create_full_shape_1d(n):
        from numpy import full, shape
        a = full(n,4)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_full_shape_2d(n):
        from numpy import full, shape
        a = full((n,n),4)
        s = shape(a)
        return len(s),s[0], s[1]
    @types('bool')
    def create_full_val(val):
        from numpy import full
        a = full(3,val)
        return a[0],a[1],a[2]
    @types('bool')
    def create_full_arg_names(val):
        from numpy import full
        a = full(fill_value = val, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)
    val  = bool(randint(2))

    f_shape_1d  = epyccel(create_full_shape_1d, language = language)
    assert(f_shape_1d(size)     == create_full_shape_1d(size))

    f_shape_2d  = epyccel(create_full_shape_2d, language = language)
    assert(f_shape_2d(size)     == create_full_shape_2d(size))

    f_val       = epyccel(create_full_val, language = language)
    assert(f_val(val)           == create_full_val(val))
    assert(type(f_val(val)[0])       == type(create_full_val(val)[0].item()))

    f_arg_names = epyccel(create_full_arg_names, language = language)
    assert(f_arg_names(val)     == create_full_arg_names(val))
    assert(type(f_arg_names(val)[0]) == type(create_full_arg_names(val)[0].item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_full_order(language):
    @types('int','int')
    def create_full_shape_C(n,m):
        from numpy import full, shape
        a = full((n,m),4, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_full_shape_F(n,m):
        from numpy import full, shape
        a = full((n,m),4, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_full_shape_C, language = language)
    assert(f_shape_C(size_1,size_2) == create_full_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_full_shape_F, language = language)
    assert(f_shape_F(size_1,size_2) == create_full_shape_F(size_1,size_2))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.c]
        )
    )
)
def test_full_dtype(language):
    @types('int')
    def create_full_val_int_int(val):
        from numpy import full
        a = full(3,val,int)
        return a[0]
    @types('int')
    def create_full_val_int_float(val):
        from numpy import full
        a = full(3,val,float)
        return a[0]
    @types('int')
    def create_full_val_int_complex(val):
        from numpy import full
        a = full(3,val,complex)
        return a[0]
    @types('real')
    def create_full_val_real_int32(val):
        from numpy import full, int32
        a = full(3,val,int32)
        return a[0]
    @types('real')
    def create_full_val_real_float32(val):
        from numpy import full, float32
        a = full(3,val,float32)
        return a[0]
    @types('real')
    def create_full_val_real_float64(val):
        from numpy import full, float64
        a = full(3,val,float64)
        return a[0]
    @types('real')
    def create_full_val_real_complex64(val):
        from numpy import full, complex64
        a = full(3,val,complex64)
        return a[0]
    @types('real')
    def create_full_val_real_complex128(val):
        from numpy import full, complex128
        a = full(3,val,complex128)
        return a[0]

    val_int   = randint(100)
    val_float = rand()*100

    f_int_int   = epyccel(create_full_val_int_int, language = language)
    assert(     f_int_int(val_int)        ==      create_full_val_int_int(val_int))
    assert(type(f_int_int(val_int))       == type(create_full_val_int_int(val_int).item()))

    f_int_float = epyccel(create_full_val_int_float, language = language)
    assert(isclose(     f_int_float(val_int)     ,      create_full_val_int_float(val_int), rtol=RTOL, atol=ATOL))
    assert(type(f_int_float(val_int))     == type(create_full_val_int_float(val_int).item()))

    f_int_complex = epyccel(create_full_val_int_complex, language = language)
    assert(isclose(     f_int_complex(val_int)     ,      create_full_val_int_complex(val_int), rtol=RTOL, atol=ATOL))
    assert(type(f_int_complex(val_int))     == type(create_full_val_int_complex(val_int).item()))

    f_real_int32   = epyccel(create_full_val_real_int32, language = language)
    assert(     f_real_int32(val_float)        ==      create_full_val_real_int32(val_float))
    assert(type(f_real_int32(val_float))       == type(create_full_val_real_int32(val_float).item()))

    f_real_float32   = epyccel(create_full_val_real_float32, language = language)
    assert(isclose(     f_real_float32(val_float)       ,      create_full_val_real_float32(val_float), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float32(val_float))       == type(create_full_val_real_float32(val_float).item()))

    f_real_float64   = epyccel(create_full_val_real_float64, language = language)
    assert(isclose(     f_real_float64(val_float)       ,      create_full_val_real_float64(val_float), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float64(val_float))       == type(create_full_val_real_float64(val_float).item()))

    f_real_complex64   = epyccel(create_full_val_real_complex64, language = language)
    assert(isclose(     f_real_complex64(val_float)       ,      create_full_val_real_complex64(val_float), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex64(val_float))       == type(create_full_val_real_complex64(val_float).item()))

    f_real_complex128   = epyccel(create_full_val_real_complex128, language = language)
    assert(isclose(     f_real_complex128(val_float)       ,      create_full_val_real_complex128(val_float), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex128(val_float))       == type(create_full_val_real_complex128(val_float).item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_full_combined_args(language):
    def create_full_1_shape():
        from numpy import full, shape
        a = full((2,1),4.0,int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_1_val():
        from numpy import full
        a = full((2,1),4.0,int,'F')
        return a[0,0]
    def create_full_2_shape():
        from numpy import full, shape
        a = full((4,2),dtype=float,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_2_val():
        from numpy import full
        a = full((4,2),dtype=float,fill_value=1)
        return a[0,0]
    def create_full_3_shape():
        from numpy import full, shape
        a = full(order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_3_val():
        from numpy import full
        a = full(order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        return a[0,0]

    f1_shape = epyccel(create_full_1_shape, language = language)
    f1_val   = epyccel(create_full_1_val, language = language)
    assert(f1_shape() == create_full_1_shape())
    assert(f1_val()   == create_full_1_val()  )
    assert(type(f1_val())  == type(create_full_1_val().item()))

    f2_shape = epyccel(create_full_2_shape, language = language)
    f2_val   = epyccel(create_full_2_val, language = language)
    assert(f2_shape() == create_full_2_shape()    )
    assert(isclose(f2_val()  , create_full_2_val()      , rtol=RTOL, atol=ATOL))
    assert(type(f2_val())  == type(create_full_2_val().item()))

    f3_shape = epyccel(create_full_3_shape, language = language)
    f3_val   = epyccel(create_full_3_val, language = language)
    assert(             f3_shape() ==    create_full_3_shape()      )
    assert(isclose(     f3_val()  ,      create_full_3_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f3_val())  == type(create_full_3_val().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_empty_basic(language):
    @types('int')
    def create_empty_shape_1d(n):
        from numpy import empty, shape
        a = empty(n)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_empty_shape_2d(n):
        from numpy import empty, shape
        a = empty((n,n))
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_empty_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_empty_shape_1d(size))

    f_shape_2d  = epyccel(create_empty_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_empty_shape_2d(size))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_empty_order(language):
    @types('int','int')
    def create_empty_shape_C(n,m):
        from numpy import empty, shape
        a = empty((n,m), order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_empty_shape_F(n,m):
        from numpy import empty, shape
        a = empty((n,m), order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_empty_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_empty_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_empty_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_empty_shape_F(size_1,size_2))

def test_empty_dtype(language):
    def create_empty_val_int():
        from numpy import empty
        a = empty(3,int)
        return a[0]
    def create_empty_val_float():
        from numpy import empty
        a = empty(3,float)
        return a[0]
    def create_empty_val_complex():
        from numpy import empty
        a = empty(3,complex)
        return a[0]
    def create_empty_val_int32():
        from numpy import empty, int32
        a = empty(3,int32)
        return a[0]
    def create_empty_val_float32():
        from numpy import empty, float32
        a = empty(3,float32)
        return a[0]
    def create_empty_val_float64():
        from numpy import empty, float64
        a = empty(3,float64)
        return a[0]
    def create_empty_val_complex64():
        from numpy import empty, complex64
        a = empty(3,complex64)
        return a[0]
    def create_empty_val_complex128():
        from numpy import empty, complex128
        a = empty(3,complex128)
        return a[0]

    f_int_int   = epyccel(create_empty_val_int, language = language)
    assert(type(f_int_int())         == type(create_empty_val_int().item()))

    f_int_float = epyccel(create_empty_val_float, language = language)
    assert(type(f_int_float())       == type(create_empty_val_float().item()))

    f_int_complex = epyccel(create_empty_val_complex, language = language)
    assert(type(f_int_complex())     == type(create_empty_val_complex().item()))

    f_real_int32   = epyccel(create_empty_val_int32, language = language)
    assert(type(f_real_int32())      == type(create_empty_val_int32().item()))

    f_real_float32   = epyccel(create_empty_val_float32, language = language)
    assert(type(f_real_float32())    == type(create_empty_val_float32().item()))

    f_real_float64   = epyccel(create_empty_val_float64, language = language)
    assert(type(f_real_float64())    == type(create_empty_val_float64().item()))

    f_real_complex64   = epyccel(create_empty_val_complex64, language = language)
    assert(type(f_real_complex64())  == type(create_empty_val_complex64().item()))

    f_real_complex128   = epyccel(create_empty_val_complex128, language = language)
    assert(type(f_real_complex128()) == type(create_empty_val_complex128().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_empty_combined_args(language):
    def create_empty_1_shape():
        from numpy import empty, shape
        a = empty((2,1),int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_empty_1_val():
        from numpy import empty
        a = empty((2,1),int,'F')
        return a[0,0]
    def create_empty_2_shape():
        from numpy import empty, shape
        a = empty((4,2),dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_empty_2_val():
        from numpy import empty
        a = empty((4,2),dtype=float)
        return a[0,0]
    def create_empty_3_shape():
        from numpy import empty, shape
        a = empty(order = 'F', shape = (4,2),dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_empty_3_val():
        from numpy import empty
        a = empty(order = 'F', shape = (4,2),dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_empty_1_shape, language = language)
    f1_val   = epyccel(create_empty_1_val, language = language)
    assert(     f1_shape() ==      create_empty_1_shape()      )
    assert(type(f1_val())  == type(create_empty_1_val().item()))

    f2_shape = epyccel(create_empty_2_shape, language = language)
    f2_val   = epyccel(create_empty_2_val, language = language)
    assert(all(isclose(     f2_shape(),      create_empty_2_shape()      )))
    assert(type(f2_val())  == type(create_empty_2_val().item()))

    f3_shape = epyccel(create_empty_3_shape, language = language)
    f3_val   = epyccel(create_empty_3_val, language = language)
    assert(all(isclose(     f3_shape(),      create_empty_3_shape()      )))
    assert(type(f3_val())  == type(create_empty_3_val().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_ones_basic(language):
    @types('int')
    def create_ones_shape_1d(n):
        from numpy import ones, shape
        a = ones(n)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_ones_shape_2d(n):
        from numpy import ones, shape
        a = ones((n,n))
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_ones_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_ones_shape_1d(size))

    f_shape_2d  = epyccel(create_ones_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_ones_shape_2d(size))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_ones_order(language):
    @types('int','int')
    def create_ones_shape_C(n,m):
        from numpy import ones, shape
        a = ones((n,m), order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_ones_shape_F(n,m):
        from numpy import ones, shape
        a = ones((n,m), order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_ones_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_ones_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_ones_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_ones_shape_F(size_1,size_2))

def test_ones_dtype(language):
    def create_ones_val_int():
        from numpy import ones
        a = ones(3,int)
        return a[0]
    def create_ones_val_float():
        from numpy import ones
        a = ones(3,float)
        return a[0]
    def create_ones_val_complex():
        from numpy import ones
        a = ones(3,complex)
        return a[0]
    def create_ones_val_int32():
        from numpy import ones, int32
        a = ones(3,int32)
        return a[0]
    def create_ones_val_float32():
        from numpy import ones, float32
        a = ones(3,float32)
        return a[0]
    def create_ones_val_float64():
        from numpy import ones, float64
        a = ones(3,float64)
        return a[0]
    def create_ones_val_complex64():
        from numpy import ones, complex64
        a = ones(3,complex64)
        return a[0]
    def create_ones_val_complex128():
        from numpy import ones, complex128
        a = ones(3,complex128)
        return a[0]

    f_int_int   = epyccel(create_ones_val_int, language = language)
    assert(     f_int_int()          ==      create_ones_val_int())
    assert(type(f_int_int())         == type(create_ones_val_int().item()))

    f_int_float = epyccel(create_ones_val_float, language = language)
    assert(isclose(     f_int_float()       ,      create_ones_val_float(), rtol=RTOL, atol=ATOL))
    assert(type(f_int_float())       == type(create_ones_val_float().item()))

    f_int_complex = epyccel(create_ones_val_complex, language = language)
    assert(isclose(     f_int_complex()     ,      create_ones_val_complex(), rtol=RTOL, atol=ATOL))
    assert(type(f_int_complex())     == type(create_ones_val_complex().item()))

    f_real_int32   = epyccel(create_ones_val_int32, language = language)
    assert(     f_real_int32()       ==      create_ones_val_int32())
    assert(type(f_real_int32())      == type(create_ones_val_int32().item()))

    f_real_float32   = epyccel(create_ones_val_float32, language = language)
    assert(isclose(     f_real_float32()    ,      create_ones_val_float32(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float32())    == type(create_ones_val_float32().item()))

    f_real_float64   = epyccel(create_ones_val_float64, language = language)
    assert(isclose(     f_real_float64()    ,      create_ones_val_float64(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float64())    == type(create_ones_val_float64().item()))

    f_real_complex64   = epyccel(create_ones_val_complex64, language = language)
    assert(isclose(     f_real_complex64()  ,      create_ones_val_complex64(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex64())  == type(create_ones_val_complex64().item()))

    f_real_complex128   = epyccel(create_ones_val_complex128, language = language)
    assert(isclose(     f_real_complex128() ,      create_ones_val_complex128(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex128()) == type(create_ones_val_complex128().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_ones_combined_args(language):
    def create_ones_1_shape():
        from numpy import ones, shape
        a = ones((2,1),int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_ones_1_val():
        from numpy import ones
        a = ones((2,1),int,'F')
        return a[0,0]
    def create_ones_2_shape():
        from numpy import ones, shape
        a = ones((4,2),dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_ones_2_val():
        from numpy import ones
        a = ones((4,2),dtype=float)
        return a[0,0]
    def create_ones_3_shape():
        from numpy import ones, shape
        a = ones(order = 'F', shape = (4,2),dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_ones_3_val():
        from numpy import ones
        a = ones(order = 'F', shape = (4,2),dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_ones_1_shape, language = language)
    f1_val   = epyccel(create_ones_1_val, language = language)
    assert(     f1_shape() ==      create_ones_1_shape()      )
    assert(     f1_val()   ==      create_ones_1_val()        )
    assert(type(f1_val())  == type(create_ones_1_val().item()))

    f2_shape = epyccel(create_ones_2_shape, language = language)
    f2_val   = epyccel(create_ones_2_val, language = language)
    assert(     f2_shape() ==      create_ones_2_shape()      )
    assert(isclose(     f2_val()  ,      create_ones_2_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f2_val())  == type(create_ones_2_val().item()))

    f3_shape = epyccel(create_ones_3_shape, language = language)
    f3_val   = epyccel(create_ones_3_val, language = language)
    assert(     f3_shape() ==      create_ones_3_shape()      )
    assert(isclose(     f3_val()  ,      create_ones_3_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f3_val())  == type(create_ones_3_val().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_zeros_basic(language):
    @types('int')
    def create_zeros_shape_1d(n):
        from numpy import zeros, shape
        a = zeros(n)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_zeros_shape_2d(n):
        from numpy import zeros, shape
        a = zeros((n,n))
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_zeros_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_zeros_shape_1d(size))

    f_shape_2d  = epyccel(create_zeros_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_zeros_shape_2d(size))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_zeros_order(language):
    @types('int','int')
    def create_zeros_shape_C(n,m):
        from numpy import zeros, shape
        a = zeros((n,m), order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_zeros_shape_F(n,m):
        from numpy import zeros, shape
        a = zeros((n,m), order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_zeros_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_zeros_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_zeros_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_zeros_shape_F(size_1,size_2))

def test_zeros_dtype(language):
    def create_zeros_val_int():
        from numpy import zeros
        a = zeros(3,int)
        return a[0]
    def create_zeros_val_float():
        from numpy import zeros
        a = zeros(3,float)
        return a[0]
    def create_zeros_val_complex():
        from numpy import zeros
        a = zeros(3,complex)
        return a[0]
    def create_zeros_val_int32():
        from numpy import zeros, int32
        a = zeros(3,int32)
        return a[0]
    def create_zeros_val_float32():
        from numpy import zeros, float32
        a = zeros(3,float32)
        return a[0]
    def create_zeros_val_float64():
        from numpy import zeros, float64
        a = zeros(3,float64)
        return a[0]
    def create_zeros_val_complex64():
        from numpy import zeros, complex64
        a = zeros(3,complex64)
        return a[0]
    def create_zeros_val_complex128():
        from numpy import zeros, complex128
        a = zeros(3,complex128)
        return a[0]

    f_int_int   = epyccel(create_zeros_val_int, language = language)
    assert(     f_int_int()          ==      create_zeros_val_int())
    assert(type(f_int_int())         == type(create_zeros_val_int().item()))

    f_int_float = epyccel(create_zeros_val_float, language = language)
    assert(isclose(     f_int_float()       ,      create_zeros_val_float(), rtol=RTOL, atol=ATOL))
    assert(type(f_int_float())       == type(create_zeros_val_float().item()))

    f_int_complex = epyccel(create_zeros_val_complex, language = language)
    assert(isclose(     f_int_complex()     ,      create_zeros_val_complex(), rtol=RTOL, atol=ATOL))
    assert(type(f_int_complex())     == type(create_zeros_val_complex().item()))

    f_real_int32   = epyccel(create_zeros_val_int32, language = language)
    assert(     f_real_int32()       ==      create_zeros_val_int32())
    assert(type(f_real_int32())      == type(create_zeros_val_int32().item()))

    f_real_float32   = epyccel(create_zeros_val_float32, language = language)
    assert(isclose(     f_real_float32()    ,      create_zeros_val_float32(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float32())    == type(create_zeros_val_float32().item()))

    f_real_float64   = epyccel(create_zeros_val_float64, language = language)
    assert(isclose(     f_real_float64()    ,      create_zeros_val_float64(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float64())    == type(create_zeros_val_float64().item()))

    f_real_complex64   = epyccel(create_zeros_val_complex64, language = language)
    assert(isclose(     f_real_complex64()  ,      create_zeros_val_complex64(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex64())  == type(create_zeros_val_complex64().item()))

    f_real_complex128   = epyccel(create_zeros_val_complex128, language = language)
    assert(isclose(     f_real_complex128() ,      create_zeros_val_complex128(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex128()) == type(create_zeros_val_complex128().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_zeros_combined_args(language):
    def create_zeros_1_shape():
        from numpy import zeros, shape
        a = zeros((2,1),int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_zeros_1_val():
        from numpy import zeros
        a = zeros((2,1),int,'F')
        return a[0,0]
    def create_zeros_2_shape():
        from numpy import zeros, shape
        a = zeros((4,2),dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_zeros_2_val():
        from numpy import zeros
        a = zeros((4,2),dtype=float)
        return a[0,0]
    def create_zeros_3_shape():
        from numpy import zeros, shape
        a = zeros(order = 'F', shape = (4,2),dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_zeros_3_val():
        from numpy import zeros
        a = zeros(order = 'F', shape = (4,2),dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_zeros_1_shape, language = language)
    f1_val   = epyccel(create_zeros_1_val, language = language)
    assert(     f1_shape() ==      create_zeros_1_shape()      )
    assert(     f1_val()   ==      create_zeros_1_val()        )
    assert(type(f1_val())  == type(create_zeros_1_val().item()))

    f2_shape = epyccel(create_zeros_2_shape, language = language)
    f2_val   = epyccel(create_zeros_2_val, language = language)
    assert(     f2_shape() ==      create_zeros_2_shape()      )
    assert(isclose(     f2_val()  ,      create_zeros_2_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f2_val())  == type(create_zeros_2_val().item()))

    f3_shape = epyccel(create_zeros_3_shape, language = language)
    f3_val   = epyccel(create_zeros_3_val, language = language)
    assert(     f3_shape() ==      create_zeros_3_shape()      )
    assert(isclose(     f3_val()  ,      create_zeros_3_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f3_val())  == type(create_zeros_3_val().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_array(language):
    def create_array_list_val():
        from numpy import array
        a = array([[1,2,3],[4,5,6]])
        return a[0,0]
    def create_array_list_shape():
        from numpy import array, shape
        a = array([[1,2,3],[4,5,6]])
        s = shape(a)
        return len(s), s[0], s[1]
    def create_array_tuple_val():
        from numpy import array
        a = array(((1,2,3),(4,5,6)))
        return a[0,0]
    def create_array_tuple_shape():
        from numpy import array, shape
        a = array(((1,2,3),(4,5,6)))
        s = shape(a)
        return len(s), s[0], s[1]
    f1_shape = epyccel(create_array_list_shape, language = language)
    f1_val   = epyccel(create_array_list_val, language = language)
    assert(f1_shape() == create_array_list_shape())
    assert(f1_val()   == create_array_list_val())
    assert(type(f1_val()) == type(create_array_list_val().item()))
    f2_shape = epyccel(create_array_tuple_shape, language = language)
    f2_val   = epyccel(create_array_tuple_val, language = language)
    assert(f2_shape() == create_array_tuple_shape())
    assert(f2_val()   == create_array_tuple_val())
    assert(type(f2_val()) == type(create_array_tuple_val().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_rand_basic(language):
    def create_val():
        from numpy.random import rand # pylint: disable=reimported
        return rand()

    f1 = epyccel(create_val, language = language)
    y = [f1() for i in range(10)]
    assert(all([yi <  1 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_rand_args(language):
    @types('int')
    def create_array_size_1d(n):
        from numpy.random import rand # pylint: disable=reimported
        from numpy import shape
        a = rand(n)
        return shape(a)[0]

    @types('int','int')
    def create_array_size_2d(n,m):
        from numpy.random import rand # pylint: disable=reimported
        from numpy import shape
        a = rand(n,m)
        return shape(a)[0], shape(a)[1]

    @types('int','int','int')
    def create_array_size_3d(n,m,p):
        from numpy.random import rand # pylint: disable=reimported
        from numpy import shape
        a = rand(n,m,p)
        return shape(a)[0], shape(a)[1], shape(a)[2]

    def create_array_vals_1d():
        from numpy.random import rand # pylint: disable=reimported
        a = rand(4)
        return a[0], a[1], a[2], a[3]

    def create_array_vals_2d():
        from numpy.random import rand # pylint: disable=reimported
        a = rand(2,2)
        return a[0,0], a[0,1], a[1,0], a[1,1]

    n = randint(10)
    m = randint(10)
    p = randint(5)
    f_1d = epyccel(create_array_size_1d, language = language)
    assert( f_1d(n)       == create_array_size_1d(n)      )

    f_2d = epyccel(create_array_size_2d, language = language)
    assert( f_2d(n, m)    == create_array_size_2d(n, m)   )

    f_3d = epyccel(create_array_size_3d, language = language)
    assert( f_3d(n, m, p) == create_array_size_3d(n, m, p))

    g_1d = epyccel(create_array_vals_1d, language = language)
    y = g_1d()
    assert(all([yi <  1 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

    g_2d = epyccel(create_array_vals_2d, language = language)
    y = g_2d()
    assert(all([yi <  1 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_rand_expr(language):
    def create_val():
        from numpy.random import rand # pylint: disable=reimported
        x = 2*rand()
        return x

    f1 = epyccel(create_val, language = language)
    y = [f1() for i in range(10)]
    assert(all([yi <  2 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.xfail(reason="a is not allocated")
def test_rand_expr_array(language):
    def create_array_vals_2d():
        from numpy.random import rand # pylint: disable=reimported
        a = rand(2,2)*0.5 + 3
        return a[0,0], a[0,1], a[1,0], a[1,1]

    f2 = epyccel(create_array_vals_2d, language = language)
    y = f2()
    assert(all([yi <  3.5 for yi in y]))
    assert(all([yi >= 3   for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="randint not implemented"),
            pytest.mark.c]
        )
    )
)
def test_randint_basic(language):
    def create_rand():
        from numpy.random import randint # pylint: disable=reimported
        return randint(-10, 10)

    @types('int')
    def create_val(high):
        from numpy.random import randint # pylint: disable=reimported
        return randint(high)

    @types('int','int')
    def create_val_low(low, high):
        from numpy.random import randint # pylint: disable=reimported
        return randint(low, high)

    f0 = epyccel(create_rand, language = language)
    y = [f0() for i in range(10)]
    assert(all([yi <  10 for yi in y]))
    assert(all([yi >= -10 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

    f1 = epyccel(create_val, language = language)
    y = [f1(100) for i in range(10)]
    assert(all([yi <  100 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

    f2 = epyccel(create_val_low, language = language)
    y = [f2(25, 100) for i in range(10)]
    assert(all([yi <  100 for yi in y]))
    assert(all([yi >= 25 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="randint not implemented"),
            pytest.mark.c]
        )
    )
)
def test_randint_expr(language):
    @types('int')
    def create_val(high):
        from numpy.random import randint # pylint: disable=reimported
        x = 2*randint(high)
        return x

    @types('int','int')
    def create_val_low(low, high):
        from numpy.random import randint # pylint: disable=reimported
        x = 2*randint(low, high)
        return x

    f1 = epyccel(create_val, language = language)
    y = [f1(27) for i in range(10)]
    assert(all([yi <  54 for yi in y]))
    assert(all([yi >= 0  for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

    f2 = epyccel(create_val_low, language = language)
    y = [f2(21,46) for i in range(10)]
    assert(all([yi <  92 for yi in y]))
    assert(all([yi >= 42 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_sum_int(language):
    @types('int[:]')
    def sum_call(x):
        from numpy import sum as np_sum
        return np_sum(x)

    f1 = epyccel(sum_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == sum_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_sum_real(language):
    @types('real[:]')
    def sum_call(x):
        from numpy import sum as np_sum
        return np_sum(x)

    f1 = epyccel(sum_call, language = language)
    x = rand(10)
    assert(isclose(f1(x), sum_call(x), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_sum_phrase(language):
    @types('real[:]','real[:]')
    def sum_phrase(x,y):
        from numpy import sum as np_sum
        a = np_sum(x)*np_sum(y)
        return a

    f2 = epyccel(sum_phrase, language = language)
    x = rand(10)
    y = rand(15)
    assert(isclose(f2(x,y), sum_phrase(x,y), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_sum_property(language):
    @types('int[:]')
    def sum_call(x):
        return x.sum()

    f1 = epyccel(sum_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == sum_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_min_int(language):
    @types('int[:]')
    def min_call(x):
        from numpy import min as np_min
        return np_min(x)

    f1 = epyccel(min_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == min_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_min_real(language):
    @types('real[:]')
    def min_call(x):
        from numpy import min as np_min
        return np_min(x)

    f1 = epyccel(min_call, language = language)
    x = rand(10)
    assert(isclose(f1(x), min_call(x), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_min_phrase(language):
    @types('real[:]','real[:]')
    def min_phrase(x,y):
        from numpy import min as np_min
        a = np_min(x)*np_min(y)
        return a

    f2 = epyccel(min_phrase, language = language)
    x = rand(10)
    y = rand(15)
    assert(isclose(f2(x,y), min_phrase(x,y), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_min_property(language):
    @types('int[:]')
    def min_call(x):
        return x.min()

    f1 = epyccel(min_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == min_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_max_int(language):
    @types('int[:]')
    def max_call(x):
        from numpy import max as np_max
        return np_max(x)

    f1 = epyccel(max_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == max_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_max_real(language):
    @types('real[:]')
    def max_call(x):
        from numpy import max as np_max
        return np_max(x)

    f1 = epyccel(max_call, language = language)
    x = rand(10)
    assert(isclose(f1(x), max_call(x), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_max_phrase(language):
    @types('real[:]','real[:]')
    def max_phrase(x,y):
        from numpy import max as np_max
        a = np_max(x)*np_max(y)
        return a

    f2 = epyccel(max_phrase, language = language)
    x = rand(10)
    y = rand(15)
    assert(isclose(f2(x,y), max_phrase(x,y), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_max_property(language):
    @types('int[:]')
    def max_call(x):
        return x.max()

    f1 = epyccel(max_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == max_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)

def test_full_like_basic_int(language):
    @types('int')
    def create_full_like_shape_1d(n):
        from numpy import full_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, n, int, 'F')
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_full_like_shape_2d(n):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, n, int , 'F')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int')
    def create_full_like_val(val):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, val, int, 'F')
        return a[0],a[1],a[2]
    @types('int')
    def create_full_like_arg_names(val):
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, val, int, 'F', shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)

    f_shape_1d  = epyccel(create_full_like_shape_1d, language = language)
    assert(f_shape_1d(size) == create_full_like_shape_1d(size))

    f_shape_2d  = epyccel(create_full_like_shape_2d, language = language)
    assert(f_shape_2d(size) == create_full_like_shape_2d(size))

    f_val       = epyccel(create_full_like_val, language = language)
    assert(f_val(size)      == create_full_like_val(size))
    assert(type(f_val(size)[0])       == type(create_full_like_val(size)[0].item()))

    f_arg_names = epyccel(create_full_like_arg_names, language = language)
    assert(f_arg_names(size) == create_full_like_arg_names(size))
    assert(type(f_arg_names(size)[0]) == type(create_full_like_arg_names(size)[0].item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_full_like_basic_real(language):
    @types('real')
    def create_full_like_shape_1d(n):
        from numpy import full_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, n, float, 'F')
        s = shape(a)
        return len(s),s[0]
    @types('real')
    def create_full_like_shape_2d(n):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, n, float, 'F')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('real')
    def create_full_like_val(val):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, val, float, 'F')
        return a[0],a[1],a[2]
    @types('real')
    def create_full_like_arg_names(val):
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, val, float, 'F', shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)
    val  = rand()*5

    f_shape_1d  = epyccel(create_full_like_shape_1d, language = language)
    assert(f_shape_1d(size)     == create_full_like_shape_1d(size))

    f_shape_2d  = epyccel(create_full_like_shape_2d, language = language)
    assert(f_shape_2d(size)     == create_full_like_shape_2d(size))

    f_val       = epyccel(create_full_like_val, language = language)
    assert(f_val(val)           == create_full_like_val(val))
    assert(type(f_val(val)[0])       == type(create_full_like_val(val)[0].item()))

    f_arg_names = epyccel(create_full_like_arg_names, language = language)
    assert(f_arg_names(val)     == create_full_like_arg_names(val))
    assert(type(f_arg_names(val)[0]) == type(create_full_like_arg_names(val)[0].item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented"),
            pytest.mark.c]
        )
    )
)
def test_full_like_basic_bool(language):
    @types('int')
    def create_full_like_shape_1d(n):
        from numpy import full_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, n, int, 'F')
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_full_like_shape_2d(n):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, n, int, 'F')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('bool')
    def create_full_like_val(val):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr , 3, bool, 'F')
        return a[0],a[1],a[2]
    @types('bool')
    def create_full_like_arg_names(val):
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr ,fill_value = val, dtype=bool, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)
    val  = bool(randint(2))

    f_shape_1d  = epyccel(create_full_like_shape_1d, language = language)
    assert(f_shape_1d(size)     == create_full_like_shape_1d(size))

    f_shape_2d  = epyccel(create_full_like_shape_2d, language = language)
    assert(f_shape_2d(size)     == create_full_like_shape_2d(size))

    f_val       = epyccel(create_full_like_val, language = language)
    assert(f_val(val)           == create_full_like_val(val))
    assert(type(f_val(val)[0])       == type(create_full_like_val(val)[0].item()))

    f_arg_names = epyccel(create_full_like_arg_names, language = language)
    assert(f_arg_names(val)     == create_full_like_arg_names(val))
    assert(type(f_arg_names(val)[0]) == type(create_full_like_arg_names(val)[0].item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_full_like_order(language):
    @types('int')
    def create_full_like_shape_C(n):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,4, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int')
    def create_full_like_shape_F(n):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,4, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_C  = epyccel(create_full_like_shape_C, language = language)
    assert(f_shape_C(size) == create_full_like_shape_C(size))

    f_shape_F  = epyccel(create_full_like_shape_F, language = language)
    assert(f_shape_F(size) == create_full_like_shape_F(size))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.c]
        )
    )
)
def test_full_like_dtype(language):
    @types('int')
    def create_full_like_val_int_int(val):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,int)
        return a[0]
    @types('int')
    def create_full_like_val_int_float(val):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,float)
        return a[0]
    @types('int')
    def create_full_like_val_int_complex(val):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,complex)
        return a[0]
    @types('real')
    def create_full_like_val_real_int32(val):
        from numpy import full_like, int32, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,int32)
        return a[0]
    @types('real')
    def create_full_like_val_real_float32(val):
        from numpy import full_like, float32, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,float32)
        return a[0]
    @types('real')
    def create_full_like_val_real_float64(val):
        from numpy import full_like, float64, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,float64)
        return a[0]
    @types('real')
    def create_full_like_val_real_complex64(val):
        from numpy import full_like, complex64, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,complex64)
        return a[0]
    @types('real')
    def create_full_like_val_real_complex128(val):
        from numpy import full_like, complex128, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,complex128)
        return a[0]

    val_int   = randint(100)
    val_float = rand()*100

    f_int_int   = epyccel(create_full_like_val_int_int, language = language)
    assert(     f_int_int(val_int)        ==      create_full_like_val_int_int(val_int))
    assert(type(f_int_int(val_int))       == type(create_full_like_val_int_int(val_int).item()))

    f_int_float = epyccel(create_full_like_val_int_float, language = language)
    assert(isclose(     f_int_float(val_int)     ,      create_full_like_val_int_float(val_int), rtol=RTOL, atol=ATOL))
    assert(type(f_int_float(val_int))     == type(create_full_like_val_int_float(val_int).item()))

    f_int_complex = epyccel(create_full_like_val_int_complex, language = language)
    assert(isclose(     f_int_complex(val_int)     ,      create_full_like_val_int_complex(val_int), rtol=RTOL, atol=ATOL))
    assert(type(f_int_complex(val_int))     == type(create_full_like_val_int_complex(val_int).item()))

    f_real_int32   = epyccel(create_full_like_val_real_int32, language = language)
    assert(     f_real_int32(val_float)        ==      create_full_like_val_real_int32(val_float))
    assert(type(f_real_int32(val_float))       == type(create_full_like_val_real_int32(val_float).item()))

    f_real_float32   = epyccel(create_full_like_val_real_float32, language = language)
    assert(isclose(     f_real_float32(val_float)       ,      create_full_like_val_real_float32(val_float), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float32(val_float))       == type(create_full_like_val_real_float32(val_float).item()))

    f_real_float64   = epyccel(create_full_like_val_real_float64, language = language)
    assert(isclose(     f_real_float64(val_float)       ,      create_full_like_val_real_float64(val_float), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float64(val_float))       == type(create_full_like_val_real_float64(val_float).item()))

    f_real_complex64   = epyccel(create_full_like_val_real_complex64, language = language)
    assert(isclose(     f_real_complex64(val_float)       ,      create_full_like_val_real_complex64(val_float), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex64(val_float))       == type(create_full_like_val_real_complex64(val_float).item()))

    f_real_complex128   = epyccel(create_full_like_val_real_complex128, language = language)
    assert(isclose(     f_real_complex128(val_float)       ,      create_full_like_val_real_complex128(val_float), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex128(val_float))       == type(create_full_like_val_real_complex128(val_float).item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_full_like_combined_args(language):
    def create_full_like_1_shape():
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,5,int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_like_1_val():
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, 4.0, int,'F')
        return a[0,0]
    def create_full_like_2_shape():
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,dtype=float,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_like_2_val():
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,dtype=float,fill_value=1)
        return a[0,0]
    def create_full_like_3_shape():
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_like_3_val():
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        return a[0,0]


    f1_shape = epyccel(create_full_like_1_shape, language = language)
    f1_val   = epyccel(create_full_like_1_val, language = language)
    assert(f1_shape() == create_full_like_1_shape())
    assert(f1_val()   == create_full_like_1_val()  )
    assert(type(f1_val())  == type(create_full_like_1_val().item()))

    f2_shape = epyccel(create_full_like_2_shape, language = language)
    f2_val   = epyccel(create_full_like_2_val, language = language)
    assert(f2_shape() == create_full_like_2_shape()    )
    assert(isclose(f2_val()  , create_full_like_2_val()      , rtol=RTOL, atol=ATOL))
    assert(type(f2_val())  == type(create_full_like_2_val().item()))

    f3_shape = epyccel(create_full_like_3_shape, language = language)
    f3_val   = epyccel(create_full_like_3_val, language = language)
    assert(             f3_shape() ==    create_full_like_3_shape()      )
    assert(isclose(     f3_val()  ,      create_full_like_3_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f3_val())  == type(create_full_like_3_val().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_empty_like_basic(language):
    @types('int')
    def create_empty_like_shape_1d(n):
        from numpy import empty_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr,int)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_empty_like_shape_2d(n):
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr,int)
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_empty_like_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_empty_like_shape_1d(size))

    f_shape_2d  = epyccel(create_empty_like_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_empty_like_shape_2d(size))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_empty_like_order(language):
    @types('int','int')
    def create_empty_like_shape_C(n,m):
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, int, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int', 'int')
    def create_empty_like_shape_F(n,m):
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, int, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_empty_like_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_empty_like_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_empty_like_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_empty_like_shape_F(size_1,size_2))

def test_empty_like_dtype(language):

    def create_empty_like_val_int():
        from numpy import empty_like, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, int)
        return a[0]

    def create_empty_like_val_float():
        from numpy import empty_like, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, dtype=float)
        return a[0]

    def create_empty_like_val_complex():
        from numpy import empty_like, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, dtype=complex)
        return a[0]

    def create_empty_like_val_int32():
        from numpy import empty_like, int32, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, dtype=int32)
        return a[0]

    def create_empty_like_val_float32():
        from numpy import empty_like, float32, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, dtype=float32)
        return a[0]

    def create_empty_like_val_float64():
        from numpy import empty_like, float64, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr,dtype=float64)
        return a[0]

    def create_empty_like_val_complex64():
        from numpy import empty_like, complex64, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr,dtype=complex64)
        return a[0]

    def create_empty_like_val_complex128():
        from numpy import empty_like, complex128, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr,dtype=complex128)
        return a[0]


    f_int_int   = epyccel(create_empty_like_val_int, language = language)
    assert(type(f_int_int())         == type(create_empty_like_val_int().item()))

    f_int_float = epyccel(create_empty_like_val_float, language = language)
    assert(type(f_int_float())       == type(create_empty_like_val_float().item()))

    f_int_complex = epyccel(create_empty_like_val_complex, language = language)
    assert(type(f_int_complex())     == type(create_empty_like_val_complex().item()))

    f_real_int32   = epyccel(create_empty_like_val_int32, language = language)
    assert(type(f_real_int32())      == type(create_empty_like_val_int32().item()))

    f_real_float32   = epyccel(create_empty_like_val_float32, language = language)
    assert(type(f_real_float32())    == type(create_empty_like_val_float32().item()))

    f_real_float64   = epyccel(create_empty_like_val_float64, language = language)
    assert(type(f_real_float64())    == type(create_empty_like_val_float64().item()))

    f_real_complex64   = epyccel(create_empty_like_val_complex64, language = language)
    assert(type(f_real_complex64())  == type(create_empty_like_val_complex64().item()))

    f_real_complex128   = epyccel(create_empty_like_val_complex128, language = language)
    assert(type(f_real_complex128()) == type(create_empty_like_val_complex128().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_empty_like_combined_args(language):

    def create_empty_like_1_shape():
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr,dtype=int,order='F')
        s = shape(a)
        return len(s),s[0],s[1]

    def create_empty_like_1_val():
        from numpy import empty_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, dtype=int,order='F')
        return a[0,0]

    def create_empty_like_2_shape():
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_empty_like_2_val():
        from numpy import empty_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, dtype=float)
        return a[0,0]

    def create_empty_like_3_shape():
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr,shape = (4,2), order = 'F',dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_empty_like_3_val():
        from numpy import empty_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, shape = (4,2),order = 'F',dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_empty_like_1_shape, language = language)
    f1_val   = epyccel(create_empty_like_1_val, language = language)
    assert(     f1_shape() ==      create_empty_like_1_shape()      )
    assert(type(f1_val())  == type(create_empty_like_1_val().item()))

    f2_shape = epyccel(create_empty_like_2_shape, language = language)
    f2_val   = epyccel(create_empty_like_2_val, language = language)
    assert(all(isclose(     f2_shape(),      create_empty_like_2_shape()      )))
    assert(type(f2_val())  == type(create_empty_like_2_val().item()))

    f3_shape = epyccel(create_empty_like_3_shape, language = language)
    f3_val   = epyccel(create_empty_like_3_val, language = language)
    assert(all(isclose(     f3_shape(),      create_empty_like_3_shape()      )))
    assert(type(f3_val())  == type(create_empty_like_3_val().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_ones_like_basic(language):
    @types('int')
    def create_ones_like_shape_1d(n):
        from numpy import ones_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_ones_like_shape_2d(n):
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr)
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_ones_like_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_ones_like_shape_1d(size))

    f_shape_2d  = epyccel(create_ones_like_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_ones_like_shape_2d(size))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_ones_like_order(language):
    @types('int','int')
    def create_ones_like_shape_C(n,m):
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_ones_like_shape_F(n,m):
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_ones_like_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_ones_like_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_ones_like_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_ones_like_shape_F(size_1,size_2))

def test_ones_like_dtype(language):

    def create_ones_like_val_int():
        from numpy import ones_like, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, int)
        return a[0]

    def create_ones_like_val_float():
        from numpy import ones_like, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr,float)
        return a[0]

    def create_ones_like_val_complex():
        from numpy import ones_like, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, complex)
        return a[0]

    def create_ones_like_val_int32():
        from numpy import ones_like, int32, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr,int32)
        return a[0]

    def create_ones_like_val_float32():
        from numpy import ones_like, float32, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, float32)
        return a[0]

    def create_ones_like_val_float64():
        from numpy import ones_like, float64, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, float64)
        return a[0]

    def create_ones_like_val_complex64():
        from numpy import ones_like, complex64, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, complex64)
        return a[0]

    def create_ones_like_val_complex128():
        from numpy import ones_like, complex128, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, complex128)
        return a[0]


    f_int_int   = epyccel(create_ones_like_val_int, language = language)
    assert(     f_int_int()          ==      create_ones_like_val_int())
    assert(type(f_int_int())         == type(create_ones_like_val_int().item()))

    f_int_float = epyccel(create_ones_like_val_float, language = language)
    assert(isclose(     f_int_float()       ,      create_ones_like_val_float(), rtol=RTOL, atol=ATOL))
    assert(type(f_int_float())       == type(create_ones_like_val_float().item()))

    f_int_complex = epyccel(create_ones_like_val_complex, language = language)
    assert(isclose(     f_int_complex()     ,      create_ones_like_val_complex(), rtol=RTOL, atol=ATOL))
    assert(type(f_int_complex())     == type(create_ones_like_val_complex().item()))

    f_real_int32   = epyccel(create_ones_like_val_int32, language = language)
    assert(     f_real_int32()       ==      create_ones_like_val_int32())
    assert(type(f_real_int32())      == type(create_ones_like_val_int32().item()))

    f_real_float32   = epyccel(create_ones_like_val_float32, language = language)
    assert(isclose(     f_real_float32()    ,      create_ones_like_val_float32(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float32())    == type(create_ones_like_val_float32().item()))

    f_real_float64   = epyccel(create_ones_like_val_float64, language = language)
    assert(isclose(     f_real_float64()    ,      create_ones_like_val_float64(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float64())    == type(create_ones_like_val_float64().item()))

    f_real_complex64   = epyccel(create_ones_like_val_complex64, language = language)
    assert(isclose(     f_real_complex64()  ,      create_ones_like_val_complex64(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex64())  == type(create_ones_like_val_complex64().item()))

    f_real_complex128   = epyccel(create_ones_like_val_complex128, language = language)
    assert(isclose(     f_real_complex128() ,      create_ones_like_val_complex128(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex128()) == type(create_ones_like_val_complex128().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_ones_like_combined_args(language):

    def create_ones_like_1_shape():
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,int,'F')
        s = shape(a)
        return len(s),s[0],s[1]

    def create_ones_like_1_val():
        from numpy import ones_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,int,'F')
        return a[0,0]

    def create_ones_like_2_shape():
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_ones_like_2_val():
        from numpy import ones_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,dtype=float)
        return a[0,0]

    def create_ones_like_3_shape():
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,shape = (4,2),order = 'F',dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_ones_like_3_val():
        from numpy import ones_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,shape = (4,2),order = 'F',dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_ones_like_1_shape, language = language)
    f1_val   = epyccel(create_ones_like_1_val, language = language)
    assert(     f1_shape() ==      create_ones_like_1_shape()      )
    assert(     f1_val()   ==      create_ones_like_1_val()        )
    assert(type(f1_val())  == type(create_ones_like_1_val().item()))

    f2_shape = epyccel(create_ones_like_2_shape, language = language)
    f2_val   = epyccel(create_ones_like_2_val, language = language)
    assert(     f2_shape() ==      create_ones_like_2_shape()      )
    assert(isclose(     f2_val()  ,      create_ones_like_2_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f2_val())  == type(create_ones_like_2_val().item()))

    f3_shape = epyccel(create_ones_like_3_shape, language = language)
    f3_val   = epyccel(create_ones_like_3_val, language = language)
    assert(     f3_shape() ==      create_ones_like_3_shape()      )
    assert(isclose(     f3_val()  ,      create_ones_like_3_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f3_val())  == type(create_ones_like_3_val().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_zeros_like_basic(language):
    @types('int')
    def create_zeros_like_shape_1d(n):
        from numpy import zeros_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr, int)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_zeros_like_shape_2d(n):
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr,int)
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_zeros_like_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_zeros_like_shape_1d(size))

    f_shape_2d  = epyccel(create_zeros_like_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_zeros_like_shape_2d(size))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_zeros_like_order(language):
    @types('int','int')
    def create_zeros_like_shape_C(n,m):
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_zeros_like_shape_F(n,m):
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_zeros_like_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_zeros_like_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_zeros_like_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_zeros_like_shape_F(size_1,size_2))

def test_zeros_like_dtype(language):

    def create_zeros_like_val_int():
        from numpy import zeros_like, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,int)
        return a[0]

    def create_zeros_like_val_float():
        from numpy import zeros_like, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,float)
        return a[0]

    def create_zeros_like_val_complex():
        from numpy import zeros_like, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,complex)
        return a[0]

    def create_zeros_like_val_int32():
        from numpy import zeros_like, int32, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,int32)
        return a[0]

    def create_zeros_like_val_float32():
        from numpy import zeros_like, float32, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,float32)
        return a[0]

    def create_zeros_like_val_float64():
        from numpy import zeros_like, float64, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,float64)
        return a[0]

    def create_zeros_like_val_complex64():
        from numpy import zeros_like, complex64, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,complex64)
        return a[0]

    def create_zeros_like_val_complex128():
        from numpy import zeros_like, complex128, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,complex128)
        return a[0]

    f_int_int   = epyccel(create_zeros_like_val_int, language = language)
    assert(     f_int_int()          ==      create_zeros_like_val_int())
    assert(type(f_int_int())         == type(create_zeros_like_val_int().item()))

    f_int_float = epyccel(create_zeros_like_val_float, language = language)
    assert(isclose(     f_int_float()       ,      create_zeros_like_val_float(), rtol=RTOL, atol=ATOL))
    assert(type(f_int_float())       == type(create_zeros_like_val_float().item()))

    f_int_complex = epyccel(create_zeros_like_val_complex, language = language)
    assert(isclose(     f_int_complex()     ,      create_zeros_like_val_complex(), rtol=RTOL, atol=ATOL))
    assert(type(f_int_complex())     == type(create_zeros_like_val_complex().item()))

    f_real_int32   = epyccel(create_zeros_like_val_int32, language = language)
    assert(     f_real_int32()       ==      create_zeros_like_val_int32())
    assert(type(f_real_int32())      == type(create_zeros_like_val_int32().item()))

    f_real_float32   = epyccel(create_zeros_like_val_float32, language = language)
    assert(isclose(     f_real_float32()    ,      create_zeros_like_val_float32(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float32())    == type(create_zeros_like_val_float32().item()))

    f_real_float64   = epyccel(create_zeros_like_val_float64, language = language)
    assert(isclose(     f_real_float64()    ,      create_zeros_like_val_float64(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_float64())    == type(create_zeros_like_val_float64().item()))

    f_real_complex64   = epyccel(create_zeros_like_val_complex64, language = language)
    assert(isclose(     f_real_complex64()  ,      create_zeros_like_val_complex64(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex64())  == type(create_zeros_like_val_complex64().item()))

    f_real_complex128   = epyccel(create_zeros_like_val_complex128, language = language)
    assert(isclose(     f_real_complex128() ,      create_zeros_like_val_complex128(), rtol=RTOL, atol=ATOL))
    assert(type(f_real_complex128()) == type(create_zeros_like_val_complex128().item()))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented yet"),
            pytest.mark.c]
        )
    )
)
def test_zeros_like_combined_args(language):

    def create_zeros_like_1_shape():
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr,int,'F')
        s = shape(a)
        return len(s),s[0],s[1]

    def create_zeros_like_1_val():
        from numpy import zeros_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, int,'F')
        return a[0,0]

    def create_zeros_like_2_shape():
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_zeros_like_2_val():
        from numpy import zeros_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, dtype=float)
        return a[0,0]

    def create_zeros_like_3_shape():
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, shape = (4,2), order = 'F',dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_zeros_like_3_val():
        from numpy import zeros_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, shape = (4,2), order = 'F',dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_zeros_like_1_shape, language = language)
    f1_val   = epyccel(create_zeros_like_1_val, language = language)
    assert(     f1_shape() ==      create_zeros_like_1_shape()      )
    assert(     f1_val()   ==      create_zeros_like_1_val()        )
    assert(type(f1_val())  == type(create_zeros_like_1_val().item()))

    f2_shape = epyccel(create_zeros_like_2_shape, language = language)
    f2_val   = epyccel(create_zeros_like_2_val, language = language)
    assert(     f2_shape() ==      create_zeros_like_2_shape()      )
    assert(isclose(     f2_val()  ,      create_zeros_like_2_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f2_val())  == type(create_zeros_like_2_val().item()))

    f3_shape = epyccel(create_zeros_like_3_shape, language = language)
    f3_val   = epyccel(create_zeros_like_3_val, language = language)
    assert(     f3_shape() ==      create_zeros_like_3_shape()      )
    assert(isclose(     f3_val()  ,      create_zeros_like_3_val()        , rtol=RTOL, atol=ATOL))
    assert(type(f3_val())  == type(create_zeros_like_3_val().item()))

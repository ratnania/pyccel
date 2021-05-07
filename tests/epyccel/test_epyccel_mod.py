# pylint: disable=missing-function-docstring, missing-module-docstring/
import sys
from numpy.random import randint, uniform
from numpy import isclose

from pyccel.decorators import types
from pyccel.epyccel import epyccel

# Relative and absolute tolerances for array comparisons in the form
# numpy.isclose(a, b, rtol, atol). Windows has larger round-off errors.
if sys.platform == 'win32':
    RTOL = 1e-13
    ATOL = 1e-14
else:
    RTOL = 1e-14
    ATOL = 1e-15

def test_modulo_int_int(language):
    @types(int, int)
    def modulo_i_i(x, y):
        return x % y

    f = epyccel(modulo_i_i, language=language)
    x = randint(0, 1e6)
    nx = randint(-1e6, -1)
    y = randint(low=-100, high=100)
    y = 1 if y == 0 else y


    f_output = f(x, y)
    modulo_i_i_output = modulo_i_i(x, y)
    assert modulo_i_i_output == f_output
    assert isinstance(f_output, type(modulo_i_i_output))

    #test negative x
    f_output = f(nx, y)
    modulo_i_i_output = modulo_i_i(nx, y)
    assert modulo_i_i_output == f_output
    assert isinstance(f_output, type(modulo_i_i_output))

def test_modulo_real_real(language):
    @types('real', 'real')
    def modulo_r_r(x, y):
        return x % y

    f = epyccel(modulo_r_r, language=language)
    x = uniform(low=0, high=1e6)
    nx = uniform(low=-1e6, high=-1)
    y = uniform(low=-1e2, high=1e2)
    y = 1 if y == 0 else y

    f_output = f(x, y)
    modulo_r_r_output = modulo_r_r(x, y)
    assert(isclose(f_output, modulo_r_r_output, rtol=RTOL, atol=ATOL))
    assert isinstance(f_output, type(modulo_r_r_output))

    # test negative x
    f_output = f(nx, y)
    modulo_r_r_output = modulo_r_r(nx, y)
    assert(isclose(f_output, modulo_r_r_output, rtol=RTOL, atol=ATOL))
    assert isinstance(f_output, type(modulo_r_r_output))

def test_modulo_real_int(language):
    @types('real', 'int')
    def modulo_r_i(x, y):
        return x % y

    f = epyccel(modulo_r_i, language=language)
    x = uniform(low=0, high=1e6)
    nx = uniform(low=-1e6, high=-1)
    y = randint(low=-100, high=100)
    y = 1 if y == 0 else y


    f_output = f(x, y)
    modulo_r_i_output = modulo_r_i(x, y)
    assert(isclose(f_output, modulo_r_i_output, rtol=RTOL, atol=ATOL))
    assert isinstance(f_output, type(modulo_r_i_output))

    # test negative x
    f_output = f(nx, y)
    modulo_r_i_output = modulo_r_i(nx, y)
    assert(isclose(f_output, modulo_r_i_output, rtol=RTOL, atol=ATOL))
    assert isinstance(f_output, type(modulo_r_i_output))


def test_modulo_int_real(language):
    @types('int', 'real')
    def modulo_i_r(x, y):
        return x % y

    f = epyccel(modulo_i_r, language=language)
    x = randint(0, 1e6)
    nx = randint(-1e6, -1)
    y = uniform(low=-1e2, high=1e2)
    y = 1 if y == 0 else y

    f_output = f(x, y)
    modulo_i_r_output = modulo_i_r(x, y)
    assert(isclose(f_output, modulo_i_r_output, rtol=RTOL, atol=ATOL))
    assert isinstance(f_output, type(modulo_i_r_output))

    # test negative x
    f_output = f(nx, y)
    modulo_i_r_output = modulo_i_r(nx, y)
    assert(isclose(f_output, modulo_i_r_output, rtol=RTOL, atol=ATOL))
    assert isinstance(f_output, type(modulo_i_r_output))

def test_modulo_multiple(language):
    @types('int', 'real', 'int')
    def modulo_multiple(x, y, z):
        return x % y % z

    f = epyccel(modulo_multiple, language=language)
    x = randint(1e6)
    y = uniform(low=1e2, high=1e4)
    z = randint(low=1, high=1e2)

    assert(isclose(f(x, y, z), modulo_multiple(x, y, z), rtol=RTOL, atol=ATOL))
    assert isinstance(f(x, y, z), type(modulo_multiple(x, y, z)))

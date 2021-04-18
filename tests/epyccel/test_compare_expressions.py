# pylint: disable=missing-function-docstring, missing-module-docstring/
from utilities import epyccel_test
from pyccel.decorators import types

#==============================================================================
@types('int, int, int')
def mod_eq_pow(a, m, n):
    return a%m == n**2

@types('int, int, int')
def mod_neq_pow(a, m, n):
    return a%m != n**2

@types('int, int, int')
def idiv_gt_add(a, m, n):
    return a//m > n+1

#==============================================================================
def test_mod_eq_pow(language):
    test = epyccel_test(mod_eq_pow, lang=language)
    # True
    test.compare_epyccel(10, 3, 1)
    test.compare_epyccel(19, 10, 3)
    test.compare_epyccel(21, 3, 0)
    # False
    test.compare_epyccel(10, 5, 2)
    test.compare_epyccel(19, 10, 1)
    test.compare_epyccel(21, 3, 1)

def test_mod_neq_pow(language):
    test = epyccel_test(mod_neq_pow, lang=language)
    # True
    test.compare_epyccel(10, 5, 2)
    test.compare_epyccel(19, 10, 1)
    test.compare_epyccel(21, 3, 1)
    # False
    test.compare_epyccel(10, 3, 1)
    test.compare_epyccel(19, 10, 3)
    test.compare_epyccel(21, 3, 0)

def test_idiv_gt_add(language):
    test = epyccel_test(idiv_gt_add, lang=language)
    # True
    test.compare_epyccel(10, 3, 2)
    test.compare_epyccel(8, 2, 2)
    test.compare_epyccel(16, 3, 4)
    # False
    test.compare_epyccel(10, 3, 2)
    test.compare_epyccel(8, 2, 3)
    test.compare_epyccel(16, 3, 5)

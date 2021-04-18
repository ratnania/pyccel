# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest

from modules import base
from utilities import epyccel_test


def test_is_false(language):
    test = epyccel_test(base.is_false, lang=language)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_is_true(language):
    test = epyccel_test(base.is_true, lang=language)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_compare_is(language):
    test = epyccel_test(base.compare_is, lang=language)
    test.compare_epyccel( True, True )
    test.compare_epyccel( True, False )
    test.compare_epyccel( False, True )
    test.compare_epyccel( False, False )

def test_compare_is_not(language):
    test = epyccel_test(base.compare_is_not, lang=language)
    test.compare_epyccel( True, True )
    test.compare_epyccel( True, False )
    test.compare_epyccel( False, True )
    test.compare_epyccel( False, False )

def test_compare_is_int(language):
    test = epyccel_test(base.compare_is_int, lang=language)
    test.compare_epyccel( True, 1 )
    test.compare_epyccel( True, 0 )
    test.compare_epyccel( False, 1 )
    test.compare_epyccel( False, 0 )

def test_compare_is_not_int(language):
    test = epyccel_test(base.compare_is_not_int, lang=language)
    test.compare_epyccel( True, 1 )
    test.compare_epyccel( True, 0 )
    test.compare_epyccel( False, 1 )
    test.compare_epyccel( False, 0 )

def test_not_false(language):
    test = epyccel_test(base.not_false, lang=language)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_not_true(language):
    test = epyccel_test(base.not_true, lang=language)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_eq_false(language):
    test = epyccel_test(base.eq_false, lang=language)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_eq_true(language):
    test = epyccel_test(base.eq_true, lang=language)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_neq_false(language):
    test = epyccel_test(base.eq_false, lang=language)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_neq_true(language):
    test = epyccel_test(base.eq_true, lang=language)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_not(language):
    test = epyccel_test(base.not_val, lang=language)
    test.compare_epyccel( True )
    test.compare_epyccel( False )

def test_not_int(language):
    test = epyccel_test(base.not_int, lang=language)
    test.compare_epyccel( 0 )
    test.compare_epyccel( 4 )

def test_compare_is_nil(language):
    test = epyccel_test(base.is_nil, lang=language)
    test.compare_epyccel( None )

def test_compare_is_not_nil(language):
    test = epyccel_test(base.is_not_nil, lang=language)
    test.compare_epyccel( None )

def test_cast_int(language):
    test = epyccel_test(base.cast_int, lang=language)
    test.compare_epyccel( 4 )
    test = epyccel_test(base.cast_float_to_int, lang=language)
    test.compare_epyccel( 4.5 )

def test_cast_bool(language):
    test = epyccel_test(base.cast_bool, lang=language)
    test.compare_epyccel( True )

def test_cast_float(language):
    test = epyccel_test(base.cast_float, lang=language)
    test.compare_epyccel( 4.5 )
    test = epyccel_test(base.cast_int_to_float, lang=language)
    test.compare_epyccel( 4 )

def test_if_0_int(language):
    test = epyccel_test(base.if_0_int, lang=language)
    test.compare_epyccel( 22 )
    test.compare_epyccel( 0 )

def test_if_0_real(language):
    test = epyccel_test(base.if_0_real, lang=language)
    test.compare_epyccel( 22.3 )
    test.compare_epyccel( 0.0 )

def test_same_int(language):
    test = epyccel_test(base.is_same_int, lang=language)
    test.compare_epyccel( 22 )
    test = epyccel_test(base.isnot_same_int, lang=language)
    test.compare_epyccel( 22 )

def test_same_float(language):
    test = epyccel_test(base.is_same_float, lang=language)
    test.compare_epyccel( 22.2 )
    test = epyccel_test(base.isnot_same_float, lang=language)
    test.compare_epyccel( 22.2 )

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Strings are not yet implemented for C language"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_same_string(language):
    test = epyccel_test(base.is_same_string, lang=language)
    test.compare_epyccel()
    test = epyccel_test(base.isnot_same_string, lang=language)
    test.compare_epyccel()

def test_same_complex(language):
    test = epyccel_test(base.is_same_complex, lang=language)
    test.compare_epyccel( complex(2,3) )
    test = epyccel_test(base.isnot_same_complex, lang=language)
    test.compare_epyccel( complex(2,3) )

def test_is_types(language):
    test = epyccel_test(base.is_types, lang=language)
    test.compare_epyccel( 1, 1.0 )

def test_isnot_types(language):
    test = epyccel_test(base.isnot_types, lang=language)
    test.compare_epyccel( 1, 1.0 )

def test_none_is_none(language):
    test = epyccel_test(base.none_is_none, lang=language)
    test.compare_epyccel()

def test_none_isnot_none(language):
    test = epyccel_test(base.none_isnot_none, lang=language)
    test.compare_epyccel()

def test_pass_if(language):
    test = epyccel_test(base.pass_if, lang=language)
    test.compare_epyccel(2)

def test_pass2_if(language):
    test = epyccel_test(base.pass2_if, lang=language)
    test.compare_epyccel(0.2)
    test.compare_epyccel(0.0)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.xfail(reason="Confusion around ValuedVariable means it cannot be used in python"),
            pytest.mark.python]
        )
    )
)
def test_use_optional(language):
    test = epyccel_test(base.use_optional, lang=language)
    test.compare_epyccel()
    test.compare_epyccel(6)

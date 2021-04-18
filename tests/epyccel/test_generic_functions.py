 # pylint: disable=missing-function-docstring, missing-module-docstring/
import sys
import pytest
import numpy as np
import modules.generic_functions as mod
import modules.generic_functions_2 as mod2
from pyccel.epyccel import epyccel

def language(request):
    return request.param

def test_gen_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_1()
    x = modnew.tst_gen_1()
    assert np.array_equal(x, x_expected)

def test_gen_2(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_2()
    x = modnew.tst_gen_2()
    assert np.array_equal(x ,x_expected)

def test_gen_3(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_3()
    x = modnew.tst_gen_3()
    assert np.array_equal(x, x_expected)

def test_gen_4(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_4()
    x = modnew.tst_gen_4()
    assert np.array_equal(x, x_expected)

def test_gen_5(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_5()
    x = modnew.tst_gen_5()
    assert np.array_equal(x, x_expected)

def test_gen_6(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_6()
    x = modnew.tst_gen_6()
    assert np.array_equal(x, x_expected)

def test_gen_7(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_7()
    x = modnew.tst_gen_7()
    assert np.array_equal(x, x_expected)

def test_multi_heads_1(language):

    f1 = epyccel(mod2.multi_heads_1, language = language)
    f2 = mod2.multi_heads_1

    assert f1(5, 5) == f2(5, 5)
    assert f1(5, 7.3) == f2(5, 7.3)

def test_tmplt_1(language):
    f1 = epyccel(mod2.tmplt_1, language = language)
    f2 = mod2.tmplt_1

    assert f1(5, 5) == f2(5, 5)
    assert f1(5.5, 7.3) == f2(5.5, 7.3)

def test_multi_tmplt_1(language):
    f1 = epyccel(mod2.multi_tmplt_1, language = language)
    f2 = mod2.multi_tmplt_1

    assert f1(5, 5, 7) == f2(5, 5, 7)
    assert f1(5, 5, 7.3) == f2(5, 5, 7.3)
    assert f1(4.5, 4.5, 8) == f2(4.5, 4.5, 8)
    assert f1(7.5, 3.5, 7.7) == f2(7.5, 3.5, 7.7)

def test_tmplt_head_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_tmplt_head_1()
    x = modnew.tst_tmplt_head_1()
    assert np.array_equal(x, x_expected)

def test_local_overide_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_local_overide_1()
    x = modnew.tst_local_overide_1()
    assert np.array_equal(x, x_expected)

def test_tmplt_tmplt_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_tmplt_tmplt_1()
    x = modnew.tst_tmplt_tmplt_1()
    assert np.array_equal(x, x_expected)

def test_tmplt_2(language):
    f1 = epyccel(mod2.tmplt_2, language = language)
    f2 = mod2.tmplt_2

    assert f1(5, 5) == f2(5, 5)
    assert f1(5.5, 7.3) == f2(5.5, 7.3)

def test_multi_tmplt_2(language):
    f1 = epyccel(mod2.multi_tmplt_2, language = language)
    f2 = mod2.multi_tmplt_2

    assert f1(5, 5) == f2(5, 5)
    assert f1(5, 7.3) == f2(5, 7.3)

#--------------------------------------------------------------------
# TEST DEFAULT ARGUMENTS
#--------------------------------------------------------------------

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Confusion around ValuedVariable means it cannot be used in python"),
            pytest.mark.python]
        )
    )
)
def test_default_var_1(language):
    f1 = epyccel(mod2.default_var_1, language = language)
    f2 = mod2.default_var_1

    assert f1(5.3) == f2(5.3)
    assert f1(5) == f2(5)
    assert f1(5.3, 2) == f2(5.3, 2)
    assert f1(5, 2) == f2(5, 2)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Confusion around ValuedVariable means it cannot be used in python"),
            pytest.mark.python]
        )
    )
)
def test_default_var_2(language):
    f1 = epyccel(mod2.default_var_2, language = language)
    f2 = mod2.default_var_2

    assert f1(5.3) == f2(5.3)
    assert f1(5) ==  f2(5)
    assert f1(5.3, complex(1, 3)) == f2(5.3, complex(1, 3))
    assert f1(5, 4.44+15.2j) == f2(5, 4.44+15.2j)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Confusion around ValuedVariable means it cannot be used in python"),
            pytest.mark.python]
        )
    )
)
def test_default_var_3(language):
    f1 = epyccel(mod2.default_var_3, language = language)
    f2 = mod2.default_var_3

    assert f1(5.3) == f1(5.3)
    assert f1(5) == f1(5)
    assert f1(5.3, True) == f1(5.3, True)
    assert f1(5, True) == f2(5, True)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Confusion around ValuedVariable means it cannot be used in python"),
            pytest.mark.python]
        )
    )
)
def test_default_var_4(language):
    f1 = epyccel(mod2.default_var_4 , language = language)
    f2 = mod2.default_var_4

    assert f1(5, 5) == f2(5, 5)
    assert f1(5.3, 5) == f2(5.3, 5)
    assert f1(4) == f2(4)
    assert f1(5.2) == f2(5.2)

#--------------------------------------------------------------------
# TEST OPTIONAL ARGUMENTS
#--------------------------------------------------------------------

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Confusion around ValuedVariable means it cannot be used in python"),
            pytest.mark.python]
        )
    )
)
def test_optional_var_1(language):
    f1 = epyccel(mod2.optional_var_1 , language = language)
    f2 = mod2.optional_var_1

    assert f1(5.3) == f2(5.3)
    assert f1(5) == f2(5)
    assert f1(5.3, 2) == f2(5.3, 2)
    assert f1(5, 2) == f2(5, 2)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Confusion around ValuedVariable means it cannot be used in python"),
            pytest.mark.python]
        )
    )
)
def test_optional_var_2(language):
    f1 = epyccel(mod2.optional_var_2 , language = language)
    f2 = mod2.optional_var_2

    assert f1(5.3) == f2(5.3)
    assert f1(5) == f2(5)
    assert f1(5.3, complex(1, 5)) == f2(5.3, complex(1, 5))
    assert f1(5, complex(1, 4)) == f2(5, complex(1, 4))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Confusion around ValuedVariable means it cannot be used in python"),
            pytest.mark.python]
        )
    )
)
def test_optional_var_3(language):
    f1 = epyccel(mod2.optional_var_3 , language = language)
    f2 = mod2.optional_var_3

    assert f1(5, 5.5) == f2(5, 5.5)
    assert f1(5.3, 5.5) == f2(5.3, 5.5)
    assert f1(4) == f2(4)
    assert f1(5.2) == f2(5.2)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason="Confusion around ValuedVariable means it cannot be used in python"),
            pytest.mark.python]
        )
    )
)
def test_optional_var_4(language):
    f1 = epyccel(mod2.optional_var_4 , language = language)
    f2 = mod2.optional_var_4

    assert f1(complex(5, 4), 5) == f2(complex(5, 4), 5)
    assert f1(2.2, 5)  == f2(2.2, 5)
    assert f1(4.2) == f2(4.2)
    assert f1(complex(4, 6)) == f2(complex(4, 6))

#--------------------------------------------------------------------
# TEST DATA TYPES
#--------------------------------------------------------------------
def test_int_types(language):
    f1 = epyccel(mod2.int_types , language = language)
    f2 = mod2.int_types

    assert f1(10, 5) == f2(10, 5)
    assert f1(int(15) , int(10)) == f2(int(15) , int(10))
    assert f1(np.int16(5), np.int16(4)) == f2(np.int16(5), np.int16(4))
    assert f1(np.int8(4), np.int8(7)) == f2(np.int8(4), np.int8(7))
    assert f1(np.int32(155), np.int32(177)) == f2(np.int32(155), np.int32(177))
    assert f1(np.int64(155), np.int64(177)) == f2(np.int64(155), np.int64(177))

def test_float_types(language):
    f1 = epyccel(mod2.float_types , language = language)
    f2 = mod2.float_types

    assert f1(10.5, 5.5) == f2(10.5, 5.5)
    assert f1(np.float32(155.2), np.float32(177.1)) == f2(np.float32(155.2), np.float32(177.1))
    assert f1(np.float64(166.6), np.float64(255.6)) == f2(np.float64(166.6), np.float64(255.6))

def test_complex_types(language):
    f1 = epyccel(mod2.complex_types , language = language)
    f2 = mod2.complex_types

    assert f1(complex(1, 2.2), complex(1, 2.2)) == f2(complex(1, 2.2), complex(1, 2.2))
    assert f1(np.complex64(15.5 + 2.0j) , np.complex64(10.5 + 3.4j)) == f2(np.complex64(15.5 + 2.0j) , np.complex64(10.5 + 3.4j))
    assert f1(np.complex128(15.5+ 2.0j) , np.complex128(10.5+ 3.4j)) == f2(np.complex128(15.5+ 2.0j) , np.complex128(10.5+ 3.4j))

def test_mix_types_1(language):
    f1 = epyccel(mod2.mix_types_1 , language = language)
    f2 = mod2.mix_types_1

    assert f1(complex(1, 2), 15, np.int16(5)) == f2(complex(1, 2), 15, np.int16(5))
    assert f1(complex(1, 2), 15, True) == f2(complex(1, 2), 15, True)
    assert f1(complex(1, 2), 7.0, np.int16(5)) == f2(complex(1, 2), 7.0, np.int16(5))
    assert f1(complex(1, 2), 7.0, False) == f2(complex(1, 2), 7.0, False)
    assert f1(15, 14, np.int16(2012)) == f2(15, 14, np.int16(2012))
    assert f1(15, 14, True) == f2(15, 14, True)
    assert f1(15, 7.0, np.int16(2012)) == f2(15, 7.0, np.int16(2012))
    assert f1(15, 14, False) == f2(15, 14, False)


def test_mix_types_2(language):
    f1 = epyccel(mod2.mix_types_2 , language = language)
    f2 = mod2.mix_types_2

    assert f1(-1, -1) == f2(-1, -1)
    assert f1(np.int64(4), np.int64(16)) == f2(np.int64(4), np.int64(16))
    assert f1(np.int16(4), np.int16(4)) == f2(np.int16(4), np.int16(4))
    assert f1(5.7, -1.2) == f2(5.7, -1.2)
    assert f1(complex(7.2, 3.12), complex(7.2, 3.12)) == f2(complex(7.2, 3.12), complex(7.2, 3.12))
    assert f1(np.float32(16), np.float32(16)) == f2(np.float32(16), np.float32(16))

def test_mix_types_3(language):
    f1 = epyccel(mod2.mix_types_3 , language = language)
    f2 = mod2.mix_types_3

    assert f1(-1, -1) == f2(-1, -1)
    assert f1(np.int32(4), np.int32(16)) == f2(np.int32(4), np.int32(16))

#--------------------------------------------------------------------
# TEST ARRAYS
#--------------------------------------------------------------------

def test_mix_array_1(language):
    f1 = epyccel(mod2.mix_array_1, language = language)
    f2 = mod2.mix_array_1

    a = 5
    x1 = np.array( [1,2,3], dtype=np.int64)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2 )

    x1 = np.array( [1.0,2.0,3.0], dtype=np.float64)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2)

    x1 = np.array( [1+ 2j,5 +2j,3.0 + 3j], dtype=np.complex128)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2)

def test_mix_array_2(language):
    f1 = epyccel(mod2.mix_array_2, language = language)
    f2 = mod2.mix_array_2

    a = 5
    x1 = np.array([1.0,2.0,3.0], dtype=np.float64)
    x2 = np.copy(x1)
    y1 = np.array([1,2,3], dtype=np.int64)
    y2 = np.copy(y1)
    f1(x1, y1, a)
    f2(x2, y2, a)

    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)

    x1 = np.array([1+ 2j, 5 +2j, 3.0 + 3j], dtype=np.complex128)
    x2 = np.copy(x1)
    y1 = np.array([0+ 5j, 5.0 -2j, 3.0 - 3j], dtype=np.complex128)
    y2 = np.copy(y1)
    f1(x1, y1, a)
    f2(x2, y2, a)

    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)

def test_mix_int_array(language):
    f1 = epyccel(mod2.mix_int_array_1, language = language)
    f2 = mod2.mix_int_array_1

    a = 5
    x1 = np.array([155,221,333], dtype=np.int64)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2 )

    x1 = np.array([127,229,3], dtype=np.int32)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2 )

    x1 = np.array([16,-27,34], dtype=np.int16)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2 )

    x1 = np.array([166,20,-5], dtype=np.int8)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2 )

def test_mix_int_array_2(language):
    f1 = epyccel(mod2.mix_int_array_2, language = language)
    f2 = mod2.mix_int_array_2

    a = 5
    x1 = np.array([155,221,333], dtype=np.int64)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2 )

    x1 = np.array([127,229,3], dtype=np.int32)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2 )

    x1 = np.array([127,229,3], dtype=int)
    x2 = np.copy(x1)
    f1(x1, a)
    f2(x2, a)
    assert np.array_equal( x1, x2 )

def test_mix_float_array(language):
    f1 = epyccel(mod2.mix_float_array_1, language = language)
    f2 = mod2.mix_float_array_1

    a = 5.44
    x1 = np.array([1.15,2.44,3.785], dtype=np.float64)
    x2 = np.copy(x1)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)

    x1 = np.array([1.1555,2115.44,3492.785], dtype=np.float32)
    x2 = np.copy(x1)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)

def test_mix_complex_array(language):
    f1 = epyccel(mod2.mix_complex_array_1, language = language)
    f2 = mod2.mix_complex_array_1

    a = 7.5
    x1 = np.array([10.33+ 2.55j, 5.125 +2.10j, 314.0 + 3.44j], dtype=np.complex128)
    x2 = np.copy(x1)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)

    x1 = np.array([145.32+ 25.55j, 57.15 +2.15j, 13.44j], dtype=np.complex64)
    x2 = np.copy(x1)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)

def test_dup_header(language):
    f1 = epyccel(mod2.dup_header , language = language)
    f2 = mod2.dup_header

    assert f1(0.0) == f2(0.0)

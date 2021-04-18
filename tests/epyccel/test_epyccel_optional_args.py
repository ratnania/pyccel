# pylint: disable=missing-function-docstring, missing-module-docstring/

import pytest
import numpy as np

from pyccel.epyccel import epyccel
from pyccel.decorators import types

@pytest.fixture(params=[
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="Confusion around ValuedVariable means it cannot be used in python"),
        pytest.mark.python]
    )]
)
def language(request):
    return request.param

#------------------------------------------------------------------------------
def test_f1(language):
    @types('int')
    def f1(x = None):
        if x is None :
            return 5
        return x + 5

    f = epyccel(f1, language = language)

    # ...
    assert f(2) == f1(2)
    assert f() == f1()
    assert f(None) == f1(None)
    assert f(0) == f1(0)
    # ...
#------------------------------------------------------------------------------
def test_f2(language):
    @types('real')
    def f2(x = None):
        if x is None :
            return 2.5
        return x + 2.5

    f = epyccel(f2, language = language)

    # ...
    assert f(2.0) == f2(2.0)
    assert f() == f2()
    assert f(None) == f2(None)
    assert f(0.0) == f2(0.0)
    # ...
#------------------------------------------------------------------------------
def test_f3(language):
    @types('complex')
    def f3(x = None):
        if x is None :
            return complex(2, 5.2)
        return x + complex(2.5, 2)

    f = epyccel(f3, language = language)

    # ...
    assert f(complex(1, 2.2)) == f3(complex(1, 2.2))
    assert f() == f3()
    assert f(None) == f3(None)
    # ...
#------------------------------------------------------------------------------
def test_f4(language):
    @types('bool')
    def f4(x = None):
        if x is None :
            return True
        return False

    f = epyccel(f4, language = language)

    # ...
    assert f(True) == f4(True)
    assert f() == f4()
    assert f(None) == f4(None)
    assert f(False) == f4(False)
    # ...
#------------------------------------------------------------------------------
def test_f5(language):
    import modules.Module_3 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.func(1) == modnew.func(1)
    assert mod.func() == modnew.func()
    assert mod.func(None) == modnew.func(None)
    assert mod.func(0) == modnew.func(0)

#------------------------------------------------------------------------------
def test_f6(language):
    import modules.Module_4 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_1() == modnew.call_optional_1()
    assert mod.call_optional_2(None) == modnew.call_optional_2(None)
    assert mod.call_optional_2(0) == modnew.call_optional_2(0)
    assert mod.call_optional_2() == modnew.call_optional_2()
#------------------------------------------------------------------------------
def test_f7(language):
    import modules.Module_5 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_1(3) == modnew.call_optional_1(3)
    assert mod.call_optional_2() == modnew.call_optional_2()
    assert mod.call_optional_3(3) == modnew.call_optional_3(3)

#------------------------------------------------------------------------------
def test_f9(language):
    import modules.Module_5 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_4(3) == modnew.call_optional_4(3)
    assert mod.call_optional_5(3) == modnew.call_optional_5(3)
    assert mod.call_optional_6() == modnew.call_optional_6()
    assert mod.call_optional_7() == modnew.call_optional_7()
    assert mod.call_optional_8() == modnew.call_optional_8()

#------------------------------------------------------------------------------
def test_f10(language):
    import modules.Module_5 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_9() == modnew.call_optional_9()
    assert mod.call_optional_10() == modnew.call_optional_10()

#------------------------------------------------------------------------------
def test_f11(language):
    import modules.Module_5 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_11() == modnew.call_optional_11()
    assert mod.call_optional_12() == modnew.call_optional_12()

#------------------------------------------------------------------------------
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Arrays cannot be optional. See #757"),
            pytest.mark.c]
        )
    )
)
def test_optional_args_1d(language):
    @types( 'int[:]', 'int[:]')
    def f12(x, y = None):
        if y is None:
            x[:] *= 2
        else :
            x[:] = x // y
    f = epyccel(f12, language = language)

    x1 = np.array( [1,2,3], dtype=int )
    x2 = np.copy(x1)
    f(x1)
    f12(x2)

    # ...
    assert np.array_equal(x1, x2)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Arrays cannot be optional. See #757"),
            pytest.mark.c]
        )
    )
)
def test_optional_2d_F(language):
    @types('int32[:,:](order=F)', 'int32[:,:](order=F)')
    def f13(x, y = None):
        if y is None:
            x[:] *= 2
        else :
            x[:] = x // y
    f = epyccel(f13, language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    f(x1)
    f13 (x2)

    # ...
    assert np.array_equal(x1, x2)
#------------------------------------------------------------------------------

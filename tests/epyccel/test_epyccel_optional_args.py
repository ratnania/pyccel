import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import types
from conftest       import *

#------------------------------------------------------------------------------
@pytest.mark.parametrize( 'language', [
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="f2py does not support optional arguments"),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c)
    ]
)
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
@pytest.mark.parametrize( 'language', [
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="f2py does not support optional arguments"),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c)
    ]
)
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
@pytest.mark.parametrize('language' , [
    pytest.param("fortran", marks = [
        pytest.mark.xfail(reason="f2py does not support optional arguments"),
        pytest.mark.fortran]),
    pytest.param("c" , marks = pytest.mark.c)
])
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
@pytest.mark.parametrize( 'language', [
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="f2py does not support optional arguments"),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c)
    ]
)
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

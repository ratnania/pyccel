from pyccel.decorators import types
from pyccel.epyccel import epyccel
import shutil

from conftest import *
def clean_test():
    shutil.rmtree('__pycache__', ignore_errors=True)
    shutil.rmtree('__epyccel__', ignore_errors=True)

def test_or_boolean(language):
    @types('bool', 'bool')
    def or_bool(a, b):
        c = False
        if (a):
            c = True
        if (b):
            c = True
        return c
    epyc_or_bool = epyccel(or_bool, language=language)

    assert(epyc_or_bool(True,True)==or_bool(True,True))
    assert(epyc_or_bool(True,False)==or_bool(True,False))
    assert(epyc_or_bool(False,False)==or_bool(False,False))

def test_real_greater_bool(language):
    @types('real', 'real')
    def real_greater_bool(x0, x1):
        greater = False
        if x0 > x1:
            greater = True
        return greater

    epyc_real_greater_bool = epyccel(real_greater_bool, language=language)

    assert(real_greater_bool(1.0,2.0)==epyc_real_greater_bool(1.0,2.0))
    assert(real_greater_bool(1.5,1.2)==epyc_real_greater_bool(1.5,1.2))

def test_input_output_matching_types():
    @types('float32', 'float32')
    def add_real(a, b):
        c = a+b
        return c

    epyc_add_real = epyccel(add_real , fflags="-Werror -Wconversion-extra")

    assert(add_real(1.0,2.0)==epyc_add_real(1.0,2.0))

def test_output_types_1(language):
    @types('float32')
    def cast_to_int(a):
        b = int(a)
        return b

    f = epyccel(cast_to_int, language = language)
    assert(type(cast_to_int(5.2)) == type(f(5.2)))

def test_output_types_2(language):
    @types('int')
    def cast_to_float(a):
        b = float(a)
        return b

    f = epyccel(cast_to_float,language= language)
    assert(type(cast_to_float(5)) ==  type(f(5)))    

def test_output_types_3(language):
    @types('int')
    def cast_to_bool(a):
        b = bool(a)
        return b
    
    f = epyccel(cast_to_bool, language=language)
    assert(cast_to_bool(1) == f(1))



##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module():
    clean_test()


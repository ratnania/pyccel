# pylint: disable=missing-function-docstring, missing-module-docstring/
import inspect
import pytest
import numpy as np

from pyccel.epyccel import epyccel
from modules import tuples as tuples_module

def is_func_with_0_args(f):
    """ Test if name 'f' corresponds to an argument in the
    tuples module with no arguments
    """
    func = getattr(tuples_module,f)
    return inspect.isfunction(func) and len(inspect.signature(func).parameters)==0

inhomog_homog = ['inhomogenous_tuple_2_levels_1', 'inhomogenous_tuple_2_levels_2']

tuple_funcs = [(f, getattr(tuples_module,f)) for f in tuples_module.__all__
                                            if is_func_with_0_args(f)]
inhomogeneous_funcs = [(n,f) for n,f in tuple_funcs if 'inhomog' in n and n not in inhomog_homog]
homogeneous_funcs = [(n,f) for n,f in tuple_funcs if ((n,f) not in inhomogeneous_funcs)]

failing_tests = {
        'homogenous_tuple_string':'String has no precision',
        'tuple_multi_indexing_1':'Multi object part of numpy array stored in sympy Tuple',
        'tuple_multi_indexing_2':'Multi object part of numpy array stored in sympy Tuple',
        'tuple_homogeneous_return':"Can't return a tuple",
        'tuple_inhomogeneous_return':"Can't return a tuple",
        'tuple_visitation_inhomogeneous':"Can't iterate over an inhomogeneous tuple",
        }

def compare_python_pyccel( p_output, f_output ):
    if p_output is None:
        assert(f_output is None)
        return
    if not hasattr(p_output, '__len__'):
        p_output = [p_output]
    if not hasattr(f_output, '__len__'):
        f_output = [f_output]
    assert(len(p_output) == len(f_output))

    for pth, pycc in zip(p_output, f_output):

        if isinstance(pth, bool):
            pycc_bool = (pycc == 1)
            assert(pth == pycc_bool)

        elif isinstance(pth, (int, str)):
            assert(isinstance(pycc,type(pth)))
            assert(pth==pycc)

        else:
            assert(np.isclose(pth,pycc))

homog_marks = [f[1] if f[0] not in failing_tests else
        pytest.param(f[1], marks = pytest.mark.xfail(reason=failing_tests[f[0]])) for f in homogeneous_funcs]

@pytest.mark.parametrize('test_func',homog_marks)
def test_homogeneous_tuples(test_func):
    f1 = test_func
    f2 = epyccel( f1 )

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)

inhomog_marks = [f[1] if f[0] not in failing_tests else
        pytest.param(f[1], marks = pytest.mark.xfail(reason=failing_tests[f[0]])) for f in inhomogeneous_funcs]

@pytest.mark.parametrize('test_func',inhomog_marks)
def test_inhomogeneous_tuples(test_func, language):
    f1 = test_func
    f2 = epyccel( f1, language=language )

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)

@pytest.mark.parametrize('test_func',
        [tuples_module.tuple_unpacking_3,
         tuples_module.tuple_unpacking_4]
)
def test_tuples_with_2d_args(test_func, language):
    f1 = test_func
    f2 = epyccel( f1, language=language )

    python_x = np.random.randint(100, size=(3,4))
    pyccel_x = python_x.copy()

    f1(python_x)
    f2(pyccel_x)
    np.allclose(python_x, pyccel_x)


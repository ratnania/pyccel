#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=missing-function-docstring

"""
Handling the transitions between python code and C code.
"""

import numpy as np

from ..errors.errors import Errors
from ..errors.messages import PYCCEL_RESTRICTION_TODO

from .basic     import Basic

from .datatypes import DataType
from .datatypes import NativeInteger, NativeReal, NativeComplex
from .datatypes import NativeBool, NativeString, NativeGeneric
from .literals  import LiteralFalse, LiteralTrue

from .core      import FunctionCall, FunctionDef, FunctionAddress
from .core      import AliasAssign, Assign, Return, If, DottedVariable
from .core      import IfSection

from .literals  import LiteralTrue

from .numpyext  import NumpyReal, NumpyImag

from .operators import PyccelEq

from .variable  import Variable, ValuedVariable, VariableAddress


errors = Errors()

__all__ = (
#
# --------- CLASSES -----------
#
    'PyccelPyObject',
    'PyccelPyArrayObject',
    'PyArgKeywords',
    'PyArg_ParseTupleNode',
    'PyBuildValueNode',
#--------- CONSTANTS ----------
    'Py_True',
    'Py_False',
    'Py_None',
#----- C / PYTHON FUNCTIONS ---
    'pycomplex_real',
    'pycomplex_imag',
    'pycomplex_fromdoubles',
    'Py_DECREF',
    'PyLong_AsLong',
    'PyFloat_AsDouble',
    'PythonType_Check',
    'NumpyType_Check',
    'PyErr_SetString',
#------- CAST FUNCTIONS ------
    'pyint_to_bool',
    'bool_to_pyobj',
    'pycomplex_to_complex',
    'complex_to_pycomplex',
    'pybool_to_bool',
#--------- Numpy ----------
    'numpy_get_ndims',
    'numpy_get_data',
    'numpy_get_dim',
    'numpy_get_stride',
    'numpy_check_flag',
    'numpy_get_base',
    'numpy_itemsize',
    'numpy_flag_own_data',
    'numpy_flag_c_contig',
    'numpy_flag_f_contig',
    'numpy_dtype_registry',
)

class PyccelPyObject(DataType):
    __slots__ = ()
    _name = 'pyobject'

class PyccelPyArrayObject(DataType):
    """ Datatype representing a PyArrayObject which is the
    class used to hold numpy objects"""
    __slots__ = ()
    _name = 'pyarrayobject'

PyArray_Type = Variable(NativeGeneric(), 'PyArray_Type')

#TODO: Is there an equivalent to static so this can be a static list of strings?
class PyArgKeywords(Basic):
    """
    Represents the list containing the names of all arguments to a function.
    This information allows the function to be called by keyword

    Parameters
    ----------
    name : str
        The name of the variable in which the list is stored
    arg_names : list of str
        A list of the names of the function arguments
    """
    __slots__ = ('_name','_arg_names')
    _attribute_nodes = ()
    def __init__(self, name, arg_names):
        self._name = name
        self._arg_names = arg_names
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def arg_names(self):
        return self._arg_names

#using the documentation of PyArg_ParseTuple() and Py_BuildValue https://docs.python.org/3/c-api/arg.html
pytype_parse_registry = {
    (NativeInteger(), 4)       : 'i',
    (NativeInteger(), 8)       : 'l',
    (NativeInteger(), 2)       : 'h',
    (NativeInteger(), 1)       : 'b',
    (NativeReal(), 8)          : 'd',
    (NativeReal(), 4)          : 'f',
    (NativeComplex(), 4)       : 'O',
    (NativeComplex(), 8)       : 'O',
    (NativeBool(), 4)          : 'p',
    (NativeString(), 0)        : 's',
    (PyccelPyObject(), 0)      : 'O',
    (PyccelPyArrayObject(), 0) : 'O!',
    }

class PyArg_ParseTupleNode(Basic):
    """
    Represents a call to the function from Python.h which collects the expected arguments

    Parameters
    ----------
    python_func_args: Variable
        Args provided to the function in python
    python_func_kwargs: Variable
        Kwargs provided to the function in python
    c_func_args: list of Variable
        List of expected arguments. This helps determine the expected output types
    parse_args: list of Variable
        List of arguments into which the result will be collected
    arg_names : list of str
        A list of the names of the function arguments
    is_interface : boolean
        Default value False and True when working with interface functions
    """
    __slots__ = ('_pyarg','_pykwarg','_parse_args','_arg_names','_flags')
    _attribute_nodes = ('_pyarg','_pykwarg','_parse_args','_arg_names')

    def __init__(self, python_func_args,
                        python_func_kwargs,
                        c_func_args, parse_args,
                        arg_names):
        if not isinstance(python_func_args, Variable):
            raise TypeError('Python func args should be a Variable')
        if not isinstance(python_func_kwargs, Variable):
            raise TypeError('Python func kwargs should be a Variable')
        if not all(isinstance(c, (Variable, FunctionAddress)) for c in c_func_args):
            raise TypeError('C func args should be a list of Variables')
        if not isinstance(parse_args, list) and any(not isinstance(c, Variable) for c in parse_args):
            raise TypeError('Parse args should be a list of Variables')
        if not isinstance(arg_names, PyArgKeywords):
            raise TypeError('Parse args should be a list of Variables')
        if len(parse_args) != len(c_func_args):
            raise TypeError('There should be the same number of c_func_args and parse_args')

        self._flags      = ''
        i = 0

        while i < len(c_func_args) and not isinstance(c_func_args[i], ValuedVariable):
            self._flags += self.get_pytype(c_func_args[i], parse_args[i])
            i+=1
        if i < len(c_func_args):
            self._flags += '|'
        while i < len(c_func_args):
            self._flags += self.get_pytype(c_func_args[i], parse_args[i])
            i+=1
        # Restriction as of python 3.8
        if any([isinstance(a, (Variable, FunctionAddress)) and a.is_kwonly for a in c_func_args]):
            errors.report('Kwarg only arguments without default values will not raise an error if they are not passed',
                          symbol=c_func_args, severity='warning')

        parse_args = [[PyArray_Type, a] if isinstance(a, Variable) and a.dtype is PyccelPyArrayObject()
                else [a] for a in parse_args]
        parse_args = [a for arg in parse_args for a in arg]

        self._pyarg      = python_func_args
        self._pykwarg    = python_func_kwargs
        self._parse_args = parse_args
        self._arg_names  = arg_names
        super().__init__()

    def get_pytype(self, c_arg, parse_arg):
        """Return the needed flag to parse or build value
        """
        if isinstance(c_arg, FunctionAddress):
            return 'O'
        else:
            try:
                return pytype_parse_registry[(parse_arg.dtype, parse_arg.precision)]
            except KeyError as e:
                raise NotImplementedError("Type not implemented for argument collection : "+str(type(parse_arg))) from e

    @property
    def pyarg(self):
        return self._pyarg

    @property
    def pykwarg(self):
        return self._pykwarg

    @property
    def flags(self):
        return self._flags

    @property
    def args(self):
        return self._parse_args

    @property
    def arg_names(self):
        return self._arg_names

class PyBuildValueNode(Basic):
    """
    Represents a call to the function from Python.h which create a new value based on a format string

    Parameters
    ---------
    parse_args: list of Variable
        List of arguments which the result will be buit from
    """
    __slots__ = ('_flags','_result_args',)
    _attribute_nodes = ('_result_args',)

    def __init__(self, result_args = ()):
        self._flags = ''
        self._result_args = result_args
        for i in result_args:
            self._flags += pytype_parse_registry[(i.dtype, i.precision)]
        super().__init__()

    @property
    def flags(self):
        return self._flags

    @property
    def args(self):
        return self._result_args

#-------------------------------------------------------------------
#                      Python.h functions
#-------------------------------------------------------------------

# Python.h object  representing Booleans True and False
Py_True = Variable(PyccelPyObject(), 'Py_True',is_pointer=True)
Py_False = Variable(PyccelPyObject(), 'Py_False',is_pointer=True)

# Python.h object representing None
Py_None = Variable(PyccelPyObject(), 'Py_None', is_pointer=True)

# https://docs.python.org/3/c-api/refcounting.html#c.Py_DECREF
Py_DECREF = FunctionDef(name = 'Py_DECREF',
                        body = [],
                        arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)],
                        results = [])

#-------------------------------------------------------------------
#                      cwrapper.h functions
#-------------------------------------------------------------------

def Python_to_C(c_object):
    """
    Create FunctionDef responsible for casting python argument to C
    Parameters:
    ----------
    c_object  : Variable
        The variable needed for the generation of the cast_function
    Returns
    -------
    FunctionDef : cast type FunctionDef
    """
    try :
        cast_function = py_to_c_registry[(c_object.dtype, c_object.precision)]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')
    cast_func = FunctionDef(name = cast_function,
                       body      = [],
                       arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)],
                       results   = [Variable(dtype=c_object.dtype, name = 'v', precision = c_object.precision)])

    return cast_func

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
py_to_c_registry = {
    (NativeBool(), 4)      : 'PyBool_to_Bool',
    (NativeInteger(), 1)   : 'PyInt8_to_Int8',
    (NativeInteger(), 2)   : 'PyInt16_to_Int16',
    (NativeInteger(), 4)   : 'PyInt32_to_Int32',
    (NativeInteger(), 8)   : 'PyInt64_to_Int64',
    (NativeReal(), 4)      : 'PyFloat_to_Float',
    (NativeReal(), 8)      : 'PyDouble_to_Double',
    (NativeComplex(), 4)   : 'PyComplex_to_Complex64',
    (NativeComplex(), 8)   : 'PyComplex_to_Complex128'}

def C_to_Python(c_object):
    """
    Create FunctionDef responsible for casting c argument to python
    Parameters:
    ----------
    c_object  : Variable
        The variable needed for the generation of the cast_function
    Returns
    -------
    FunctionDef : cast type FunctionDef
    """
    try :
        cast_function = c_to_py_registry[(c_object.dtype, c_object.precision)]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')

    cast_func = FunctionDef(name = cast_function,
                       body      = [],
                       arguments = [Variable(dtype=c_object.dtype, name = 'v', precision = c_object.precision)],
                       results   = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)])

    return cast_func

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
# TODO create cast functions of different precision of int issue #735
c_to_py_registry = {
    (NativeBool(), 4)      : 'Bool_to_PyBool',
    (NativeInteger(), 1)   : 'Int8_to_PyLong',
    (NativeInteger(), 2)   : 'Int16_to_PyLong',
    (NativeInteger(), 4)   : 'Int32_to_PyLong',
    (NativeInteger(), 8)   : 'Int64_to_PyLong',
    (NativeReal(), 4)      : 'Float_to_PyDouble',
    (NativeReal(), 8)      : 'Double_to_PyDouble',
    (NativeComplex(), 4)   : 'Complex64_to_PyComplex',
    (NativeComplex(), 8)   : 'Complex128_to_PyComplex'}


#-------------------------------------------------------------------
#              errors and check functions
#-------------------------------------------------------------------

# https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Occurred
PyErr_Occurred = FunctionDef(name      = 'PyErr_Occurred',
                             arguments = [],
                             results   = [Variable(dtype = PyccelPyObject(), name = 'r', is_pointer = True)],
                             body      = [])

def PyErr_SetString(exception, message):
    """
    Generate function Call of c/python api PyErr_SetString
    https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetString
    with a defined error message used to set the error indicator.

    Parameters:
    ----------
    exception  : str
        The exception type
    message    : str
        Error message
    Returns
    FunctionCall : raise error FunctionCall
    """
    func = FunctionDef(name = 'PyErr_SetString',
                  body      = [],
                  arguments = [Variable(dtype = PyccelPyObject(), name = 'o'),
                               Variable(dtype = NativeString(), name = 's')],
                  results   = [])

    exception = Variable(PyccelPyObject(), name = exception)

    return FunctionCall(func, [exception, message])


def generate_datatype_error(variable):
    """
    Generate TypeError exception from the variable information (datatype, precision)
    Parameters:
    ----------
    variable : Variable

    Returns:
    -------
    func     : FunctionCall
        call to PyErr_SetString with TypeError as exception and custom message
    """
    dtype     = variable.dtype

    if isinstance(dtype, NativeBool):
        precision = ''
    if isinstance(dtype, NativeComplex):
        precision = '{} bit '.format(variable.precision * 2 * 8)
    else:
        precision = '{} bit '.format(variable.precision * 8)

    message = '"Argument must be {precision}{dtype}"'.format(
            precision = precision,
            dtype     = variable.dtype)
    return PyErr_SetString('PyExc_TypeError', message)


# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
check_type_registry = {
    (NativeBool(), 4)      : 'PyIs_Bool',
    (NativeInteger(), 1)   : 'PyIs_Int8',
    (NativeInteger(), 2)   : 'PyIs_Int16',
    (NativeInteger(), 4)   : 'PyIs_Int32',
    (NativeInteger(), 8)   : 'PyIs_Int64',
    (NativeReal(), 4)      : 'PyIs_Float',
    (NativeReal(), 8)      : 'PyIs_Double',
    (NativeComplex(), 4)   : 'PyIs_Complex64',
    (NativeComplex(), 8)   : 'PyIs_Complex128'}

def scalar_object_check(py_object, c_object, precision_check = False):
    """
    Create FunctionCall responsible for checking python argument data type
    Parameters:
    ----------
    py_object  : Variable
        The python argument of the check function
    c_object   : Variable
        The variable needed for the generation of the type check
    precision_check : Boolean
        True if checking the exact precision is needed
    Returns
    -------
    FunctionCall : Check type FunctionCall
    """

    try :
        check_type = check_type_registry[c_object.dtype, c_object.precision]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')

    precision_check = LiteralTrue() if precision_check else LiteralFalse()

    check_func = FunctionDef(name = check_type,
                    body      = [],
                    arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True),
                                 Variable(dtype = NativeBool(), name = 'hard_check')],
                    results   = [Variable(dtype=NativeBool(), name = 'r')])

    return FunctionCall(check_func, [py_object, precision_check])

# This registry is used for interface management,
# mapping each data type to a given flag
# Those flags are used in a bitset #TODO
flags_registry = {
    (NativeInteger(), 4)       : 1,
    (NativeInteger(), 8)       : 2,
    (NativeInteger(), 2)       : 3,
    (NativeInteger(), 1)       : 4,
    (NativeReal(), 8)          : 5,
    (NativeReal(), 4)          : 6,
    (NativeComplex(), 4)       : 7,
    (NativeComplex(), 8)       : 8,
    (NativeBool(), 4)          : 9,
    (NativeString(), 0)        : 10
}

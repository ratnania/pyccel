#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Handling the transitions between python code and C code using (Python/C Api).
"""

import numpy as np

from ..errors.errors   import Errors
from ..errors.messages import PYCCEL_RESTRICTION_TODO

from .basic     import Basic

from .literals  import LiteralTrue, LiteralFalse

from .datatypes import (DataType, NativeInteger, NativeReal, NativeComplex,
                        NativeBool, NativeString, NativeGeneric, NativeVoid)

from .operators import PyccelOr, PyccelNot

from .core      import FunctionCall, FunctionDef, FunctionAddress

from .variable  import Variable, ValuedVariable

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
    'malloc',
    'free',
    'sizeof',
#------- CAST FUNCTIONS ------
    'C_to_Python',
    'Python_to_C',
#-------CHECK FUNCTIONS ------
    'PythonType_Check',
    'scalar_checker',
#-------- Regestry -----------
    'flags_registry'
#---------Helpers-------------
    'generate_datatype_error'
)


#-------------------------------------------------------------------
#                        Python DataTypes
#-------------------------------------------------------------------

class PyccelPyObject(DataType):
    """ Datatype representing a PyObject which is the
    class used to hold python objects"""
    _name = 'pyobject'

class PyccelPyArrayObject(DataType):
    """ Datatype representing a PyArrayObject which is the
    class used to hold numpy objects"""
    _name = 'pyarrayobject'

PyArray_Type         = Variable(NativeGeneric(), 'PyArray_Type')
Py_CLEANUP_SUPPORTED = Variable(dtype=NativeInteger(),  name = 'Py_CLEANUP_SUPPORTED')

#-------------------------------------------------------------------
#                  Parsing and Building Classes
#-------------------------------------------------------------------

# TODO: Is there an equivalent to static so this can be a static list of strings?
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
    _attribute_nodes = ()
    def __init__(self, name, arg_names):
        self._name      = name
        self._arg_names = arg_names
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def arg_names(self):
        return self._arg_names

class PyArg_ParseTupleNode(Basic):
    """
    Represents a call to the function from Python.h which collects the expected arguments

    Parameters
    ----------
    python_func_args: Variable
        Args provided to the function in python
    python_func_kwargs: Variable
        Kwargs provided to the function in python
    converter_functions : dict
        dictionary maping argument to cast functions
    parse_args: list of Variable
        List of arguments into which the result will be collected
    arg_names : list of str
        A list of the names of the function arguments
    """

    _attribute_nodes = ('_pyarg','_pykwarg','_parse_args','_arg_names')

    def __init__(self, python_func_args,
                       python_func_kwargs,
                       arg_names,
                       func_args,
                       parse_args = [],
                       converters = None):

        if not isinstance(python_func_args, Variable):
            raise TypeError('Python func args should be a Variable')
        if not isinstance(python_func_kwargs, Variable):
            raise TypeError('Python func kwargs should be a Variable')
        if not isinstance(arg_names, PyArgKeywords):
            raise TypeError('Parse args should be a list of Variables')
        if not isinstance(func_args, list) and any(not isinstance(c, Variable) for c in func_args):
            raise TypeError('C func args should be a list of Variables')

        self._flags = ''
        i           = 0
        args_count  = len(func_args)
        while i < args_count and not isinstance(func_args[i], ValuedVariable):
            self._flags += self.get_pytype(func_args[i], converters)
            i += 1
        if i < args_count:
            self._flags += '|'
        while i < args_count:
            self._flags += self.get_pytype(func_args[i], converters)
            i += 1
        # Restriction as of python 3.8
        if any([isinstance(a, (Variable, FunctionAddress)) and a.is_kwonly for a in parse_args]):
            errors.report('Kwarg only arguments without default values will not raise an error if they are not passed',
                          symbol=parse_args, severity='warning')

        if converters:
            parse_args = [[c, a] for a, c in zip(func_args, converters)]
            parse_args = [a for args in parse_args for a in args]
        
        else:
            parse_args = [[PyArray_Type, a] if isinstance(a, Variable) and a.dtype is PyccelPyArrayObject()
                                            else [a] for a in parse_args]
            parse_args = [a for arg in parse_args for a in arg]

        self._pyarg               = python_func_args
        self._pykwarg             = python_func_kwargs
        self._parse_args          = parse_args
        self._arg_names           = arg_names
        super().__init__()


    def get_pytype(self, argument, converter):
        """Return the needed flag to parse or build value
        """
        if converter is not None:
            return 'O&'
        
        if argument.rank > 0:
            return 'O!'
        
        return 'O'


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

    converter_functions : dict
        dictionary maping argument to cast functions
    """
    _attribute_nodes = ('_result_args',)

    def __init__(self, result_args = (), converters = []):
        self._flags       = ''
        for i in result_args:
            self._flags  += 'O&'

        if (len(result_args) != len(converters)):
            raise TypeError('There should be same number of converter functions and arguments')

        self._result_args = result_args
        self._converters   = converters

        super().__init__()

    @property
    def flags(self):
        return self._flags

    @property
    def args(self):
        return self._result_args

    @property
    def converters(self):
        return self._converters

def get_custom_key(variable, for_interface):
    """
    """
    dtype     = variable.dtype
    precision = variable.precision
    rank      = variable.rank
    order     = variable.order
    optional  = variable.is_optional

    return (for_interface, optional, dtype, precision, order, rank)


#-------------------------------------------------------------------
#                      Python.h Constants
#-------------------------------------------------------------------

# Python.h object  representing Booleans True and False
Py_True  = Variable(PyccelPyObject(), 'Py_True',is_pointer=True)
Py_False = Variable(PyccelPyObject(), 'Py_False',is_pointer=True)

# Python.h object representing None
Py_None  = Variable(PyccelPyObject(), 'Py_None', is_pointer=True)

#-------------------------------------------------------------------
#                  C memory management functions
#-------------------------------------------------------------------

malloc   = FunctionDef(name      = 'malloc',
                       arguments = [Variable(name = 'size', dtype = NativeInteger())],
                       results   = [Variable(name = 'ptr', dtype = NativeVoid(), is_pointer  = True)],
                       body      = [])

free     = FunctionDef(name      = 'free',
                       arguments = [Variable(name = 'ptr', dtype = NativeVoid(), is_pointer = True)],
                       results   = [],
                       body      = [])

#sizeof operator presented as FunctionDef node
sizeof   = FunctionDef(name      = 'sizeof',
                       arguments = [Variable(name = 'ptr', dtype = NativeGeneric())],
                       results   = [Variable(name = 'size', dtype = NativeInteger())],
                       body      = [])

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
    used to set the error indicator.

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


# All functions used for type are from c python api :
# https://docs.python.org/3/c-api/long.html#c.PyLong_Check
# https://docs.python.org/3/c-api/complex.html#c.PyComplex_Check
# https://docs.python.org/3/c-api/float.html#c.PyFloat_Check
# https://docs.python.org/3/c-api/bool.html#c.PyBool_Check

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

def scalar_object_check(py_object, c_object, hard_check = False):
    """
    Create FunctionCall responsible for checking python argument data type
    Parameters:
    ----------
    py_object  : Variable
        The python argument of the check function
    c_object   : Variable
        The variable needed for the generation of the type check
    hard_check : boolean
        True if checking the exact precision is needed
    Returns
    -------
    FunctionCall : Check type FunctionCall
    """

    try :
        check_type = check_type_registry[c_object.dtype, c_object.precision]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')

    hard_check = LiteralTrue() if hard_check else LiteralFalse()

    check_func = FunctionDef(name = check_type,
                    body      = [],
                    arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True),
                                 Variable(dtype = NativeBool(), name = 'hard_check')],
                    results   = [Variable(dtype=NativeBool(), name = 'r')])

    return FunctionCall(check_func, [py_object, hard_check])

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

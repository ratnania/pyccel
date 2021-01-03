# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
The Python interpreter has a number of built-in functions and types that are
always available.

In this module we implement some of them in alphabetical order.

"""

from sympy import Symbol, Function, Tuple
from sympy import Expr, Not
from sympy import sympify
from sympy.tensor import Indexed, IndexedBase
from sympy.core.function import Application

from pyccel.ast.datatypes import iso_c_binding

from .basic     import Basic, PyccelAstNode
from .datatypes import (NativeInteger, NativeBool, NativeReal,
                        NativeComplex, NativeString, str_dtype,
                        NativeGeneric, default_precision)
from .literals  import LiteralInteger, LiteralFloat, LiteralComplex
from .literals  import Literal, LiteralImaginaryUnit, get_default_literal_value

__all__ = (
    'PythonReal',
    'PythonImag',
    'PythonBool',
    'PythonComplex',
    'PythonEnumerate',
    'PythonFloat',
    'PythonInt',
    'PythonTuple',
    'PythonLen',
    'PythonList',
    'PythonMap',
    'PythonPrint',
    'PythonRange',
    'PythonZip',
    'PythonMax',
    'PythonMin',
    'python_builtin_datatype'
)

#==============================================================================
# TODO [YG, 06.03.2020]: avoid core duplication between builtins and core
local_sympify = {
    'N'    : Symbol('N'),
    'S'    : Symbol('S'),
    'zeros': Symbol('zeros'),
    'ones' : Symbol('ones'),
    'Point': Symbol('Point')
}

#==============================================================================
class PythonComplexProperty(Application, PyccelAstNode):
    """Represents a call to the .real or .imag property

    e.g:
    > a = 1+2j
    > a.real
    1.0

    arg : Variable, Literal
    """
    _dtype = NativeReal()
    _rank  = 0
    _shape = ()

    def __init__(self, arg):
        self._precision = arg.precision

    @property
    def internal_var(self):
        """Return the variable on which the function was called"""
        return self._args[0]

    def __str__(self):
        return 'Real({0})'.format(str(self.internal_var))

#==============================================================================
class PythonReal(PythonComplexProperty):
    """Represents a call to the .real property

    e.g:
    > a = 1+2j
    > a.real
    1.0

    arg : Variable, Literal
    """
    def __new__(cls, arg):
        if arg.dtype is not NativeComplex():
            return arg
        else:
            return PythonComplexProperty.__new__(cls, arg)

#==============================================================================
class PythonImag(PythonComplexProperty):
    """Represents a call to the .imag property

    e.g:
    > a = 1+2j
    > a.imag
    1.0

    arg : Variable, Literal
    """
    def __new__(cls, arg):
        if arg.dtype is not NativeComplex():
            return get_default_literal_value(arg.dtype)
        else:
            return PythonComplexProperty.__new__(cls, arg)


#==============================================================================
class PythonBool(Expr, PyccelAstNode):
    """ Represents a call to Python's native bool() function.
    """
    _rank = 0
    _shape = ()
    _precision = default_precision['bool']
    _dtype = NativeBool()

    def __new__(cls, arg):
        return Expr.__new__(cls, arg)

    @property
    def arg(self):
        return self.args[0]

    def __str__(self):
        return 'Bool({})'.format(str(self.arg))

    def _sympystr(self, printer):
        return self.__str__()

#==============================================================================
class PythonComplex(Expr, PyccelAstNode):
    """ Represents a call to Python's native complex() function.
    """

    _rank = 0
    _shape = ()
    _precision = default_precision['complex']
    _dtype = NativeComplex()

    def __new__(cls, arg0, arg1=LiteralFloat(0)):

        if isinstance(arg0, Literal) and isinstance(arg1, Literal):
            real_part = 0
            imag_part = 0

            # Collect real and imag part from first argument
            if isinstance(arg0, LiteralComplex):
                real_part += arg0.real.python_value
                imag_part += arg0.imag.python_value
            else:
                real_part += arg0.python_value

            # Collect real and imag part from second argument
            if isinstance(arg1, LiteralComplex):
                real_part -= arg1.imag.python_value
                imag_part += arg1.real.python_value
            else:
                imag_part += arg1.python_value

            return LiteralComplex(real_part, imag_part, precision = cls._precision)


        # Split arguments depending on their type to ensure that the arguments are
        # either a complex and LiteralFloat(0) or 2 floats
        from .operators import PyccelAdd, PyccelMul

        if arg0.dtype is NativeComplex() and arg1.dtype is NativeComplex():
            # both args are complex
            return PyccelAdd(arg0, PyccelMul(arg1, LiteralImaginaryUnit()))
        return Expr.__new__(cls, arg0, arg1)

    def __init__(self, arg0, arg1 = LiteralFloat(0)):
        self._is_cast = arg0.dtype is NativeComplex() and \
                        isinstance(arg1, Literal) and arg1.python_value == 0

        if self._is_cast:
            self._real_part = PythonReal(arg0)
            self._imag_part = PythonImag(arg0)
            self._internal_var = arg0

        else:
            from .operators import PyccelAdd, PyccelMinus, PyccelUnarySub

            if arg0.dtype is NativeComplex() and \
                    not (isinstance(arg1, Literal) and arg1.python_value == 0):
                # first arg is complex. Second arg is non-0
                self._real_part = PythonReal(arg0)
                self._imag_part = PyccelAdd(PythonImag(arg0), arg1)
            elif arg1.dtype is NativeComplex():
                if isinstance(arg0, Literal) and arg0.python_value == 0:
                    # second arg is complex. First arg is 0
                    self._real_part = PyccelUnarySub(PythonImag(arg1))
                    self._imag_part = PythonReal(arg1)
                else:
                    # Second arg is complex. First arg is non-0
                    self._real_part = PyccelMinus(arg0, PythonImag(arg1))
                    self._imag_part = PythonReal(arg1)
            else:
                self._real_part = arg0
                self._imag_part = arg1

    @property
    def is_cast(self):
        """ Indicates if the function is casting or assembling a complex """
        return self._is_cast

    @property
    def real(self):
        """ Returns the real part of the complex """
        return self._real_part

    @property
    def imag(self):
        """ Returns the imaginary part of the complex """
        return self._imag_part

    @property
    def internal_var(self):
        """ When the complex call is a cast, returns the variable being cast """
        assert(self._is_cast)
        return self._internal_var

    def __str__(self):
        return "complex({}, {})".format(str(self._args[0]), str(self._args[1]))

#==============================================================================
class PythonEnumerate(Basic):

    """
    Represents the enumerate stmt

    """

    def __new__(cls, arg):
        if not isinstance(arg, (Symbol, Indexed, IndexedBase)):
            raise TypeError('Expecting an arg of valid type')
        return Basic.__new__(cls, arg)

    @property
    def element(self):
        return self._args[0]

#==============================================================================
class PythonFloat(Expr, PyccelAstNode):
    """ Represents a call to Python's native float() function.
    """
    _rank = 0
    _shape = ()
    _precision = default_precision['real']
    _dtype = NativeReal()

    def __new__(cls, arg):
        if isinstance(arg, LiteralFloat):
            return LiteralFloat(arg, precision = cls._precision)
        elif isinstance(arg, LiteralInteger):
            return LiteralFloat(arg.p, precision = cls._precision)
        else:
            return Expr.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]


    def __str__(self):
        return 'LiteralFloat({0})'.format(str(self.arg))

    def _sympystr(self, printer):
        return self.__str__()

#==============================================================================
class PythonInt(Expr, PyccelAstNode):
    """ Represents a call to Python's native int() function.
    """

    _rank      = 0
    _shape     = ()
    _precision = default_precision['integer']
    _dtype     = NativeInteger()

    def __new__(cls, arg):
        if isinstance(arg, LiteralInteger):
            return LiteralInteger(arg.p, precision = cls._precision)
        else:
            return Expr.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonTuple(Expr, PyccelAstNode):
    """ Represents a call to Python's native tuple() function.
    """
    _iterable        = True
    _order = 'C'

    def __new__(cls, *args):
        return Expr.__new__(cls, *args)

    def __init__(self, *args):
        if self.stage == 'syntactic' or len(args) == 0:
            return
        self._rank      = 1
        self._dtype     = NativeGeneric()
        self._precision = 0
        self._shape     = (LiteralInteger(len(args)), )

    def __getitem__(self,items):
        if isinstance(items, (list, tuple)):
            i = items[0]
            additional_items = items[1:]
        else:
            i = items
            additional_items = []
        if isinstance(i, LiteralInteger):
            i = i.p
        if additional_items:
            return self._args[i][additional_items]
        else:
            return self._args[i]

    def __add__(self,other):
        return PythonTuple(*(self._args + other._args))

    def __iter__(self):
        return self._args.__iter__()

    def __len__(self):
        return len(self._args)

#==============================================================================
class PythonLen(Function, PyccelAstNode):

    """
    Represents a 'len' expression in the code.
    """

    _rank      = 0
    _shape     = ()
    _precision = default_precision['int']
    _dtype     = NativeInteger()

    def __new__(cls, arg):
        return Basic.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonList(Tuple, PyccelAstNode):
    _order = 'C'
    _is_homogeneous = True
    """ Represent lists in the code with dynamic memory management."""
    def __init__(self, *args, **kwargs):
        if self.stage == 'syntactic':
            return
        self._is_homogeneous = all(a.dtype is not NativeGeneric() and \
                             args[0].dtype == a.dtype and \
                             args[0].rank  == a.rank  and \
                             args[0].order == a.order for a in args[1:])

        bools     = [a for a in args if a.dtype is NativeBool()]
        integers  = [a for a in args if a.dtype is NativeInteger()]
        reals     = [a for a in args if a.dtype is NativeReal()]
        complexes = [a for a in args if a.dtype is NativeComplex()]
        strs      = [a for a in args if a.dtype is NativeString()]
        if strs:
            self._dtype = NativeString()
            self._rank  = 0
            self._shape = ()

        else:
            if complexes:
                self._dtype     = NativeComplex()
                self._precision = max(a.precision for a in complexes)
            elif reals:
                self._dtype     = NativeReal()
                self._precision = max(a.precision for a in reals)
            elif integers:
                self._dtype     = NativeInteger()
                self._precision = max(a.precision for a in integers)
            elif bools:
                self._dtype     = NativeBool()
                self._precision  = max(a.precision for a in bools)

            shapes = [a.shape for a in args]
            self._rank = max(a.rank for a in args) + 1
            if all(sh is not None for sh in shapes):
                self._shape = (LiteralInteger(len(args)), ) + shapes[0]
                self._rank  = len(self._shape)

    @property
    def is_homogeneous(self):
        return self._is_homogeneous

#==============================================================================
class PythonMap(Basic):
    """ Represents the map stmt
    """
    def __new__(cls, *args):
        if len(args)<2:
            raise TypeError('wrong number of arguments')
        return Basic.__new__(cls, *args)

#==============================================================================
class PythonPrint(Basic):

    """Represents a print function in the code.

    expr : sympy expr
        The expression to return.

    Examples

    >>> from sympy import symbols
    >>> from pyccel.ast.core import Print
    >>> n,m = symbols('n,m')
    >>> Print(('results', n,m))
    Print((results, n, m))
    """

    def __new__(cls, expr):
        if not isinstance(expr, list):
            expr = sympify(expr, locals=local_sympify)
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class PythonRange(Basic):

    """
    Represents a range.

    Examples

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Range
    >>> from sympy import Symbol
    >>> s = Variable('int', 's')
    >>> e = Symbol('e')
    >>> Range(s, e, 1)
    Range(0, n, 1)
    """

    def __new__(cls, *args):
        start = LiteralInteger(0)
        stop = None
        step = LiteralInteger(1)

        _valid_args = (LiteralInteger, Symbol, Indexed)

        if isinstance(args, (tuple, list, Tuple)):
            if len(args) == 1:
                stop = args[0]
            elif len(args) == 2:
                start = args[0]
                stop = args[1]
            elif len(args) == 3:
                start = args[0]
                stop = args[1]
                step = args[2]
            else:
                raise ValueError('Range has at most 3 arguments')
        elif isinstance(args, _valid_args):
            stop = args
        else:
            raise TypeError('expecting a list or valid stop')

        return Basic.__new__(cls, start, stop, step)

    @property
    def start(self):
        return self._args[0]

    @property
    def stop(self):
        return self._args[1]

    @property
    def step(self):
        return self._args[2]


#==============================================================================
class PythonZip(Basic):

    """
    Represents a zip stmt.

    """

    def __new__(cls, *args):
        if not isinstance(args, (tuple, list, Tuple)):
            raise TypeError('args must be an iterable')
        elif len(args) < 2:
            raise ValueError('args must be of length > 2')
        return Basic.__new__(cls, *args)

    @property
    def element(self):
        return self._args[0]

#==============================================================================
class PythonAbs(Function, PyccelAstNode):
    """Represents a call to  python abs for code generation.

    arg : Variable
    """
    def __init__(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = NativeInteger() if x.dtype is NativeInteger() else NativeReal()
        self._precision = default_precision[str_dtype(self._dtype)]
        self._order     = x.order

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonSum(Function, PyccelAstNode):
    """Represents a call to  python sum for code generation.

    arg : list , tuple , PythonTuple, Tuple, List, Variable
    """

    def __new__(cls, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, PythonList,
                                Variable, Expr)): # pylint: disable=undefined-variable
            raise TypeError('Uknown type of  %s.' % type(arg))

        return Basic.__new__(cls, arg)

    def __init__(self, arg):
        self._dtype = arg.dtype
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision[str_dtype(self._dtype)]

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonMax(Function, PyccelAstNode):
    """Represents a call to  python max for code generation.

    arg : list , tuple , PythonTuple, Tuple, List
    """
    def __new__(cls, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, PythonList)):
            raise TypeError('Uknown type of  %s.' % type(arg))
        return Basic.__new__(cls, arg)

    def __init__(self, x):
        self._shape     = ()
        self._rank      = 0
        self._dtype     = x.dtype
        self._precision = x.precision


#==============================================================================
class PythonMin(Function, PyccelAstNode):
    """Represents a call to  python min for code generation.

    arg : list , tuple , PythonTuple, Tuple, List, Variable
    """
    def __init__(self, x):
        self._shape     = ()
        self._rank      = 0
        self._dtype     = x.dtype
        self._precision = x.precision

#==============================================================================
python_builtin_datatypes_dict = {
    'bool'   : PythonBool,
    'float'  : PythonFloat,
    'int'    : PythonInt,
    'complex': PythonComplex
}

def python_builtin_datatype(name):
    """
    Given a symbol name, return the corresponding datatype.

    name: str
        Datatype as written in Python.

    """
    if not isinstance(name, str):
        raise TypeError('name must be a string')

    if name in python_builtin_datatypes_dict:
        return python_builtin_datatypes_dict[name]

    return None

builtin_functions_dict = {
    'abs'      : PythonAbs,
    'range'    : PythonRange,
    'zip'      : PythonZip,
    'enumerate': PythonEnumerate,
    'int'      : PythonInt,
    'float'    : PythonFloat,
    'complex'  : PythonComplex,
    'bool'     : PythonBool,
    'sum'      : PythonSum,
    'len'      : PythonLen,
    'max'      : PythonMax,
    'min'      : PythonMin,
    'not'      : Not,   # TODO [YG, 20.05.2020]: do not use Sympy's Not
}

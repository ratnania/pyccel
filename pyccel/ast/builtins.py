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

from .basic     import Basic, PyccelAstNode
from .datatypes import (NativeInteger, NativeBool, NativeReal,
                        NativeComplex, NativeString, str_dtype,
                        NativeGeneric, default_precision)
from .internals import PyccelInternalFunction
from .literals  import LiteralInteger, LiteralFloat, LiteralComplex, Nil
from .literals  import Literal, LiteralImaginaryUnit, get_default_literal_value
from .operators import PyccelAdd, PyccelAnd, PyccelMul, PyccelIsNot
from .operators import PyccelMinus, PyccelUnarySub, PyccelNot

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
class PythonComplexProperty(PyccelInternalFunction):
    """Represents a call to the .real or .imag property

    e.g:
    > a = 1+2j
    > a.real
    1.0

    arg : Variable, Literal
    """
    __slots__ = ('_precision')
    _dtype = NativeReal()
    _rank  = 0
    _shape = ()
    _order = None

    def __init__(self, arg):
        self._precision = arg.precision
        super().__init__(arg)

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
    __slots__ = ()
    def __new__(cls, arg):
        if isinstance(arg.dtype, NativeBool):
            return PythonInt(arg)
        elif not isinstance(arg.dtype, NativeComplex):
            return arg
        else:
            return super().__new__(cls)

#==============================================================================
class PythonImag(PythonComplexProperty):
    """Represents a call to the .imag property

    e.g:
    > a = 1+2j
    > a.imag
    1.0

    arg : Variable, Literal
    """
    __slots__ = ()
    def __new__(cls, arg):
        if arg.dtype is not NativeComplex():
            return get_default_literal_value(arg.dtype)
        else:
            return super().__new__(cls)


#==============================================================================
class PythonBool(PyccelAstNode):
    """ Represents a call to Python's native bool() function.
    """
    __slots__ = ('_arg',)
    _dtype = NativeBool()
    _precision = default_precision['bool']
    _rank = 0
    _shape = ()
    _order = None
    _attribute_nodes = ('_arg',)

    def __new__(cls, arg):
        if getattr(arg, 'is_optional', None):
            bool_expr = super().__new__(cls)
            bool_expr.__init__(arg)
            return PyccelAnd(PyccelIsNot(arg, Nil()), bool_expr)
        else:
            return super().__new__(cls)

    def __init__(self, arg):
        self._arg = arg
        super().__init__()

    @property
    def arg(self):
        return self._arg

    def __str__(self):
        return 'Bool({})'.format(str(self.arg))

#==============================================================================
class PythonComplex(PyccelAstNode):
    """ Represents a call to Python's native complex() function.
    """
    __slots__ = ('_real_part', '_imag_part', '_internal_var', '_is_cast')

    _dtype = NativeComplex()
    _precision = default_precision['complex']
    _rank = 0
    _shape = ()
    _order = None
    _real_cast = PythonReal
    _imag_cast = PythonImag
    _attribute_nodes = ('_real_part', '_imag_part', '_internal_var')

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

        if arg0.dtype is NativeComplex() and arg1.dtype is NativeComplex():
            # both args are complex
            return PyccelAdd(arg0, PyccelMul(arg1, LiteralImaginaryUnit()))
        return super().__new__(cls)

    def __init__(self, arg0, arg1 = LiteralFloat(0)):
        self._is_cast = arg0.dtype is NativeComplex() and \
                        isinstance(arg1, Literal) and arg1.python_value == 0

        if self._is_cast:
            self._real_part = self._real_cast(arg0)
            self._imag_part = self._imag_cast(arg0)
            self._internal_var = arg0

        else:
            self._internal_var = None

            if arg0.dtype is NativeComplex() and \
                    not (isinstance(arg1, Literal) and arg1.python_value == 0):
                # first arg is complex. Second arg is non-0
                self._real_part = self._real_cast(arg0)
                self._imag_part = PyccelAdd(self._imag_cast(arg0), arg1)
            elif arg1.dtype is NativeComplex():
                if isinstance(arg0, Literal) and arg0.python_value == 0:
                    # second arg is complex. First arg is 0
                    self._real_part = PyccelUnarySub(self._imag_cast(arg1))
                    self._imag_part = self._real_cast(arg1)
                else:
                    # Second arg is complex. First arg is non-0
                    self._real_part = PyccelMinus(arg0, self._imag_cast(arg1))
                    self._imag_part = self._real_cast(arg1)
            else:
                self._real_part = self._real_cast(arg0)
                self._imag_part = self._real_cast(arg1)
        super().__init__()

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
        return "complex({}, {})".format(str(self.real), str(self.imag))

#==============================================================================
class PythonEnumerate(Basic):

    """
    Represents the enumerate stmt

    """
    __slots__ = ('_element',)
    _attribute_nodes = ('_element',)

    def __init__(self, arg):
        if PyccelAstNode.stage != "syntactic" and \
                not isinstance(arg, PyccelAstNode):
            raise TypeError('Expecting an arg of valid type')
        self._element = arg
        super().__init__()

    @property
    def element(self):
        return self._element

#==============================================================================
class PythonFloat(PyccelAstNode):
    """ Represents a call to Python's native float() function.
    """
    __slots__ = ('_arg')
    _dtype = NativeReal()
    _precision = default_precision['real']
    _rank = 0
    _shape = ()
    _order = None
    _attribute_nodes = ('_arg',)

    def __new__(cls, arg):
        if isinstance(arg, LiteralFloat) and arg.precision == cls._precision:
            return arg
        if isinstance(arg, (LiteralInteger, LiteralFloat)):
            return LiteralFloat(arg.python_value, precision = cls._precision)
        return super().__new__(cls)

    def __init__(self, arg):
        self._arg = arg
        super().__init__()

    @property
    def arg(self):
        return self._arg

    def __str__(self):
        return 'float({0})'.format(str(self.arg))

#==============================================================================
class PythonInt(PyccelAstNode):
    """ Represents a call to Python's native int() function.
    """

    __slots__ = ('_arg')
    _dtype = NativeInteger()
    _precision = default_precision['integer']
    _rank = 0
    _shape = ()
    _order = None
    _attribute_nodes  = ('_arg',)

    def __new__(cls, arg):
        if isinstance(arg, LiteralInteger):
            return LiteralInteger(arg.python_value, precision = cls._precision)
        else:
            return super().__new__(cls)

    def __init__(self, arg):
        self._arg = arg
        super().__init__()

    @property
    def arg(self):
        return self._arg

#==============================================================================
class PythonTuple(PyccelAstNode):
    """ Represents a call to Python's native tuple() function.
    """
    __slots__ = ('_args','_inconsistent_shape','_is_homogeneous','_dtype','_precision','_rank','_shape')
    _iterable        = True
    _attribute_nodes = ('_args',)
    _order = 'C'

    def __init__(self, *args):
        self._args = args
        super().__init__()
        if self.stage == 'syntactic':
            return
        elif len(args) == 0:
            self._dtype = NativeGeneric()
            self._precision = 0
            self._rank = 0
            self._shape = ()
            self._is_homogeneous = False
            return
        arg0 = args[0]
        is_homogeneous = arg0.dtype is not NativeGeneric() and \
                         all(a.dtype is not NativeGeneric() and \
                             arg0.dtype == a.dtype and \
                             arg0.rank  == a.rank  and \
                             arg0.order == a.order for a in args[1:])
        self._inconsistent_shape = not all(arg0.shape==a.shape   for a in args[1:])
        self._is_homogeneous = is_homogeneous
        if is_homogeneous:
            integers  = [a for a in args if a.dtype is NativeInteger()]
            reals     = [a for a in args if a.dtype is NativeReal()]
            complexes = [a for a in args if a.dtype is NativeComplex()]
            bools     = [a for a in args if a.dtype is NativeBool()]
            strs      = [a for a in args if a.dtype is NativeString()]
            if strs:
                self._dtype = NativeString()
                self._precision = 0
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
                else:
                    raise TypeError('cannot determine the type of {}'.format(self))


                shapes     = [a.shape for a in args]
                self._rank = max(a.rank for a in args) + 1
                if all(sh is not None for sh in shapes):
                    self._shape = (LiteralInteger(len(args)), ) + shapes[0]
                    self._rank  = len(self._shape)

        else:
            self._rank      = max(a.rank for a in args) + 1
            self._dtype     = NativeGeneric()
            self._precision = 0
            self._shape     = (LiteralInteger(len(args)), ) + args[0].shape

    def __getitem__(self,i):
        return self._args[i]

    def __add__(self,other):
        return PythonTuple(*(self._args + other._args))

    def __iter__(self):
        return self._args.__iter__()

    def __len__(self):
        return len(self._args)

    def __str__(self):
        return '({})'.format(', '.join(str(a) for a in self))

    def __repr__(self):
        return 'PythonTuple({})'.format(', '.join(str(a) for a in self))

    @property
    def is_homogeneous(self):
        return self._is_homogeneous

    @property
    def inconsistent_shape(self):
        return self._inconsistent_shape

    @property
    def args(self):
        """ Arguments of the tuple
        """
        return self._args

#==============================================================================
class PythonLen(PyccelInternalFunction):

    """
    Represents a 'len' expression in the code.
    """

    __slots__ = ()
    _dtype     = NativeInteger()
    _precision = default_precision['int']
    _rank      = 0
    _shape     = ()
    _order     = None

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def arg(self):
        return self._args[0]

    def __str__(self):
        return 'len({})'.format(str(self.arg))

#==============================================================================
class PythonList(PythonTuple):
    """ Represents a call to Python's native list() function.
    """
    __slots__ = ()

#==============================================================================
class PythonMap(Basic):
    """ Represents the map stmt
    """
    __slots__ = ('_args',)
    _attribute_nodes = ('_args',)

    def __init__(self, *args):
        if len(args)<2:
            raise TypeError('wrong number of arguments')
        self._args = args
        super().__init__()

    @property
    def args(self):
        """ Arguments of the map
        """
        return self._args

#==============================================================================
class PythonPrint(Basic):

    """Represents a print function in the code.

    expr : PyccelAstNode
        The expression to print

    Examples

    >>> from pyccel.ast.internals import symbols
    >>> from pyccel.ast.core import Print
    >>> n,m = symbols('n,m')
    >>> Print(('results', n,m))
    Print((results, n, m))
    """
    __slots__ = ('_expr')
    _attribute_nodes = ('_expr',)

    def __init__(self, expr):
        self._expr = expr
        super().__init__()

    @property
    def expr(self):
        return self._expr

#==============================================================================
class PythonRange(Basic):

    """
    Represents a range.

    Examples

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Range
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> s = Variable('int', 's')
    >>> e = PyccelSymbol('e')
    >>> Range(s, e, 1)
    Range(0, n, 1)
    """
    __slots__ = ('_start','_stop','_step')
    _attribute_nodes = ('_start', '_stop', '_step')

    def __init__(self, *args):
        # Define default values
        n = len(args)

        if n == 1:
            self._start = LiteralInteger(0)
            self._stop  = args[0]
            self._step  = LiteralInteger(1)
        elif n == 2:
            self._start = args[0]
            self._stop  = args[1]
            self._step  = LiteralInteger(1)
        elif n == 3:
            self._start = args[0]
            self._stop  = args[1]
            self._step  = args[2]
        else:
            raise ValueError('Range has at most 3 arguments')

        super().__init__()

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step


#==============================================================================
class PythonZip(PyccelInternalFunction):

    """
    Represents a zip stmt.

    """
    __slots__ = ('_length','_args')
    _attribute_nodes = ('_args',)

    def __init__(self, *args):
        if not isinstance(args, (tuple, list)):
            raise TypeError('args must be a list or tuple')
        elif len(args) < 2:
            raise ValueError('args must be of length > 2')
        super().__init__(*args)
        if PyccelAstNode.stage == 'syntactic':
            self._length = None
            return
        else:
            lengths = [a.shape[0].python_value for a in self.args if isinstance(a.shape[0], LiteralInteger)]
            if lengths:
                self._length = min(lengths)
            else:
                self._length = self.args[0].shape[0]

    @property
    def length(self):
        """ Length of the shortest zip argument
        """
        return self._length

#==============================================================================
class PythonAbs(PyccelInternalFunction):
    """Represents a call to  python abs for code generation.

    arg : Variable
    """
    __slots__ = ('_dtype','_precision','_rank','_shape','_order')
    def __init__(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = NativeInteger() if x.dtype is NativeInteger() else NativeReal()
        self._precision = default_precision[str_dtype(self._dtype)]
        self._order     = x.order
        super().__init__(x)

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonSum(PyccelInternalFunction):
    """Represents a call to  python sum for code generation.

    arg : list , tuple , PythonTuple, List, Variable
    """
    __slots__ = ('_dtype','_precision')
    _rank  = 0
    _shape = ()
    _order = None

    def __init__(self, arg):
        if not isinstance(arg, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(arg))
        self._dtype = arg.dtype
        self._precision = default_precision[str_dtype(self._dtype)]
        super().__init__(arg)

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonMax(PyccelInternalFunction):
    """Represents a call to  python max for code generation.

    arg : list , tuple , PythonTuple, List
    """
    __slots__ = ('_dtype','_precision')
    _rank  = 0
    _shape = ()
    _order = None

    def __init__(self, *x):
        if len(x)==1:
            x = x[0]

        if isinstance(x, (list, tuple)):
            x = PythonTuple(*x)
        elif not isinstance(x, (PythonTuple, PythonList)):
            raise TypeError('Unknown type of  %s.' % type(x))
        if not x.is_homogeneous:
            raise NotImplementedError("Cannot determine dtype of max call with inhomogeneous arguments : {}".format(x))
        self._dtype     = x.dtype
        self._precision = x.precision
        super().__init__(x)


#==============================================================================
class PythonMin(PyccelInternalFunction):
    """Represents a call to  python min for code generation.

    arg : list , tuple , PythonTuple, List, Variable
    """
    __slots__ = ('_dtype','_precision')
    _rank  = 0
    _shape = ()
    _order = None
    def __init__(self, *x):
        if len(x)==1:
            x = x[0]

        if isinstance(x, (list, tuple)):
            x = PythonTuple(*x)
        elif not isinstance(x, (PythonTuple, PythonList)):
            raise TypeError('Unknown type of  %s.' % type(x))
        if not x.is_homogeneous:
            raise NotImplementedError("Cannot determine dtype of min call with inhomogeneous arguments : {}".format(x))
        self._dtype     = x.dtype
        self._precision = x.precision
        super().__init__(x)

#==============================================================================
class Lambda(Basic):
    """Represents a call to python lambda for temporary functions

    Parameters
    ==========
    variables : tuple of symbols
                The arguments to the lambda expression
    expr      : PyccelAstNode
                The expression carried out when the lambda function is called
    """
    __slots__ = ('_variables', '_expr')
    _attribute_nodes = ('_variables', '_expr')
    def __init__(self, variables, expr):
        if not isinstance(variables, (list, tuple)):
            raise TypeError("Lambda arguments must be a tuple or list")
        self._variables = tuple(variables)
        self._expr = expr
        super().__init__()

    @property
    def variables(self):
        """ The arguments to the lambda function
        """
        return self._variables

    @property
    def expr(self):
        """ The expression carried out when the lambda function is called
        """
        return self._expr

    def __call__(self, *args):
        """ Returns the expression with the arguments replaced with
        the calling arguments
        """
        assert(len(args) == len(self.variables))
        return self.expr.subs(self.variables, args)

    def __str__(self):
        return "{args} -> {expr}".format(args=self.variables,
                expr = self.expr)

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
    'not'      : PyccelNot,
}

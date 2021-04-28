#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# TODO [EB 12.03.21]: Remove pylint command with PR #797
# pylint: disable=W0201
"""
Module handling all python builtin operators
These operators all have a precision as detailed here:
    https://docs.python.org/3/reference/expressions.html#operator-precedence
They also have specific rules to determine the dtype, precision, rank, shape
"""

from ..errors.errors        import Errors, PyccelSemanticError

from .basic                 import PyccelAstNode

from .datatypes             import (NativeBool, NativeInteger, NativeReal,
                                    NativeComplex, NativeString, default_precision,
                                    NativeNumeric)

from .literals              import Literal, LiteralInteger, LiteralFloat, LiteralComplex, Nil
from .literals              import convert_to_literal

errors = Errors()

__all__ = (
    'PyccelOperator',
    'PyccelPow',
    'PyccelAdd',
    'PyccelMinus',
    'PyccelMul',
    'PyccelDiv',
    'PyccelMod',
    'PyccelFloorDiv',
    'PyccelEq',
    'PyccelNe',
    'PyccelLt',
    'PyccelLe',
    'PyccelGt',
    'PyccelGe',
    'PyccelAnd',
    'PyccelOr',
    'PyccelNot',
    'PyccelAssociativeParenthesis',
    'PyccelUnary',
    'PyccelUnarySub',
    'Relational',
    'PyccelIs',
    'PyccelIsNot',
    'IfTernaryOperator'
)

#==============================================================================
def broadcast(shape_1, shape_2):
    """ This function broadcast two shapes using numpy broadcasting rules """

    from pyccel.ast.sympy_helper import pyccel_to_sympy

    a = len(shape_1)
    b = len(shape_2)
    if a>b:
        new_shape_2 = (LiteralInteger(1),)*(a-b) + tuple(shape_2)
        new_shape_1 = shape_1
    elif b>a:
        new_shape_1 = (LiteralInteger(1),)*(b-a) + tuple(shape_1)
        new_shape_2 = shape_2
    else:
        new_shape_2 = shape_2
        new_shape_1 = shape_1

    new_shape = []
    for e1,e2 in zip(new_shape_1, new_shape_2):
        used_names = set()
        symbol_map = {}
        sy_e1 = pyccel_to_sympy(e1, symbol_map, used_names)
        sy_e2 = pyccel_to_sympy(e2, symbol_map, used_names)
        if sy_e1 == sy_e2:
            new_shape.append(e1)
        elif sy_e1 == 1:
            new_shape.append(e2)
        elif sy_e2 == 1:
            new_shape.append(e1)
        elif sy_e1.is_constant() and not sy_e2.is_constant():
            new_shape.append(e1)
        elif sy_e2.is_constant() and not sy_e1.is_constant():
            new_shape.append(e2)
        elif not sy_e2.is_constant() and not sy_e1.is_constant()\
                and not (sy_e1 - sy_e2).is_constant():
            new_shape.append(e1)
        else:
            shape1_code = '({})'.format(' '.join([str(s)+',' for s in shape_1]))
            shape2_code = '({})'.format(' '.join([str(s)+',' for s in shape_2]))
            msg = 'operands could not be broadcast together with shapes {} {}'
            msg = msg.format(shape1_code, shape2_code)
            raise PyccelSemanticError(msg)
    return tuple(new_shape)

#==============================================================================

class PyccelOperator(PyccelAstNode):
    """
    Abstract superclass for all builtin operators.
    The __init__ function is common
    but the functions called by __init__ are specialised

    Parameters
    ----------
    args: tuple
        The arguments passed to the operator
    """
    __slots__ = ('_args', )
    _attribute_nodes = ('_args',)

    def __init__(self, *args):
        self._args = tuple(self._handle_precedence(args))

        if self.stage == 'syntactic':
            super().__init__()
            return
        self._set_dtype()
        self._set_shape_rank()
        # rank is None for lambda functions
        self._set_order()
        super().__init__()

    def _set_dtype(self):
        self._dtype, self._precision = self._calculate_dtype(*self._args)  # pylint: disable=no-member

    def _set_shape_rank(self):
        self._shape, self._rank = self._calculate_shape_rank(*self._args)  # pylint: disable=no-member

    @property
    def precedence(self):
        """ The precedence of the operator as defined here:
            https://docs.python.org/3/reference/expressions.html#operator-precedence
        """
        return self._precedence

    def _handle_precedence(self, args):
        """
        Insert parentheses where necessary by examining the precedence of the operator
        e.g:
            PyccelMul(a,PyccelAdd(b,c))
        means:
            a*(b+c)
        so this input will give:
            PyccelMul(a, PyccelAssociativeParenthesis(PyccelAdd(b,c)))

        Parentheses are also added were they are required for clarity

        Parameters
        ----------
        args: tuple
            The arguments passed to the operator

        Results
        -------
        args: tuple
            The arguments with the parentheses inserted
        """
        precedence = [getattr(a, 'precedence', 17) for a in args]

        if min(precedence) <= self._precedence:

            new_args = []

            for i, (a,p) in enumerate(zip(args, precedence)):
                if (p < self._precedence or (p == self._precedence and i != 0)):
                    new_args.append(PyccelAssociativeParenthesis(a))
                else:
                    new_args.append(a)
            args = tuple(new_args)

        return args

    def __str__(self):
        return repr(self)

    def _set_order(self):
        """ Sets the shape and rank
        This is chosen to match the arguments if they are in agreement.
        Otherwise it defaults to 'C'
        """
        if self._rank is not None and self._rank > 1:
            if all(a.order == self._args[0].order for a in self._args):
                self._order = self._args[0].order
            else:
                self._order = 'C'
        else:
            self._order = None

    @property
    def args(self):
        """ Arguments of the operator
        """
        return self._args

#==============================================================================

class PyccelUnaryOperator(PyccelOperator):
    """ Abstract superclass representing a python
    operator with only one argument

    Parameters
    ----------
    arg: PyccelAstNode
        The argument passed to the operator
    """
    __slots__ = ('_dtype', '_precision','_shape','_rank','_order')

    def __init__(self, arg):
        super().__init__(arg)

    @staticmethod
    def _calculate_dtype(*args):
        """ Sets the dtype and precision
        They are chosen to match the argument
        """
        a = args[0]
        dtype = a.dtype
        precision = a.precision
        return dtype, precision

    @staticmethod
    def _calculate_shape_rank(*args):
        """ Sets the shape and rank
        They are chosen to match the argument
        """
        a = args[0]
        rank = a.rank
        shape = a.shape
        return shape, rank

#==============================================================================

class PyccelUnary(PyccelUnaryOperator):
    """
    Class representing a call to the python positive operator.
    I.e:
        +a
    is equivalent to:
        PyccelUnary(a)

    Parameters
    ----------
    arg: PyccelAstNode
        The argument passed to the operator
    """
    __slots__ = ()
    _precedence = 14
    def _handle_precedence(self, args):
        args = PyccelUnaryOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelUnary) else a for a in args)
        return args

    def __repr__(self):
        return '+{}'.format(repr(self.args[0]))

#==============================================================================

class PyccelUnarySub(PyccelUnary):
    """
    Class representing a call to the python negative operator.
    I.e:
        -a
    is equivalent to:
        PyccelUnarySub(a)

    Parameters
    ----------
    arg: PyccelAstNode
        The argument passed to the operator
    """
    __slots__ = ()

    def __repr__(self):
        return '-{}'.format(repr(self.args[0]))

#==============================================================================

class PyccelNot(PyccelUnaryOperator):
    """
    Class representing a call to the python not operator.
    I.e:
        not a
    is equivalent to:
        PyccelNot(a)

    Parameters
    ----------
    arg: PyccelAstNode
        The argument passed to the operator
    """
    __slots__ = ()
    _precedence = 6

    @staticmethod
    def _calculate_dtype(*args):
        """ Sets the dtype and precision
        They are chosen to match the argument unless the class has
        a _dtype or _precision member
        """
        dtype = NativeBool()
        precision = default_precision['bool']
        return dtype, precision

    @staticmethod
    def _calculate_shape_rank(*args):
        """ Sets the shape and rank
        They are chosen to match the argument unless the class has
        a _shape or _rank member
        """
        rank = 0
        shape = ()
        return shape, rank

    def __repr__(self):
        return 'not {}'.format(repr(self.args[0]))

#==============================================================================

class PyccelAssociativeParenthesis(PyccelUnaryOperator):
    """
    Class representing parentheses

    Parameters
    ----------
    arg: PyccelAstNode
        The argument in the PyccelAssociativeParenthesis
    """
    __slots__ = () # ok
    _precedence = 18
    def _handle_precedence(self, args):
        return args

    def __repr__(self):
        return '({})'.format(repr(self.args[0]))

#==============================================================================

class PyccelBinaryOperator(PyccelOperator):
    """ Abstract superclass representing a python
    operator with two arguments

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ('_dtype','_precision','_shape','_rank','_order')

    def __init__(self, arg1, arg2, simplify = False):
        super().__init__(arg1, arg2)

    @classmethod
    def _calculate_dtype(cls, *args):
        """ Sets the dtype and precision

        If one argument is a string then all arguments must be strings

        If the arguments are numeric then the dtype and precision
        match the broadest type and the largest precision
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex
        """
        integers  = [a for a in args if a.dtype in (NativeInteger(),NativeBool())]
        reals     = [a for a in args if a.dtype is NativeReal()]
        complexes = [a for a in args if a.dtype is NativeComplex()]
        strs      = [a for a in args if a.dtype is NativeString()]

        if strs:
            return cls._handle_str_type(strs)
            assert len(integers + reals + complexes) == 0
        elif complexes:
            return cls._handle_complex_type(complexes)
        elif reals:
            return cls._handle_real_type(reals)
        elif integers:
            return cls._handle_integer_type(integers)
        else:
            raise TypeError('cannot determine the type of {}'.format(args))

    @staticmethod
    def _handle_str_type(strs):
        """
        Set dtype and precision when both arguments are strings
        """
        raise TypeError("unsupported operand type(s) for /: 'str' and 'str'")

    @staticmethod
    def _handle_complex_type(complexes):
        """
        Set dtype and precision when the result is complex
        """
        dtype = NativeComplex()
        precision = max(a.precision for a in complexes)
        return dtype, precision

    @staticmethod
    def _handle_real_type(reals):
        """
        Set dtype and precision when the result is real
        """
        dtype = NativeReal()
        precision = max(a.precision for a in reals)
        return dtype, precision

    @staticmethod
    def _handle_integer_type(integers):
        """
        Set dtype and precision when the result is integer
        """
        dtype = NativeInteger()
        precision = max(a.precision for a in integers)
        return dtype, precision

    @staticmethod
    def _calculate_shape_rank(*args):
        """ Sets the shape and rank

        Strings must be scalars.

        For numeric types the rank and shape is determined according
        to numpy broadcasting rules where possible
        """
        strs = [a for a in args if a.dtype is NativeString()]
        if strs:
            other = [a for a in args if a.dtype in (NativeInteger(), NativeBool(), NativeReal(), NativeComplex())]
            assert len(other) == 0
            rank  = 0
            shape = ()
        else:
            ranks  = [a.rank for a in args]
            shapes = [a.shape for a in args]

            if None in ranks:
                rank  = None
                shape = None

            elif all(sh is not None for tup in shapes for sh in tup):
                s = broadcast(args[0].shape, args[1].shape)

                shape = s
                rank  = len(s)
            else:
                rank  = max(a.rank for a in args)
                shape = [None]*rank
        return shape, rank

#==============================================================================

class PyccelArithmeticOperator(PyccelBinaryOperator):
    """ Abstract superclass representing a python
    arithmetic operator

    This class is necessary to handle specific precedence
    rules for arithmetic operators
    I.e. to handle the error:
    Extension: Unary operator following arithmetic operator (use parentheses)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    def _handle_precedence(self, args):
        args = PyccelBinaryOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelUnary) else a for a in args)
        return args

#==============================================================================

class PyccelPow(PyccelArithmeticOperator):
    """
    Class representing a call to the python exponent operator.
    I.e:
        a ** b
    is equivalent to:
        PyccelPow(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence  = 15

    def __repr__(self):
        return '{} ** {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelAdd(PyccelArithmeticOperator):
    """
    Class representing a call to the python addition operator.
    I.e:
        a + b
    is equivalent to:
        PyccelAdd(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 12

    def __new__(cls, arg1, arg2, simplify = False):
        if simplify:
            if isinstance(arg2, PyccelUnarySub):
                return PyccelMinus(arg1, arg2.args[0], simplify = True)
            elif isinstance(arg1, Literal) and isinstance(arg2, Literal):
                dtype, precision = cls._calculate_dtype(arg1, arg2)
                return convert_to_literal(arg1.python_value + arg2.python_value,
                                          dtype, precision)

        if isinstance(arg1, (LiteralInteger, LiteralFloat)) and \
            isinstance(arg2, LiteralComplex) and \
            arg2.real == LiteralFloat(0):
            return LiteralComplex(arg1, arg2.imag)
        elif isinstance(arg2, (LiteralInteger, LiteralFloat)) and \
            isinstance(arg1, LiteralComplex) and \
            arg1.real == LiteralFloat(0):
            return LiteralComplex(arg2, arg1.imag)
        else:
            return super().__new__(cls)

    @staticmethod
    def _handle_str_type(strs):
        dtype = NativeString()
        precision = None
        return dtype, precision

    def __repr__(self):
        return '{} + {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelMul(PyccelArithmeticOperator):
    """
    Class representing a call to the python multiplication operator.
    I.e:
        a * b
    is equivalent to:
        PyccelMul(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 13

    def __new__(cls, arg1, arg2, simplify = False):
        if simplify:
            if (arg1 == 1):
                return arg2
            if (arg2 == 1):
                return arg1
            if (arg1 == 0 or arg2 == 0):
                dtype, precision = cls._calculate_dtype(arg1, arg2)
                return convert_to_literal(0, dtype, precision)
            if (isinstance(arg1, PyccelUnarySub) and arg1.args[0] == 1):
                return PyccelUnarySub(arg2)
            if (isinstance(arg2, PyccelUnarySub) and arg2.args[0] == 1):
                return PyccelUnarySub(arg1)
            if isinstance(arg1, Literal) and isinstance(arg2, Literal):
                dtype, precision = cls._calculate_dtype(arg1, arg2)
                return convert_to_literal(arg1.python_value * arg2.python_value,
                                          dtype, precision)
        return super().__new__(cls)

    def __repr__(self):
        return '{} * {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelMinus(PyccelArithmeticOperator):
    """
    Class representing a call to the python subtraction operator.
    I.e:
        a - b
    is equivalent to:
        PyccelMinus(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 12

    def __new__(cls, arg1, arg2, simplify = False):
        if simplify:
            if isinstance(arg2, PyccelUnarySub):
                return PyccelAdd(arg1, arg2.args[0], simplify = True)
            elif isinstance(arg1, Literal) and isinstance(arg2, Literal):
                dtype, precision = cls._calculate_dtype(arg1, arg2)
                return convert_to_literal(arg1.python_value - arg2.python_value,
                                          dtype, precision)
        if isinstance(arg1, LiteralFloat) and \
            isinstance(arg2, LiteralComplex) and \
            arg2.real == LiteralFloat(0):
            return LiteralComplex(arg1, -arg2.imag.python_value)
        elif isinstance(arg2, LiteralFloat) and \
            isinstance(arg1, LiteralComplex) and \
            arg1.real == LiteralFloat(0):
            return LiteralComplex(-arg2.python_value, arg1.imag)
        else:
            return super().__new__(cls)

    def __repr__(self):
        return '{} - {}'.format(repr(self.args[0]), repr(self.args[1]))

#==============================================================================

class PyccelDiv(PyccelArithmeticOperator):
    """
    Class representing a call to the python division operator.
    I.e:
        a / b
    is equivalent to:
        PyccelDiv(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 13

    def __new__(cls, arg1, arg2, simplify=False):
        if simplify:
            if (arg2 == 1):
                return arg1
        return super().__new__(cls)

    @staticmethod
    def _handle_integer_type(integers):
        dtype = NativeReal()
        precision = default_precision['real']
        return dtype, precision

    def __repr__(self):
        return '{} + {}'.format(self.args[0], self.args[1])

    def __repr__(self):
        return '{} / {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelMod(PyccelArithmeticOperator):
    """
    Class representing a call to the python modulo operator.
    I.e:
        a % b
    is equivalent to:
        PyccelMod(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 13

    def __repr__(self):
        return '{} % {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelFloorDiv(PyccelArithmeticOperator):
    """
    Class representing a call to the python integer division operator.
    I.e:
        a // b
    is equivalent to:
        PyccelFloorDiv(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 13

    def __repr__(self):
        return '{} // {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelComparisonOperator(PyccelBinaryOperator):
    """ Abstract superclass representing a python
    comparison operator with two arguments

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 7
    @staticmethod
    def _calculate_dtype(*args):
        dtype = NativeBool()
        precision = default_precision['bool']
        return dtype, precision

#==============================================================================

class PyccelEq(PyccelComparisonOperator):
    """
    Class representing a call to the python equality operator.
    I.e:
        a == b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()

    def __repr__(self):
        return '{} == {}'.format(self.args[0], self.args[1])

class PyccelNe(PyccelComparisonOperator):
    """
    Class representing a call to the python inequality operator.
    I.e:
        a != b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()

    def __repr__(self):
        return '{} != {}'.format(self.args[0], self.args[1])

class PyccelLt(PyccelComparisonOperator):
    """
    Class representing a call to the python less than operator.
    I.e:
        a < b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()

    def __repr__(self):
        return '{} < {}'.format(self.args[0], self.args[1])

class PyccelLe(PyccelComparisonOperator):
    """
    Class representing a call to the python less or equal operator.
    I.e:
        a <= b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()

    def __repr__(self):
        return '{} <= {}'.format(self.args[0], self.args[1])

class PyccelGt(PyccelComparisonOperator):
    """
    Class representing a call to the python greater than operator.
    I.e:
        a > b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()

    def __repr__(self):
        return '{} > {}'.format(self.args[0], self.args[1])

class PyccelGe(PyccelComparisonOperator):
    """
    Class representing a call to the python greater or equal operator.
    I.e:
        a >= b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()

    def __repr__(self):
        return '{} >= {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelBooleanOperator(PyccelOperator):
    """ Abstract superclass representing a python
    boolean operator with two arguments

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    dtype = NativeBool()
    precision = default_precision['bool']
    rank = 0
    shape = ()
    order = None

    __slots__ = ()

    def _set_order(self):
        pass

    def _set_dtype(self):
        pass

    def _set_shape_rank(self):
        pass

#==============================================================================

class PyccelAnd(PyccelBooleanOperator):
    """
    Class representing a call to the python AND operator.
    I.e:
        a and b
    is equivalent to:
        PyccelAnd(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 5
    def _handle_precedence(self, args):
        args = PyccelBooleanOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelOr) else a for a in args)
        return args

    def __repr__(self):
        return '{} and {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelOr(PyccelBooleanOperator):
    """
    Class representing a call to the python OR operator.
    I.e:
        a or b
    is equivalent to:
        PyccelOr(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 4
    def _handle_precedence(self, args):
        args = PyccelBooleanOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelAnd) else a for a in args)
        return args

    def __repr__(self):
        return '{} or {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelIs(PyccelBooleanOperator):

    """Represents a is expression in the code.

    Examples
    --------
    >>> from pyccel.ast.operators import PyccelIs
    >>> from pyccel.ast.literals  import Nil
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> x = PyccelSymbol('x')
    >>> PyccelIs(x, Nil())
    PyccelIs(x, None)
    """
    __slots__ = ()
    _precedence = 7

    def __init__(self, arg1, arg2):
        super().__init__(arg1, arg2)

    @property
    def lhs(self):
        """ First operator argument"""
        return self._args[0]

    @property
    def rhs(self):
        """ First operator argument"""
        return self._args[1]

    def __repr__(self):
        return '{} is {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelIsNot(PyccelIs):

    """Represents a is expression in the code.

    Examples
    --------
    >>> from pyccel.ast.operators import PyccelIsNot
    >>> from pyccel.ast.literals  import Nil
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> x = PyccelSymbol('x')
    >>> PyccelIsNot(x, Nil())
    PyccelIsNot(x, None)
    """
    __slots__ = ()

    def __repr__(self):
        return '{} is not {}'.format(self.args[0], self.args[1])


#==============================================================================

class IfTernaryOperator(PyccelOperator):
    """Represent a ternary conditional operator in the code, of the form (a if cond else b)

    Parameters
    ----------
    args :
        args : type list
        format : condition , value_if_true, value_if_false

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import Assign
	>>>	from pyccel.ast.operators import IfTernaryOperator
    >>> n = PyccelSymbol('n')
    >>> x = 5 if n > 1 else 2
    >>> IfTernaryOperator(PyccelGt(n > 1),  5,  2)
    IfTernaryOperator(PyccelGt(n > 1),  5,  2)
    """
    __slots__ = ('_dtype','_precision','_shape','_rank','_order')
    _precedence = 3

    def __init__(self, cond, value_true, value_false):
        super().__init__(cond, value_true, value_false)

        if self.stage == 'syntactic':
            return
        if isinstance(value_true , Nil) or isinstance(value_false, Nil):
            errors.report('None is not implemented for Ternary Operator', severity='fatal')
        if isinstance(value_true , NativeString) or isinstance(value_false, NativeString):
            errors.report('String is not implemented for Ternary Operator', severity='fatal')
        if value_true.dtype != value_false.dtype:
            if value_true.dtype not in NativeNumeric or value_false.dtype not in NativeNumeric:
                errors.report('The types are incompatible in IfTernaryOperator', severity='fatal')
        if value_false.rank != value_true.rank :
            errors.report('Ternary Operator results should have the same rank', severity='fatal')
        if value_false.shape != value_true.shape :
            errors.report('Ternary Operator results should have the same shape', severity='fatal')

    @staticmethod
    def _calculate_dtype(cond, value_true, value_false):
        """
        Sets the dtype and precision for IfTernaryOperator
        """
        if value_true.dtype in NativeNumeric and value_false.dtype in NativeNumeric:
            dtype = max([value_true.dtype, value_false.dtype], key = NativeNumeric.index)
        else:
            dtype = value_true.dtype

        precision = max([value_true.precision, value_false.precision])
        return dtype, precision

    @staticmethod
    def _calculate_shape_rank(cond, value_true, value_false):
        """
        Sets the shape and rank and the order for IfTernaryOperator
        """
        shape = value_true.shape
        rank  = value_true.rank
        if rank is not None and rank > 1:
            if value_false.order != value_true.order :
                errors.report('Ternary Operator results should have the same order', severity='fatal')
        return shape, rank

    @property
    def cond(self):
        """
        The condition property for IfTernaryOperator class
        """
        return self._args[0]

    @property
    def value_true(self):
        """
        The value_if_cond_true property for IfTernaryOperator class
        """
        return self._args[1]

    @property
    def value_false(self):
        """
        The value_if_cond_false property for IfTernaryOperator class
        """
        return self._args[2]



#==============================================================================
Relational = (PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe, PyccelAnd, PyccelOr,  PyccelNot, PyccelIs, PyccelIsNot)


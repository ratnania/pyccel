#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from collections     import OrderedDict

from sympy.logic.boolalg      import And as sp_And


from pyccel.errors.errors import Errors
from pyccel.errors.messages import RECURSIVE_RESULTS_REQUIRED

from pyccel.utilities.metaclasses import Singleton

from .basic     import Basic, PyccelAstNode, iterable
from .builtins  import (PythonEnumerate, PythonLen, PythonMap, PythonTuple,
                        PythonRange, PythonZip, PythonBool, Lambda)
from .datatypes import (datatype, DataType, NativeSymbol,
                        NativeBool, NativeRange,
                        NativeTuple, is_iterable_datatype, str_dtype)
from .internals      import Slice, PyccelSymbol

from .literals       import LiteralInteger, Nil, convert_to_literal
from .itertoolsext   import Product
from .functionalexpr import FunctionalFor

from .operators import PyccelAdd, PyccelMinus, PyccelMul, PyccelDiv, PyccelMod, Relational

from .variable import DottedName, DottedVariable, IndexedElement
from .variable import ValuedVariable, Variable

errors = Errors()

# TODO [YG, 12.03.2020]: Move non-Python constructs to other modules
# TODO [YG, 12.03.2020]: Rename classes to avoid name clashes in pyccel/ast
__all__ = (
    'AliasAssign',
    'Allocate',
    'AnnotatedComment',
    'Argument',
    'AsName',
    'Assert',
    'Assign',
    'AugAssign',
    'Block',
    'Break',
    'ClassDef',
    'CodeBlock',
    'Comment',
    'CommentBlock',
    'ConstructorCall',
    'Continue',
    'Deallocate',
    'Declare',
    'Del',
    'Duplicate',
    'DoConcurrent',
    'EmptyNode',
    'BindCFunctionDef',
    'For',
    'ForIterator',
    'FunctionCall',
    'FunctionDef',
    'If',
    'Import',
    'Interface',
    'Module',
    'ModuleHeader',
    'ParserResult',
    'Pass',
    'Program',
    'PythonFunction',
    'Return',
    'SeparatorComment',
    'StarredArguments',
    'SymbolicAssign',
    'SymbolicPrint',
    'SympyFunction',
    'ValuedArgument',
    'While',
    'With',
    'create_variable',
    'create_incremented_string',
    'get_initial_value',
    'get_iterable_ranges',
    'inline',
    'process_shape',
    'subs'
)

#==============================================================================

# TODO - add EmptyStmt => empty lines
#      - update code examples
#      - add examples
#      - Function case
#      - AnnotatedComment case
#      - add a new Idx that uses Variable instead of Symbol

#==============================================================================
def apply(func, args, kwargs):return func(*args, **kwargs)

#==============================================================================
def subs(expr, new_elements):
    """
    Substitutes old for new in an expression.

    Parameters
    ----------
    new_elements : list of tuples like [(x,2)(y,3)]
    """

    if len(list(new_elements)) == 0:
        return expr
    if isinstance(expr, (list, tuple)):
        return [subs(i, new_elements) for i in expr]

    elif isinstance(expr, While):
        test = subs(expr.test, new_elements)
        body = subs(expr.body, new_elements)
        return While(test, body)

    elif isinstance(expr, For):
        target = subs(expr.target, new_elements)
        it = subs(expr.iterable, new_elements)
        target = expr.target
        it = expr.iterable
        body = subs(expr.body, new_elements)
        return For(target, it, body)

    elif isinstance(expr, If):
        args = []
        for block in expr.args:
            test = block[0]
            stmts = block[1]
            t = subs(test, new_elements)
            s = subs(stmts, new_elements)
            args.append((t, s))
        return If(*args)

    elif isinstance(expr, Return):

        for i in new_elements:
            expr = expr.subs(i[0],i[1])
        return expr

    elif isinstance(expr, Assign):
        new_expr = expr.subs(new_elements)
        new_expr.set_fst(expr.fst)
        return new_expr
    elif isinstance(expr, PyccelAstNode):
        return expr.subs(new_elements)

    else:
        return expr

def inline(func, args):
    local_vars = func.local_vars
    body = func.body
    body = subs(body, zip(func.arguments, args))
    return Block(str(func.name), local_vars, body)

def create_incremented_string(forbidden_exprs, prefix = 'Dummy', counter = 1):
    """This function takes a prefix and a counter and uses them to construct
    a new name of the form:
            prefix_counter
    Where counter is formatted to fill 4 characters
    The new name is checked against a list of forbidden expressions. If the
    constructed name is forbidden then the counter is incremented until a valid
    name is found

      Parameters
      ----------
      forbidden_exprs : Set
                        A set of all the values which are not valid solutions to this problem
      prefix          : str
                        The prefix used to begin the string
      counter         : int
                        The expected value of the next name

      Returns
      ----------
      name            : str
                        The incremented string name
      counter         : int
                        The expected value of the next name

    """
    assert(isinstance(forbidden_exprs, set))
    nDigits = 4

    if prefix is None:
        prefix = 'Dummy'

    name_format = "{prefix}_{counter:0="+str(nDigits)+"d}"
    name = name_format.format(prefix=prefix, counter = counter)
    counter += 1
    while name in forbidden_exprs:
        name = name_format.format(prefix=prefix, counter = counter)
        counter += 1

    forbidden_exprs.add(name)

    return name, counter

def create_variable(forbidden_names, prefix = None, counter = 1):
    """This function takes a prefix and a counter and uses them to construct
    a PyccelSymbol with a name of the form:
            prefix_counter
    Where counter is formatted to fill 4 characters
    The new name is checked against a list of forbidden expressions. If the
    constructed name is forbidden then the counter is incremented until a valid
    name is found

      Parameters
      ----------
      forbidden_exprs : Set
                        A set of all the values which are not valid solutions to this problem
      prefix          : str
                        The prefix used to begin the string
      counter         : int
                        The expected value of the next name

      Returns
      ----------
      name            : PyccelSymbol
                        A PyccelSymbol with the incremented string name
      counter         : int
                        The expected value of the next name

    """

    name, counter = create_incremented_string(forbidden_names, prefix, counter = counter)

    return PyccelSymbol(name, is_temp=True), counter


class AsName(Basic):

    """
    Represents a renaming of a variable, used with Import.

    Examples
    --------
    >>> from pyccel.ast.core import AsName
    >>> AsName('old', 'new')
    old as new

    Parameters
    ==========
    name   : str
             original name of variable or function
    target : str
             name of variable or function in this context
    """
    __slots__ = ('_name', '_target')
    _attribute_nodes = ()

    def __init__(self, name, target):
        self._name = name
        self._target = target
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def target(self):
        return self._target

    def __str__(self):
        return '{0} as {1}'.format(str(self.name), str(self.target))

    def __eq__(self, string):
        if isinstance(string, str):
            return string == self.target
        else:
            return self is string

    def __hash__(self):
        return hash(self.target)


class Duplicate(PyccelAstNode):

    """ this is equivalent to the * operator for python tuples.

    Parameters
    ----------
    value : PyccelAstNode
           an expression which represents the initilized value of the list

    shape : the shape of the array
    """
    __slots__ = ('_val', '_length','_dtype','_precision','_rank','_shape','_order')
    _attribute_nodes = ('_val', '_length')

    def __init__(self, val, length):
        self._dtype     = val.dtype
        self._precision = val.precision
        self._rank      = val.rank
        self._shape     = tuple(s if i!= 0 else PyccelMul(s, length, simplify=True) for i,s in enumerate(val.shape))
        self._order     = val.order

        self._val       = val
        self._length    = length
        super().__init__()

    @property
    def val(self):
        return self._val

    @property
    def length(self):
        return self._length

    def __str__(self):
        return '{} * {}'.format(str(self.val), str(self.length))

    def __repr__(self):
        return '{} * {}'.format(repr(self.val), repr(self.length))

class Concatenate(PyccelAstNode):

    """ this is equivalent to the + operator for python tuples

    Parameters
    ----------
    args : PyccelAstNodes
           The tuples
    """
    __slots__ = ('_args','_dtype','_precision','_rank','_shape','_order')
    _attribute_nodes = ('_args',)

    def __init__(self, arg1, arg2):
        self._dtype     = arg1.dtype
        self._precision = arg1.precision
        self._rank      = arg1.rank
        shape_addition  = arg2.shape[0]
        self._shape     = tuple(s if i!= 0 else PyccelAdd(s, shape_addition) for i,s in enumerate(arg1.shape))
        self._order     = arg1.order

        self._args = (arg1, arg2)
        super().__init__()

    @property
    def args(self):
        return self._args


class Assign(Basic):

    """Represents variable assignment for code generation.

    Parameters
    ----------
    lhs : PyccelAstNode
        In the syntactic stage:
           Object representing the lhs of the expression. These should be
           singular objects, such as one would use in writing code. Notable types
           include PyccelSymbol, and IndexedElement. Types that
           subclass these types are also supported.
        In the semantic stage:
           Variable or IndexedElement

    rhs : PyccelAstNode
        In the syntactic stage:
          Object representing the rhs of the expression
        In the semantic stage :
          PyccelAstNode with the same shape as the lhs

    status: None, str
        if lhs is not allocatable, then status is None.
        otherwise, status is {'allocated', 'unallocated'}

    like: None, Variable
        contains the name of the variable from which the lhs will be cloned.

    Examples
    --------
    >>> from pyccel.ast.internals import symbols
    >>> from pyccel.ast.variable import Variable
    >>> from pyccel.ast.core import Assign
    >>> x, y, z = symbols('x, y, z')
    >>> Assign(x, y)
    x := y
    >>> Assign(x, 0)
    x := 0
    >>> A = Variable('int', 'A', rank = 2)
    >>> Assign(x, A)
    x := A
    >>> Assign(A[0,1], x)
    IndexedElement(A, 0, 1) := x
    """
    __slots__ = ('_lhs', '_rhs', '_status', '_like')
    _attribute_nodes = ('_lhs', '_rhs')

    def __init__(
        self,
        lhs,
        rhs,
        status=None,
        like=None,
        ):
        if isinstance(lhs, (tuple, list)):
            lhs = PythonTuple(*lhs)
        if isinstance(rhs, (tuple, list)):
            rhs = PythonTuple(*rhs)
        self._lhs = lhs
        self._rhs = rhs
        self._status = status
        self._like = like
        super().__init__()

    def __str__(self):
        return '{0} := {1}'.format(str(self.lhs), str(self.rhs))

    def __repr__(self):
        return '({0} := {1})'.format(repr(self.lhs), repr(self.rhs))

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    # TODO : remove

    @property
    def expr(self):
        return self.rhs

    @property
    def status(self):
        return self._status

    @property
    def like(self):
        return self._like

    @property
    def is_alias(self):
        """Returns True if the assignment is an alias."""

        # TODO to be improved when handling classes

        lhs = self.lhs
        rhs = self.rhs
        cond = isinstance(rhs, Variable) and rhs.rank > 0
        cond = cond or isinstance(rhs, IndexedElement)
        cond = cond and isinstance(lhs, PyccelSymbol)
        cond = cond or isinstance(rhs, Variable) and rhs.is_pointer
        return cond

    @property
    def is_symbolic_alias(self):
        """Returns True if the assignment is a symbolic alias."""

        # TODO to be improved when handling classes

        lhs = self.lhs
        rhs = self.rhs
        if isinstance(lhs, Variable):
            return isinstance(lhs.dtype, NativeSymbol)
        elif isinstance(lhs, PyccelSymbol):
            if isinstance(rhs, PythonRange):
                return True
            elif isinstance(rhs, Variable):
                return isinstance(rhs.dtype, NativeSymbol)
            elif isinstance(rhs, PyccelSymbol):
                return True

        return False

#------------------------------------------------------------------------------
class Allocate(Basic):
    """
    Represents memory allocation (usually of an array) for code generation.
    This is relevant to low-level target languages, such as C or Fortran,
    where the programmer must take care of heap memory allocation.

    Parameters
    ----------
    variable : pyccel.ast.core.Variable
        The typed variable (usually an array) that needs memory allocation.

    shape : int or iterable or None
        Shape of the array after allocation (None for scalars).

    order : str {'C'|'F'}
        Ordering of multi-dimensional array after allocation
        ('C' = row-major, 'F' = column-major).

    status : str {'allocated'|'unallocated'|'unknown'}
        Variable allocation status at object creation.

    Notes
    -----
    An object of this class is immutable, although it contains a reference to a
    mutable Variable object.

    """
    __slots__ = ('_variable', '_shape', '_order', '_status')
    _attribute_nodes = ('_variable',)

    # ...
    def __init__(self, variable, *, shape, order, status):

        if not isinstance(variable, Variable):
            raise TypeError("Can only allocate a 'Variable' object, got {} instead".format(type(variable)))

        if not variable.allocatable:
            raise ValueError("Variable must be allocatable")

        if shape and not isinstance(shape, (int, tuple, list)):
            raise TypeError("Cannot understand 'shape' parameter of type '{}'".format(type(shape)))

        if variable.rank != len(shape):
            raise ValueError("Incompatible rank in variable allocation")

        # rank is None for lambda functions
        if variable.rank is not None and variable.rank > 1 and variable.order != order:
            raise ValueError("Incompatible order in variable allocation")

        if not isinstance(status, str):
            raise TypeError("Cannot understand 'status' parameter of type '{}'".format(type(status)))

        if status not in ('allocated', 'unallocated', 'unknown'):
            raise ValueError("Value of 'status' not allowed: '{}'".format(status))

        self._variable = variable
        self._shape    = shape
        self._order    = order
        self._status   = status
        super().__init__()
    # ...

    @property
    def variable(self):
        return self._variable

    @property
    def shape(self):
        return self._shape

    @property
    def order(self):
        return self._order

    @property
    def status(self):
        return self._status

    def __str__(self):
        return 'Allocate({}, shape={}, order={}, status={})'.format(
                str(self.variable), str(self.shape), str(self.order), str(self.status))

    def __eq__(self, other):
        if isinstance(other, Allocate):
            return (self.variable is other.variable) and \
                   (self.shape    == other.shape   ) and \
                   (self.order    == other.order   ) and \
                   (self.status   == other.status  )
        else:
            return False

    def __hash__(self):
        return hash((id(self.variable), self.shape, self.order, self.status))

#------------------------------------------------------------------------------
class Deallocate(Basic):
    """
    Represents memory deallocation (usually of an array) for code generation.
    This is relevant to low-level target languages, such as C or Fortran,
    where the programmer must take care of heap memory deallocation.

    Parameters
    ----------
    variable : pyccel.ast.core.Variable
        The typed variable (usually an array) that needs memory deallocation.

    Notes
    -----
    An object of this class is immutable, although it contains a reference to a
    mutable Variable object.

    """
    __slots__ = ('_variable',)
    _attribute_nodes = ('_variable',)

    # ...
    def __init__(self, variable):

        if not isinstance(variable, Variable):
            raise TypeError("Can only allocate a 'Variable' object, got {} instead".format(type(variable)))

        self._variable = variable
        super().__init__()

    # ...

    @property
    def variable(self):
        return self._variable

    def __eq__(self, other):
        if isinstance(other, Deallocate):
            return (self.variable is other.variable)
        else:
            return False

    def __hash__(self):
        return hash(id(self.variable))

#------------------------------------------------------------------------------
class CodeBlock(Basic):

    """Represents a list of stmt for code generation.
       we use it when a single statement in python
       produce multiple statement in the targeted language

       Parameters
       ==========
       body : iterable
    """
    __slots__ = ('_body','_unravelled')
    _attribute_nodes = ('_body',)


    def __init__(self, body, unravelled = False):
        ls = []
        for i in body:
            if isinstance(i, CodeBlock):
                ls += i.body
            elif i is not None and not isinstance(i, EmptyNode):
                ls.append(i)
        if not isinstance(unravelled, bool):
            raise TypeError("unravelled must be a boolean")
        self._body = tuple(ls)
        self._unravelled = unravelled
        super().__init__()

    @property
    def body(self):
        return self._body

    @property
    def unravelled(self):
        """ Indicates whether the vector syntax of python
        has been unravelled into for loops
        """
        return self._unravelled

    @property
    def lhs(self):
        return self.body[-1].lhs

    def insert2body(self, obj):
        self._body = tuple(self.body + (obj,))

    def __str__(self):
        return 'CodeBlock({})'.format(self.body)

    def __reduce_ex__(self, i):
        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments.
        """
        kwargs = dict(body = self.body)
        return (apply, (self.__class__, (), kwargs))

class AliasAssign(Basic):

    """Represents aliasing for code generation. An alias is any statement of the
    form `lhs := rhs` where

    Parameters
    ----------
    lhs : PyccelSymbol
        at this point we don't know yet all information about lhs, this is why a
        PyccelSymbol is the appropriate type.

    rhs : Variable, IndexedElement
        an assignable variable can be of any rank and any datatype, however its
        shape must be known (not None)

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import AliasAssign
    >>> from pyccel.ast.core import Variable
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x', rank=1, shape=[n])
    >>> y = PyccelSymbol('y')
    >>> AliasAssign(y, x)

    """
    __slots__ = ('_lhs','_rhs')
    _attribute_nodes = ('_lhs','_rhs')

    def __init__(self, lhs, rhs):
        if PyccelAstNode.stage == 'semantic':
            if not lhs.is_pointer:
                raise TypeError('lhs must be a pointer')

            if isinstance(rhs, FunctionCall) and not rhs.funcdef.results[0].is_pointer:
                raise TypeError("A pointer cannot point to the address of a temporary variable")

        self._lhs = lhs
        self._rhs = rhs
        super().__init__()

    def __str__(self):
        return '{0} := {1}'.format(str(self.lhs), str(self.rhs))

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs


class SymbolicAssign(Basic):

    """Represents symbolic aliasing for code generation. An alias is any statement of the
    form `lhs := rhs` where

    Parameters
    ----------
    lhs : PyccelSymbol

    rhs : Range

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import SymbolicAssign
    >>> from pyccel.ast.core import Range
    >>> r = Range(0, 3)
    >>> y = PyccelSymbol('y')
    >>> SymbolicAssign(y, r)

    """
    __slots__ = ('_lhs', '_rhs')
    _attribute_nodes = ('_lhs', '_rhs')

    def __init__(self, lhs, rhs):
        self._lhs = lhs
        self._rhs = rhs
        super().__init__()

    def __str__(self):
        return '{0} := {1}'.format(str(self.lhs), str(self.rhs))

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs


class AugAssign(Assign):
    r"""
    Represents augmented variable assignment for code generation.

    Parameters
    ----------
    lhs : PyccelAstNode
        In the syntactic stage:
           Object representing the lhs of the expression. These should be
           singular objects, such as one would use in writing code. Notable types
           include PyccelSymbol, and IndexedElement. Types that
           subclass these types are also supported.
        In the semantic stage:
           Variable or IndexedElement

    op : str
        Operator (+, -, /, \*, %).

    rhs : PyccelAstNode
        In the syntactic stage:
          Object representing the rhs of the expression
        In the semantic stage :
          PyccelAstNode with the same shape as the lhs

    status: None, str
        if lhs is not allocatable, then status is None.
        otherwise, status is {'allocated', 'unallocated'}

    like: None, Variable
        contains the name of the variable from which the lhs will be cloned.

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import AugAssign
    >>> s = Variable('int', 's')
    >>> t = Variable('int', 't')
    >>> AugAssign(s, '+', 2 * t + 1)
    s += 1 + 2*t
    """
    __slots__ = ('_op',)
    _accepted_operators = {
            '+' : PyccelAdd,
            '-' : PyccelMinus,
            '*' : PyccelMul,
            '/' : PyccelDiv,
            '%' : PyccelMod}

    def __init__(
        self,
        lhs,
        op,
        rhs,
        status=None,
        like=None,
        ):

        if op not in self._accepted_operators.keys():
            raise TypeError('Unrecognized Operator')

        self._op = op

        super().__init__(lhs, rhs, status, like)

    def __str__(self):
        return '{0} {1}= {2}'.format(str(self.lhs), self.op, str(self.rhs))

    @property
    def op(self):
        return self._op

    def to_basic_assign(self):
        """
        Convert the AugAssign to an Assign
        E.g. convert:
        a += b
        to:
        a = a + b
        """
        return Assign(self.lhs,
                self._accepted_operators[self._op](self.lhs, self.rhs),
                status = self.status,
                like   = self.like)


class While(Basic):

    """Represents a 'while' statement in the code.

    Expressions are of the form:
        "while test:
            body..."

    Parameters
    ----------
    test : PyccelAstNode
        test condition given as an expression
    body : list of Pyccel objects
        list of statements representing the body of the While statement.

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import Assign, While
    >>> n = PyccelSymbol('n')
    >>> While((n>1), [Assign(n,n-1)])
    While(n > 1, (n := n - 1,))
    """
    __slots__ = ('_body','_test','_local_vars')
    _attribute_nodes = ('_body','_test','_local_vars')

    def __init__(self, test, body, local_vars=()):

        if PyccelAstNode.stage == 'semantic':
            if test.dtype is not NativeBool():
                test = PythonBool(test)

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')

        self._test = test
        self._body = body
        self._local_vars = local_vars
        super().__init__()

    @property
    def test(self):
        return self._test

    @property
    def body(self):
        return self._body

    @property
    def local_vars(self):
        return self._local_vars


class With(Basic):

    """Represents a 'with' statement in the code.

    Expressions are of the form:
        "while test:
            body..."

    Parameters
    ----------
    test : PyccelAstNode
        test condition given as an expression
    body : list of Pyccel objects
        list of statements representing the body of the With statement.

    Examples
    --------

    """
    __slots__ = ('_test','_body')
    _attribute_nodes = ('_test','_body')

    # TODO check prelude and epilog

    def __init__(
        self,
        test,
        body,
        ):

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable')

        self._test = test
        self._body = body
        super().__init__()

    @property
    def test(self):
        return self._test

    @property
    def body(self):
        return self._body

    @property
    def block(self):
        methods = self.test.cls_base.methods
        for i in methods:
            if str(i.name) == '__enter__':
                start = i
            elif str(i.name) == '__exit__':
                end   = i
        start = inline(start,[])
        end   = inline(end  ,[])

        # TODO check if enter is empty or not first

        body = start.body.body
        body += self.body.body
        body +=  end.body.body
        return Block('with', [], body)


# TODO add a name to a block?

class Block(Basic):

    """Represents a block in the code. A block consists of the following inputs

    Parameters
    ----------
    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    body: list
        a list of statements

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> Block([n, x], [Assign(x,2.*n + 1.), Assign(n, n + 1)])
    Block([n, x], [x := 1.0 + 2.0*n, n := 1 + n])
    """
    __slots__ = ('_name','_variables','_body')
    _attribute_nodes = ('_variables','_body')

    def __init__(
        self,
        name,
        variables,
        body):
        if not isinstance(name, str):
            raise TypeError('name must be of type str')
        if not iterable(variables):
            raise TypeError('variables must be an iterable')
        for var in variables:
            if not isinstance(var, Variable):
                raise TypeError('Only a Variable instance is allowed.')
        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body, CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')
        self._name = name
        self._variables = variables
        self._body = body
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def variables(self):
        return self._variables

    @property
    def body(self):
        return self._body

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]



class Module(Basic):

    """Represents a module in the code. A block consists of the following inputs

    Parameters
    ----------
    name: str
        name of the module

    variables: list
        list of the variables that appear in the block.

    funcs: list
        a list of FunctionDef instances

    interfaces: list
        a list of Interface instances

    classes: list
        a list of ClassDef instances

    imports: list, tuple
        list of needed imports

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.core import ClassDef, FunctionDef, Module
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> z = Variable('real', 'z')
    >>> t = Variable('real', 't')
    >>> a = Variable('real', 'a')
    >>> b = Variable('real', 'b')
    >>> body = [Assign(y,x+a)]
    >>> translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
    >>> attributes   = [x,y]
    >>> methods     = [translate]
    >>> Point = ClassDef('Point', attributes, methods)
    >>> incr = FunctionDef('incr', [x], [y], [Assign(y,x+1)])
    >>> decr = FunctionDef('decr', [x], [y], [Assign(y,x-1)])
    >>> Module('my_module', [], [incr, decr], classes = [Point])
    Module(my_module, [], [FunctionDef(), FunctionDef()], [], [ClassDef(Point, (x, y), (FunctionDef(),), [public], (), [], [])], ())
    """
    __slots__ = ('_name','_variables','_funcs','_interfaces','_classes','_imports')
    _attribute_nodes = ('_variables','_funcs','_interfaces','_classes','_imports')

    def __init__(
        self,
        name,
        variables,
        funcs,
        interfaces=(),
        classes=(),
        imports=(),
        ):
        if not isinstance(name, str):
            raise TypeError('name must be a string')

        if not iterable(variables):
            raise TypeError('variables must be an iterable')
        for i in variables:
            if not isinstance(i, Variable):
                raise TypeError('Only a Variable instance is allowed.')

        if not iterable(funcs):
            raise TypeError('funcs must be an iterable')

        for i in funcs:
            if not isinstance(i, FunctionDef):
                raise TypeError('Only a FunctionDef instance is allowed.'
                                )

        if not iterable(classes):
            raise TypeError('classes must be an iterable')
        for i in classes:
            if not isinstance(i, ClassDef):
                raise TypeError('Only a ClassDef instance is allowed.')

        if not iterable(interfaces):
            raise TypeError('interfaces must be an iterable')
        for i in interfaces:
            if not isinstance(i, Interface):
                raise TypeError('Only a Inteface instance is allowed.')

        if not iterable(imports):
            raise TypeError('imports must be an iterable')
        imports = list(imports)
        for i in classes:
            imports += i.imports
        imports = set(imports)  # for unicity
        imports = tuple(imports)

        self._name = name
        self._variables = variables
        self._funcs = funcs
        self._interfaces = interfaces
        self._classes = classes
        self._imports = imports
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def variables(self):
        return self._variables

    @property
    def funcs(self):
        return self._funcs

    @property
    def interfaces(self):
        return self._interfaces

    @property
    def classes(self):
        return self._classes

    @property
    def imports(self):
        return self._imports

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]

    @property
    def body(self):
        return self.interfaces + self.funcs + self.classes

    def set_name(self, new_name):
        self._name = new_name

class ModuleHeader(Basic):

    """Represents the header file for a module

    Parameters
    ----------
    module: Module
        the module

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.core import ClassDef, FunctionDef, Module
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> z = Variable('real', 'z')
    >>> t = Variable('real', 't')
    >>> a = Variable('real', 'a')
    >>> b = Variable('real', 'b')
    >>> body = [Assign(y,x+a)]
    >>> translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
    >>> attributes   = [x,y]
    >>> methods     = [translate]
    >>> Point = ClassDef('Point', attributes, methods)
    >>> incr = FunctionDef('incr', [x], [y], [Assign(y,x+1)])
    >>> decr = FunctionDef('decr', [x], [y], [Assign(y,x-1)])
    >>> mod = Module('my_module', [], [incr, decr], classes = [Point])
    >>> ModuleHeader(mod)
    Module(my_module, [], [FunctionDef(), FunctionDef()], [], [ClassDef(Point, (x, y), (FunctionDef(),), [public], (), [], [])], ())
    """
    __slots__ = ('_module',)
    _attribute_nodes = ('_module',)

    def __init__(self, module):
        if not isinstance(module, Module):
            raise TypeError('module must be a Module')

        self._module = module
        super().__init__()

    @property
    def module(self):
        return self._module

class Program(Basic):

    """Represents a Program in the code. A block consists of the following inputs

    Parameters
    ----------
    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    body: list
        a list of statements

    imports: list, tuple
        list of needed imports

    """
    __slots__ = ('_name', '_variables', '_body', '_imports')
    _attribute_nodes = ('_variables', '_body', '_imports')

    def __init__(
        self,
        name,
        variables,
        body,
        imports=(),
        ):

        if not isinstance(name, str):
            raise TypeError('name must be a string')

        if not iterable(variables):
            raise TypeError('variables must be an iterable')

        for i in variables:
            if not isinstance(i, Variable):
                raise TypeError('Only a Variable instance is allowed.')

        if not iterable(body):
            raise TypeError('body must be an iterable')
        body = CodeBlock(body)

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        imports = set(imports)  # for unicity
        imports = tuple(imports)

        self._name = name
        self._variables = variables
        self._body = body
        self._imports = imports

        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def variables(self):
        return self._variables

    @property
    def body(self):
        return self._body

    @property
    def imports(self):
        return self._imports

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]


class For(Basic):

    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    Parameters
    ----------
    target : symbol / Variable
        symbol representing the iterator
    iter : iterable
        iterable object. for the moment only Range is used
    body : list of pyccel objects
        list of statements representing the body of the For statement.

    Examples
    --------
    >>> from pyccel.ast.variable import Variable
    >>> from pyccel.ast.core import Assign, For
    >>> from pyccel.ast.internals import symbols
    >>> i,b,e,s,x = symbols('i,b,e,s,x')
    >>> A = Variable('int', 'A', rank = 2)
    >>> For(i, (b,e,s), [Assign(x, i), Assign(A[0, 1], x)])
    For(i, (b, e, s), (x := i, IndexedElement(A, 0, 1) := x))
    """
    __slots__ = ('_target','_iterable','_body','_local_vars','_nowait_expr')
    _attribute_nodes = ('_target','_iterable','_body','_local_vars')

    def __init__(
        self,
        target,
        iter_obj,
        body,
        local_vars = (),
        ):
        if PyccelAstNode.stage == "semantic":
            cond_iter = iterable(iter_obj)
            cond_iter = cond_iter or isinstance(iter_obj, (PythonRange, Product,
                    PythonEnumerate, PythonZip, PythonMap))
            cond_iter = cond_iter or isinstance(iter_obj, Variable) \
                and is_iterable_datatype(iter_obj.dtype)
          #  cond_iter = cond_iter or isinstance(iter_obj, ConstructorCall) \
          #      and is_iterable_datatype(iter_obj.arguments[0].dtype)
            if not cond_iter:
                raise TypeError('iter_obj must be an iterable')

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a Codeblock')

        self._target = target
        self._iterable = iter_obj
        self._body = body
        self._local_vars = local_vars
        self._nowait_expr = None
        super().__init__()

    @property
    def nowait_expr(self):
        return self._nowait_expr

    @nowait_expr.setter
    def nowait_expr(self, expr):
        self._nowait_expr = expr

    @property
    def target(self):
        return self._target

    @property
    def iterable(self):
        return self._iterable

    @property
    def body(self):
        return self._body

    @property
    def local_vars(self):
        return self._local_vars

    def insert2body(self, stmt):
        self.body.insert2body(stmt)



class DoConcurrent(For):
    __slots__ = ()
    pass


class ForIterator(For):

    """Class that describes iterable classes defined by the user."""
    __slots__ = ()

    def __init__(
        self,
        target,
        iterable,
        body,
        ):

        if isinstance(iterable, Variable):
            iterable = PythonRange(PythonLen(iterable))
        super().__init__(target, iterable, body)

    # TODO uncomment later when we intriduce iterators
    # @property
    # def target(self):
    #    ts = super(ForIterator, self).target

    #    if not(len(ts) == self.depth):
    #        raise ValueError('wrong number of targets')

    #    return ts

    @property
    def depth(self):
        it = self.iterable
        if isinstance(it, Variable):
            if isinstance(it.dtype, NativeRange):
                return 1

            cls_base = it.cls_base
            if not cls_base:
                raise TypeError('cls_base undefined')

            methods = cls_base.methods_as_dict
            it_method = methods['__iter__']

            it_vars = []
            for stmt in it_method.body:
                if isinstance(stmt, Assign):
                    it_vars.append(stmt.lhs)

            n = len(set(str(var.name) for var in it_vars))
            return n
        else:

            return 1

    @property
    def ranges(self):
        return get_iterable_ranges(self.iterable)

class ConstructorCall(Basic):

    """
    It  serves as a constructor for undefined function classes.

    Parameters
    ----------
    func: FunctionDef, str
        an instance of FunctionDef or function name

    arguments: list, tuple, None
        a list of arguments.

    """
    __slots__ = ('_cls_variable', '_func', '_arguments')
    _attribute_nodes = ('_func', '_arguments')

    is_commutative = True

    # TODO improve

    def __init__(
        self,
        func,
        arguments,
        cls_variable=None,
        ):
        if not isinstance(func, (FunctionDef, Interface, str)):
            raise TypeError('Expecting func to be a FunctionDef or str')

        self._cls_variable = cls_variable
        self._func = func
        self._arguments = arguments
        super().__init__()

    def __str__(self):
        name = str(self.name)
        args = ''
        if not self.arguments is None:
            args = ', '.join(str(i) for i in self.arguments)
        return '{0}({1})'.format(name, args)

    @property
    def func(self):
        return self._func

    @property
    def arguments(self):
        return self._arguments

    @property
    def cls_variable(self):
        return self._cls_variable

    @property
    def name(self):
        if isinstance(self.func, FunctionDef):
            return self.func.name
        else:
            return self.func

class Argument(PyccelAstNode):

    """An abstract Argument data structure.

    Examples
    --------
    >>> from pyccel.ast.core import Argument
    >>> n = Argument('n')
    >>> n
    n
    """
    __slots__ = ('_name','_kwonly','_annotation')
    _attribute_nodes = ()

    def __init__(self, name, *, kwonly=False, annotation=None):
        self._name       = name
        self._kwonly     = kwonly
        self._annotation = annotation
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def is_kwonly(self):
        return self._kwonly

    @property
    def annotation(self):
        return self._annotation

    def __str__(self):
        return str(self.name)

class ValuedArgument(Basic):

    """Represents a valued argument in the code.

    Examples
    --------
    >>> from pyccel.ast.core import ValuedArgument
    >>> n = ValuedArgument('n', 4)
    >>> n
    n=4
    """
    __slots__ = ('_name','_expr','_value','_kwonly')
    _attribute_nodes = ()

    def __init__(self, expr, value, *, kwonly = False):
        # TODO should we turn back to Argument

        if isinstance(expr, Argument):
            self._name = expr.name
        elif isinstance(expr, str):
            self._name = expr
        else:
            raise TypeError('Expecting an argument')

        if isinstance(value, (bool, int, float, complex, str)) and not isinstance(value, PyccelSymbol):
            value = convert_to_literal(value)
        elif not isinstance(value, (Basic, PyccelSymbol)):
            raise TypeError("Expecting a pyccel object not {}".format(type(value)))

        if not isinstance(kwonly, bool):
            raise TypeError("kwonly must be a bool")

        self._expr   = expr
        self._value  = value
        self._kwonly = kwonly
        super().__init__()

    @property
    def argument(self):
        return self._expr

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        return self._name

    @property
    def is_kwonly(self):
        return self._kwonly

    def __str__(self):
        argument = str(self.argument)
        value = str(self.value)
        return '{0}={1}'.format(argument, value)

class FunctionCall(PyccelAstNode):

    """Represents a function call in the code.
    """
    __slots__ = ('_arguments','_funcdef','_interface','_func_name','_interface_name',
                 '_dtype','_precision','_shape','_rank','_order')
    _attribute_nodes = ('_arguments','_funcdef','_interface')

    def __init__(self, func, args, current_function=None):

        if self.stage == "syntactic":
            self._interface = None
            self._funcdef   = func
            self._arguments = tuple(args)
            self._func_name = func
            super().__init__()
            return

        # ...
        if not isinstance(func, (FunctionDef, Interface)):
            raise TypeError('> expecting a FunctionDef or an Interface')

        if isinstance(func, Interface):
            self._interface = func
            self._interface_name = func.name
            func = func.point(args)
        else:
            self._interface = None

        name = func.name
        # ...
        if isinstance(current_function, DottedName):
            current_function = current_function.name[-1]
        if current_function == name:
            func.set_recursive()

        if not isinstance(args, (tuple, list)):
            raise TypeError('args must be a list or tuple')

        # add the missing argument in the case of optional arguments
        f_args = func.arguments
        if func.cls_name:
            f_args = f_args[1:]
        if not len(args) == len(f_args):
            f_args_dict = OrderedDict((a.name,a) if isinstance(a, (ValuedVariable, ValuedFunctionAddress)) else (a.name, None) for a in f_args)
            keyword_args = []
            for i,a in enumerate(args):
                if not isinstance(a, (ValuedVariable, ValuedFunctionAddress)):
                    f_args_dict[f_args[i].name] = a
                else:
                    keyword_args = args[i:]
                    break

            for a in keyword_args:
                f_args_dict[a.name] = a.value

            args = [a.value if isinstance(a, (ValuedVariable, ValuedFunctionAddress)) else a for a in f_args_dict.values()]

        args = [FunctionAddress(a.name, a.arguments, a.results, []) if isinstance(a, FunctionDef) else a for a in args]

        if current_function == func.name:
            if len(func.results)>0 and not isinstance(func.results[0], PyccelAstNode):
                errors.report(RECURSIVE_RESULTS_REQUIRED, symbol=func, severity="fatal")

        self._funcdef       = func
        self._arguments     = args
        self._dtype         = func.results[0].dtype     if len(func.results) == 1 else NativeTuple()
        self._rank          = func.results[0].rank      if len(func.results) == 1 else None
        self._shape         = func.results[0].shape     if len(func.results) == 1 else None
        self._precision     = func.results[0].precision if len(func.results) == 1 else None
        self._order         = func.results[0].order     if len(func.results) == 1 else None
        self._func_name     = func.name
        super().__init__()

    @property
    def args(self):
        return self._arguments

    @property
    def funcdef(self):
        return self._funcdef

    @property
    def interface(self):
        return self._interface

    @property
    def func_name(self):
        return self._func_name

    @property
    def interface_name(self):
        return self._interface_name

    def __repr__(self):
        return '{}({})'.format(self.func_name, ', '.join(str(a) for a in self.args))

class DottedFunctionCall(FunctionCall):
    """
    Represents a function call in the code where
    the function is defined in another object
    (e.g. module/class)

    a.f()

    Parameters
    ==========
    func             : FunctionDef
                       The definition of the function being called
    args             : tuple
                       The arguments being passed to the function
    prefix           : PyccelAstNode
                       The object in which the function is defined
                       E.g. for a.f()
                       prefix will contain a
    current_function : str
                        The function from which this call occurs
                        (This is required in order to recognise
                        recursive functions)
    """
    __slots__ = ('_prefix',)
    _attribute_nodes = (*FunctionCall._attribute_nodes, '_prefix')

    def __init__(self, func, args, prefix, current_function=None):
        self._prefix = prefix
        super().__init__(func, args, current_function)
        self._func_name = DottedName(prefix, self._func_name)
        if self._interface:
            self._interface_name = DottedName(prefix, self._interface_name)

    @property
    def prefix(self):
        """ The object in which the function is defined
        """
        return self._prefix

class Return(Basic):

    """Represents a function return in the code.

    Parameters
    ----------
    expr : PyccelAstNode
        The expression to return.

    stmts :represent assign stmts in the case of expression return
    """
    __slots__ = ('_expr', '_stmt')
    _attribute_nodes = ('_expr', '_stmt')

    def __init__(self, expr, stmt=None):

        if stmt and not isinstance(stmt, CodeBlock):
            raise TypeError('stmt should only be of type CodeBlock')

        self._expr = expr
        self._stmt = stmt

        super().__init__()

    @property
    def expr(self):
        return self._expr

    @property
    def stmt(self):
        return self._stmt

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (self.expr, self.stmt)
        return args

class FunctionDef(Basic):

    """Represents a function definition.

    Parameters
    ----------
    name : str
        The name of the function.

    arguments : iterable
        The arguments to the function.

    results : iterable
        The direct outputs of the function.

    body : iterable
        The body of the function.

    local_vars : list of Symbols
        These are used internally by the routine.

    global_vars : list of Symbols
        Variables which will not be passed into the function.

    cls_name: str
        Class name if the function is a method of cls_name

    is_pure: bool
        True for a function without side effect

    is_elemental: bool
        True for a function that is elemental

    is_private: bool
        True for a function that is private

    is_static: bool
        True for static functions. Needed for iso_c_binding interface

    imports: list, tuple
        a list of needed imports

    decorators: list, tuple
        a list of proporties

    Examples
    --------
    >>> from pyccel.ast.core import Assign, Variable, FunctionDef
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x]
    >>> results     = [y]
    >>> body        = [Assign(y,x+1)]
    >>> FunctionDef('incr', args, results, body)
    FunctionDef(incr, (x,), (y,), [y := 1 + x], [], [], None, False, function)

    One can also use parametrized argument, using ValuedArgument

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Assign
    >>> from pyccel.ast.core import FunctionDef
    >>> from pyccel.ast.core import ValuedArgument
    >>> n = ValuedArgument('n', 4)
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x, n]
    >>> results     = [y]
    >>> body        = [Assign(y,x+n)]
    >>> FunctionDef('incr', args, results, body)
    FunctionDef(incr, (x, n=4), (y,), [y := 1 + x], [], [], None, False, function, [])
    """
    __slots__ = ('_name','_arguments','_results','_body','_local_vars',
                 '_global_vars','_cls_name','_is_static','_imports',
                 '_decorators','_headers','_is_recursive','_is_pure',
                 '_is_elemental','_is_private','_is_header','_arguments_inout',
                 '_functions','_interfaces','_doc_string')
    _attribute_nodes = ('_arguments','_results','_body','_local_vars',
                 '_global_vars','_imports','_functions','_interfaces')

    def __init__(
        self,
        name,
        arguments,
        results,
        body,
        local_vars=(),
        global_vars=(),
        cls_name=None,
        is_static=False,
        imports=(),
        decorators={},
        headers=(),
        is_recursive=False,
        is_pure=False,
        is_elemental=False,
        is_private=False,
        is_header=False,
        arguments_inout=(),
        functions=(),
        interfaces=(),
        doc_string=None):

        if isinstance(name, str):
            name = PyccelSymbol(name)
        elif isinstance(name, (tuple, list)):
            name_ = []
            for i in name:
                if isinstance(i, str):
                    name_.append(PyccelSymbol(i))
                else:
                    raise TypeError('Function name must be PyccelSymbol or string'
                                    )
            name = tuple(name_)
        else:
            raise TypeError('Function name must be PyccelSymbol or string')

        # arguments

        if not iterable(arguments):
            raise TypeError('arguments must be an iterable')

        # TODO improve and uncomment
#        if not all(isinstance(a, Argument) for a in arguments):
#            raise TypeError("All arguments must be of type Argument")

        arguments = tuple(arguments)

        # body

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')

#        body = tuple(i for i in body)
        # results

        if not iterable(results):
            raise TypeError('results must be an iterable')
        results = tuple(results)

        # if method

        if cls_name:

            if not isinstance(cls_name, str):
                raise TypeError('cls_name must be a string')

            # if not cls_variable:
             #   raise TypeError('Expecting a instance of {0}'.format(cls_name))

        if not isinstance(is_static, bool):
            raise TypeError('Expecting a boolean for is_static attribute')

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        if not isinstance(decorators, dict):
            raise TypeError('decorators must be a dict')

        if not isinstance(is_pure, bool):
            raise TypeError('Expecting a boolean for pure')

        if not isinstance(is_elemental, bool):
            raise TypeError('Expecting a boolean for elemental')

        if not isinstance(is_private, bool):
            raise TypeError('Expecting a boolean for private')

        if not isinstance(is_header, bool):
            raise TypeError('Expecting a boolean for header')

        if arguments_inout:
            if not isinstance(arguments_inout, (list, tuple)):
                raise TypeError('Expecting a list or tuple ')

            if not all([isinstance(i, bool) for i in arguments_inout]):
                raise ValueError('Expecting booleans')

        else:
            # TODO shall we keep this?
            arguments_inout = [False for a in arguments]

        if functions:
            for i in functions:
                if not isinstance(i, FunctionDef):
                    raise TypeError('Expecting a FunctionDef')

        self._name            = name
        self._arguments       = arguments
        self._results         = results
        self._body            = body
        self._local_vars      = local_vars
        self._global_vars     = global_vars
        self._cls_name        = cls_name
        self._is_static       = is_static
        self._imports         = imports
        self._decorators      = decorators
        self._headers         = headers
        self._is_recursive    = is_recursive
        self._is_pure         = is_pure
        self._is_elemental    = is_elemental
        self._is_private      = is_private
        self._is_header       = is_header
        self._arguments_inout = arguments_inout
        self._functions       = functions
        self._interfaces      = interfaces
        self._doc_string      = doc_string
        super().__init__()

    @property
    def name(self):
        """ Name of the function """
        return self._name

    @property
    def arguments(self):
        """ List of variables which are the function arguments """
        return self._arguments

    @property
    def results(self):
        """ List of variables which are the function results """
        return self._results

    @property
    def body(self):
        """ CodeBlock containing all the statements in the function """
        return self._body

    @property
    def local_vars(self):
        """ List of variables defined in the function """
        return self._local_vars

    @property
    def global_vars(self):
        """ List of global variables used in the function """
        return self._global_vars

    @property
    def cls_name(self):
        """ String containing the name of the class to which the method belongs.
        If the function is not a class procedure then this returns None """
        return self._cls_name

    @cls_name.setter
    def cls_name(self, cls_name):
        self._cls_name = cls_name

    @property
    def imports(self):
        """ List of imports in the function """
        return self._imports

    @property
    def decorators(self):
        """ List of decorators applied to the function """
        return self._decorators

    @property
    def headers(self):
        """ List of headers applied to the function """
        return self._headers

    @property
    def templates(self):
        """ List of templates used to determine the types """
        return self._templates

    @property
    def is_recursive(self):
        """ Returns True if the function is recursive (i.e. calls itself)
        and False otherwise """
        return self._is_recursive

    @property
    def is_pure(self):
        """ Returns True if the function is marked as pure and False otherwise
        Pure functions must not have any side effects.
        In other words this means that the result must be the same no matter
        how many times the function is called
        e.g:
        >>> a = f()
        >>> a = f()

        gives the same result as
        >>> a = f()

        This is notably not true for I/O functions
        """
        return self._is_pure

    @property
    def is_elemental(self):
        """ returns True if the function is marked as elemental and
        False otherwise
        An elemental function is a function with a single scalar operator
        and a scalar return value which can also be called on an array.
        When it is called on an array it returns the result of the function
        called elementwise on the array """
        return self._is_elemental

    @property
    def is_private(self):
        """ True if the function should not be exposed to
        other modules. This includes the wrapper module and
        means that the function cannot be used in an import
        or exposed to python """
        return self._is_private

    @property
    def is_header(self):
        """ True if the implementation of the function body
        is not provided False otherwise """
        return self._is_header

    @property
    def arguments_inout(self):
        """ List of variables which are the modifiable function arguments """
        return self._arguments_inout

    @property
    def functions(self):
        """ List of functions within this function """
        return self._functions

    @property
    def interfaces(self):
        """ List of interfaces within this function """
        return self._interfaces

    @property
    def doc_string(self):
        """ The docstring of the function """
        return self._doc_string

    def set_recursive(self):
        """ Mark the function as a recursive function """
        self._is_recursive = True

    def clone(self, newname):
        """
        Create an identical FunctionDef with name
        newname.

        Parameters
        ----------
        newname: str
            new name for the FunctionDef
        """
        args, kwargs = self.__getnewargs__()
        cls = type(self)
        new_func = cls(*args, **kwargs)
        new_func.rename(newname)
        return new_func


    def rename(self, newname):
        """
        Rename the FunctionDef name
        newname.

        Parameters
        ----------
        newname: str
            new name for the FunctionDef
        """

        self._name = newname

    def add_local_vars(self, *variables):
        """
        Add (a) new variable(s) to the local variables tuple

        Parameters
        ----------
        var : Variable
              The new local variable
        """
        _ = [v.set_current_user_node(self) for v in variables]
        self._local_vars += variables

    def __getnewargs__(self):
        """
          This method returns the positional and keyword arguments
            used to create an instance of this class.
        """
        args = (
        self._name,
        self._arguments,
        self._results,
        self._body)

        kwargs = {
        'local_vars':self._local_vars,
        'global_vars':self._global_vars,
        'cls_name':self._cls_name,
        'is_static':self._is_static,
        'imports':self._imports,
        'decorators':self._decorators,
        'headers':self._headers,
        'is_recursive':self._is_recursive,
        'is_pure':self._is_pure,
        'is_elemental':self._is_elemental,
        'is_private':self._is_private,
        'is_header':self._is_header,
        'arguments_inout':self._arguments_inout,
        'functions':self._functions}
        return args, kwargs

    def __reduce_ex__(self, i):
        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments.
        """
        args, kwargs = self.__getnewargs__()
        out = (apply, (self.__class__, args, kwargs))
        return out


    def __str__(self):
        result = 'None' if len(self.results) == 0 else \
                    ', '.join(str(r) for r in self.results)
        args = ', '.join(str(a) for a in self.arguments)
        return '{name}({args}) -> {result}'.format(
                name   = self.name,
                args   = args,
                result = result)

    @property
    def is_unused(self):
        return False

class Interface(Basic):

    """Represents an Interface.

    Parameters
    ----------
    name : str
        The name of the interface.

    functions : iterable
        The functions of the interface.

    is_argument: bool
        True if the interface is used for a function argument.

    Examples
    --------
    >>> from pyccel.ast.core import Interface, FunctionDef
    >>> f = FunctionDef('F', [], [], [])
    >>> Interface('I', [f])
    """
    __slots__ = ('_name','_functions','_is_argument')
    _attribute_nodes = ('_functions',)

    def __init__(
        self,
        name,
        functions,
        is_argument = False,
        ):

        if not isinstance(name, str):
            raise TypeError('Expecting an str')
        if not isinstance(functions, list):
            raise TypeError('Expecting a list')
        self._name = name
        self._functions = functions
        self._is_argument = is_argument
        super().__init__()

    @property
    def name(self):
        """Name of the interface."""
        return self._name

    @property
    def functions(self):
        """"Functions of the interface."""
        return self._functions

    @property
    def is_argument(self):
        """True if the interface is used for a function argument."""
        return self._is_argument

    @property
    def doc_string(self):
        return self._functions[0].doc_string

    def point(self, args):
        """Returns the actual function that will be called, depending on the passed arguments."""
        fs_args = [[j for j in i.arguments] for i in
                    self._functions]
        j = -1
        for i in fs_args:
            j += 1
            found = True
            for (x, y) in enumerate(args):
                dtype1 = str_dtype(y.dtype)
                dtype2 = str_dtype(i[x].dtype)
                found = found and (dtype1 in dtype2
                                or dtype2 in dtype1)
                found = found and y.rank \
                                == i[x].rank
            if found:
                break

        if found:
            return  self._functions[j]
        else:
            errors.report('Arguments types provided to {} are incompatible'.format(self.name),
                        severity='fatal')

class FunctionAddress(FunctionDef):

    """Represents a function address.

    Parameters
    ----------
    name : str
        The name of the function address.

    arguments : iterable
        The arguments to the function address.

    results : iterable
        The direct outputs of the function address.

    is_argument: bool
        if object is the argument of a function [Default value: False]

    is_kwonly: bool
        if object is an argument which can only be specified using its keyword

    is_pointer: bool
        if object is a pointer [Default value: False]

    is_optional: bool
        if object is an optional argument of a function [Default value: False]

    Examples
    --------
    >>> from pyccel.ast.core import Variable, FunctionAddress, FuncAddressDeclare, FunctionDef
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')

    a function definition can have a FunctionAddress as an argument

    >>> FunctionDef('g', [FunctionAddress('f', [x], [y], [])], [], [])

    we can also Declare a FunctionAddress

    >>> FuncAddressDeclare(FunctionAddress('f', [x], [y], []))
    """
    __slots__ = ('_is_optional','_is_pointer','_is_kwonly','_is_argument')

    def __init__(
        self,
        name,
        arguments,
        results,
        body,
        is_optional=False,
        is_pointer=False,
        is_kwonly=False,
        is_argument=False,
        **kwargs
        ):
        super().__init__(name, arguments, results, body, **kwargs)
        if not isinstance(is_argument, bool):
            raise TypeError('Expecting a boolean for is_argument')

        if not isinstance(is_pointer, bool):
            raise TypeError('Expecting a boolean for is_pointer')

        if not isinstance(is_kwonly, bool):
            raise TypeError('Expecting a boolean for kwonly')

        elif not isinstance(is_optional, bool):
            raise TypeError('is_optional must be a boolean.')

        self._is_optional   = is_optional
        self._is_pointer    = is_pointer
        self._is_kwonly     = is_kwonly
        self._is_argument   = is_argument

    @property
    def name(self):
        return self._name

    @property
    def is_pointer(self):
        return self._is_pointer

    @property
    def is_argument(self):
        return self._is_argument

    @property
    def is_kwonly(self):
        return self._is_kwonly

    @property
    def is_optional(self):
        return self._is_optional

class ValuedFunctionAddress(FunctionAddress):

    """Represents a valued function address in the code.

    Parameters
    ----------
    value: instance of FunctionDef or FunctionAddress

    Examples
    --------
    >>> from pyccel.ast.core import Variable, ValuedFunctionAddress, FunctionDef
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> f = FunctionDef('f', [], [], [])
    >>> n  = ValuedFunctionAddress('g', [x], [y], [], value=f)
    """
    __slots__ = ('_value')
    _attribute_nodes = (*FunctionAddress._attribute_nodes, '_value')

    def __init__(self, *args, **kwargs):
        self._value = kwargs.pop('value', Nil())
        super().__init__(*args, **kwargs)

    @property
    def value(self):
        return self._value

class SympyFunction(FunctionDef):

    """Represents a function definition."""
    __slots__ = ()


# TODO: [EB 06.01.2021] Is this class used? What for? See issue #668
class PythonFunction(FunctionDef):

    """Represents a Python-Function definition."""
    __slots__ = ()


class BindCFunctionDef(FunctionDef):
    """
    Contains the c-compatible version of the function which is
    used for the wrapper.
    As compared to a normal FunctionDef, this version contains
    arguments for the shape of arrays. It should be generated by
    calling ast.bind_c.as_static_function_call

    Parameters
    ----------
    *args : See FunctionDef

    original_function : FunctionDef
        The function from which the c-compatible version was created
    """
    __slots__ = ('_original_function',)
    _attribute_nodes = (*FunctionDef._attribute_nodes, '_original_function')

    def __init__(self, *args, original_function, **kwargs):
        self._original_function = original_function
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return str(self._name).lower()

    @property
    def original_function(self):
        return self._original_function


class ClassDef(Basic):

    """Represents a class definition.

    Parameters
    ----------
    name : str
        The name of the class.

    attributes: iterable
        The attributes to the class.

    methods: iterable
        Class methods

    options: list, tuple
        list of options ('public', 'private', 'abstract')

    imports: list, tuple
        list of needed imports

    superclass : str
        superclass's class name

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.core import ClassDef, FunctionDef
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> z = Variable('real', 'z')
    >>> t = Variable('real', 't')
    >>> a = Variable('real', 'a')
    >>> b = Variable('real', 'b')
    >>> body = [Assign(y,x+a)]
    >>> translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
    >>> attributes   = [x,y]
    >>> methods     = [translate]
    >>> ClassDef('Point', attributes, methods)
    ClassDef(Point, (x, y), (FunctionDef(translate, (x, y, a, b), (z, t), [y := a + x], [], [], None, False, function),), [public])
    """
    __slots__ = ('_name','_attributes','_methods','_options',
                 '_imports','_superclass','_interfaces')
    _attribute_nodes = ('_attributes', '_methods', '_imports', '_interfaces')

    def __init__(
        self,
        name,
        attributes=(),
        methods=(),
        options=('public',),
        imports=(),
        superclass=(),
        interfaces=(),
        ):

        # name

        if isinstance(name, str):
            name = PyccelSymbol(name)
        else:
            raise TypeError('Function name must be PyccelSymbol or string')

        # attributes

        if not iterable(attributes):
            raise TypeError('attributes must be an iterable')
        attributes = tuple(attributes)

        # methods

        if not iterable(methods):
            raise TypeError('methods must be an iterable')

        # options

        if not iterable(options):
            raise TypeError('options must be an iterable')

        # imports

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        if not iterable(superclass):
            raise TypeError('superclass must be iterable')

        if not iterable(interfaces):
            raise TypeError('interfaces must be iterable')

        imports = list(imports)
        for i in methods:
            imports += list(i.imports)

        imports = set(imports)  # for unicity
        imports = tuple(imports)

        # ...
        # look if the class has the method __del__
        # d_methods = {}
        # for i in methods:
        #    d_methods[str(i.name).replace('\'','')] = i
        # if not ('__del__' in d_methods):
        #    dtype = DataTypeFactory(str(name), ("_name"), prefix='Custom')
        #    this  = Variable(dtype(), 'self')

            # constructs the __del__ method if not provided
         #   args = []
         #   for a in attributes:
         #       if isinstance(a, Variable):
         #           if a.allocatable:
         #              args.append(a)

         #   args = [Variable(a.dtype, DottedName(str(this), str(a.name))) for a in args]
         #   body = [Del(a) for a in args]

         #   free = FunctionDef('__del__', [this], [], \
         #                      body, local_vars=[], global_vars=[], \
         #                      cls_name='__UNDEFINED__', imports=[])

         #  methods = list(methods) + [free]
         # TODO move this somewhere else

        methods = tuple(methods)

        # ...
        self._name = name
        self._attributes = attributes
        self._methods = methods
        self._options = options
        self._imports = imports
        self._superclass  = superclass
        self._interfaces = interfaces

        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def attributes(self):
        return self._attributes

    @property
    def methods(self):
        return self._methods

    @property
    def options(self):
        return self._options

    @property
    def imports(self):
        return self._imports

    @property
    def superclass(self):
        return self._superclass

    @property
    def interfaces(self):
        return self._interfaces

    @property
    def methods_as_dict(self):
        """Returns a dictionary that contains all methods, where the key is the
        method's name."""

        d_methods = {}
        for i in self.methods:
            d_methods[i.name] = i
        return d_methods

    @property
    def attributes_as_dict(self):
        """Returns a dictionary that contains all attributes, where the key is the
        attribute's name."""

        d_attributes = {}
        for i in self.attributes:
            d_attributes[i.name] = i
        return d_attributes

    # TODO add other attributes?


    def get_attribute(self, O, attr):
        """Returns the attribute attr of the class O of instance self."""

        if not isinstance(attr, str):
            raise TypeError('Expecting attribute to be a string')

        if isinstance(O, Variable):
            cls_name = O.name
        else:
            cls_name = str(O)

        attributes = {}
        for i in self.attributes:
            attributes[i.name] = i

        if not attr in attributes:
            raise ValueError('{0} is not an attribute of {1}'.format(attr,
                             str(self)))

        var = attributes[attr]
        name = DottedName(cls_name, var.name)
        return Variable(
            var.dtype,
            name,
            rank=var.rank,
            allocatable=var.allocatable,
            shape=var.shape,
            cls_base=var.cls_base,
            )

    @property
    def is_iterable(self):
        """Returns True if the class has an iterator."""

        names = [str(m.name) for m in self.methods]
        if '__next__' in names and '__iter__' in names:
            return True
        elif '__next__' in names:
            raise ValueError('ClassDef does not contain __iter__ method')
        elif '__iter__' in names:
            raise ValueError('ClassDef does not contain __next__ method')
        else:
            return False

    @property
    def is_with_construct(self):
        """Returns True if the class is a with construct."""

        names = [str(m.name) for m in self.methods]
        if '__enter__' in names and '__exit__' in names:
            return True
        elif '__enter__' in names:
            raise ValueError('ClassDef does not contain __exit__ method')
        elif '__exit__' in names:
            raise ValueError('ClassDef does not contain __enter__ method')
        else:
            return False

    @property
    def hide(self):
        if 'hide' in self.options:
            return True
        else:
            return self.is_iterable or self.is_with_construct

    @property
    def is_unused(self):
        return False


class Import(Basic):

    """Represents inclusion of dependencies in the code.

    Parameters
    ----------
    target : str, list, tuple
        targets to import

    Examples
    --------
    >>> from pyccel.ast.core import Import
    >>> from pyccel.ast.core import DottedName
    >>> Import('foo')
    import foo

    >>> abc = DottedName('foo', 'bar', 'baz')
    >>> Import(abc)
    import foo.bar.baz

    >>> Import(['foo', abc])
    import foo, foo.bar.baz
    """
    __slots__ = ('_source','_target','_ignore_at_print')
    _attribute_nodes = ()

    def __init__(self, source, target = None, ignore_at_print = False):

        if not source is None:
            source = Import._format(source)

        self._source = source
        self._target = set()
        self._ignore_at_print = ignore_at_print
        if isinstance(target, (str, DottedName, AsName)):
            self._target = set([Import._format(target)])
        elif iterable(target):
            for i in target:
                self._target.add(Import._format(i))
        super().__init__()

    @staticmethod
    def _format(i):
        if isinstance(i, str):
            if '.' in i:
                return DottedName(*i.split('.'))
            else:
                return PyccelSymbol(i)
        if isinstance(i, (DottedName, AsName)):
            return i
        elif isinstance(i, PyccelSymbol):
            return i
        else:
            raise TypeError('Expecting a string, PyccelSymbol DottedName, given {}'.format(type(i)))

    @property
    def target(self):
        return self._target

    @property
    def source(self):
        return self._source

    @property
    def ignore(self):
        return self._ignore_at_print

    @ignore.setter
    def ignore(self, to_ignore):
        if not isinstance(to_ignore, bool):
            raise TypeError('to_ignore must be a boolean.')
        self._ignore_at_print = to_ignore

    def __str__(self):
        source = str(self.source)
        if len(self.target) == 0:
            return 'import {source}'.format(source=source)
        else:
            target = ', '.join([str(i) for i in self.target])
            return 'from {source} import {target}'.format(source=source,
                    target=target)

    def define_target(self, new_target):
        self._target.add(new_target)

    def find_module_target(self, new_target):
        for t in self._target:
            if isinstance(t, AsName) and new_target == t.name:
                return t.target
            elif new_target == t:
                return t
        return None


# TODO: Should Declare have an optional init value for each var?

class FuncAddressDeclare(Basic):

    """Represents a FunctionAddress declaration in the code.

    Parameters
    ----------
    variable:
        An instance of FunctionAddress.
    intent: None, str
        one among {'in', 'out', 'inout'}
    value: PyccelAstNode
        variable value
    static: bool
        True for a static declaration of an array.

    Examples
    --------
    >>> from pyccel.ast.core import Variable, FunctionAddress, FuncAddressDeclare
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> FuncAddressDeclare(FunctionAddress('f', [x], [y], []))
    """
    __slots__ = ('_variable','_intent','_value','_static')
    _attribute_nodes = ('_variable', '_value')

    def __init__(
        self,
        variable,
        intent=None,
        value=None,
        static=False,
        ):

        if not isinstance(variable, FunctionAddress):
            raise TypeError('variable must be of type FunctionAddress, given {0}'.format(variable))

        if intent:
            if not intent in ['in', 'out', 'inout']:
                raise ValueError("intent must be one among {'in', 'out', 'inout'}")

        if not isinstance(static, bool):
            raise TypeError('Expecting a boolean for static attribute')

        self._variable  = variable
        self._intent    = intent
        self._value     = value
        self._static    = static
        super().__init__()

    @property
    def results(self):
        return self._variable.results

    @property
    def arguments(self):
        return self._variable.arguments

    @property
    def name(self):
        return self._variable.name

    @property
    def variable(self):
        return self._variable

    @property
    def intent(self):
        return self._intent

    @property
    def value(self):
        return self._value

    @property
    def static(self):
        return self._static

class Declare(Basic):

    """Represents a variable declaration in the code.

    Parameters
    ----------
    dtype : DataType
        The type for the declaration.
    variable(s)
        A single variable or an iterable of Variables. If iterable, all
        Variables must be of the same type.
    intent: None, str
        one among {'in', 'out', 'inout'}
    value: PyccelAstNode
        variable value
    static: bool
        True for a static declaration of an array.

    Examples
    --------
    >>> from pyccel.ast.core import Declare, Variable
    >>> Declare('int', Variable('int', 'n'))
    Declare(NativeInteger(), (n,), None)
    >>> Declare('real', Variable('real', 'x'), intent='out')
    Declare(NativeReal(), (x,), out)
    """
    __slots__ = ('_dtype','_variable','_intent','_value',
                 '_static','_passed_from_dotted')
    _attribute_nodes = ('_variable', '_value')

    def __init__(
        self,
        dtype,
        variable,
        intent=None,
        value=None,
        static=False,
        passed_from_dotted = False,
        ):
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        if not isinstance(variable, Variable):
            raise TypeError('var must be of type Variable, given {0}'.format(variable))
        if variable.dtype != dtype:
            raise ValueError('All variables must have the same dtype')

        if intent:
            if not intent in ['in', 'out', 'inout']:
                raise ValueError("intent must be one among {'in', 'out', 'inout'}")

        if not isinstance(static, bool):
            raise TypeError('Expecting a boolean for static attribute')

        if not isinstance(passed_from_dotted, bool):
            raise TypeError('Expecting a boolean for passed_from_dotted attribute')

        self._dtype = dtype
        self._variable = variable
        self._intent = intent
        self._value = value
        self._static = static
        self._passed_from_dotted = passed_from_dotted
        super().__init__()

    @property
    def dtype(self):
        return self._dtype

    @property
    def variable(self):
        return self._variable

    @property
    def intent(self):
        return self._intent

    @property
    def value(self):
        return self._value

    @property
    def static(self):
        return self._static

    @property
    def passed_from_dotted(self):
        """ Argument is the lhs of a DottedFunction
        """
        return self._passed_from_dotted


class Break(Basic):

    """Represents a break in the code."""
    __slots__ = ()
    _attribute_nodes = ()


class Continue(Basic):

    """Represents a continue in the code."""
    __slots__ = ()
    _attribute_nodes = ()


class Raise(Basic):

    """Represents a raise in the code."""
    __slots__ = ()
    _attribute_nodes = ()



class SymbolicPrint(Basic):

    """Represents a print function of symbolic expressions in the code.

    Parameters
    ----------
    expr : PyccelAstNode
        The expression to print

    Examples
    --------
    >>> from pyccel.ast.internals import symbols
    >>> from pyccel.ast.core import Print
    >>> n,m = symbols('n,m')
    >>> Print(('results', n,m))
    Print((results, n, m))
    """
    __slots__ = ('_expr',)
    _attribute_nodes = ('_expr',)

    def __init__(self, expr):
        if not iterable(expr):
            raise TypeError('Expecting an iterable')

        for i in expr:
            if not isinstance(i, (Lambda, SymbolicAssign,
                              SympyFunction)):
                raise TypeError('Expecting Lambda, SymbolicAssign, SympyFunction for {}'.format(i))

        self._expr = expr

        super().__init__()

    @property
    def expr(self):
        return self._expr


class Del(Basic):

    """Represents a memory deallocation in the code.

    Parameters
    ----------
    variables : list, tuple
        a list of pyccel variables

    Examples
    --------
    >>> from pyccel.ast.core import Del, Variable
    >>> x = Variable('real', 'x', rank=2, shape=(10,2), allocatable=True)
    >>> Del([x])
    Del([x])
    """
    __slots__ = ('_variables',)
    _attribute_nodes = ('_variables',)

    def __init__(self, expr):

        # TODO: check that the variable is allocatable

        if not iterable(expr):
            expr = tuple([expr])

        self._variables = expr
        super().__init__()

    @property
    def variables(self):
        return self._variables


class EmptyNode(Basic):
    """
    Represents an empty node in the abstract syntax tree (AST).
    When a subtree is removed from the AST, we replace it with an EmptyNode
    object that acts as a placeholder. Using an EmptyNode instead of None
    is more explicit and avoids confusion. Further, finding a None in the AST
    is signal of an internal bug.

    Parameters
    ----------
    text : str
       the comment line

    Examples
    --------
    >>> from pyccel.ast.core import EmptyNode
    >>> EmptyNode()

    """
    __slots__ = ()
    _attribute_nodes = ()

    def __str__(self):
        return ''


class Comment(Basic):

    """Represents a Comment in the code.

    Parameters
    ----------
    text : str
       the comment line

    Examples
    --------
    >>> from pyccel.ast.core import Comment
    >>> Comment('this is a comment')
    # this is a comment
    """
    __slots__ = ('_text')
    _attribute_nodes = ()

    def __init__(self, text):
        self._text = text
        super().__init__()

    @property
    def text(self):
        return self._text

    def __str__(self):
        return '# {0}'.format(str(self.text))

    def __reduce_ex__(self, i):
        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments.
        """
        kwargs = dict(text = self.text)
        return (apply, (self.__class__, (), kwargs))


class SeparatorComment(Comment):

    """Represents a Separator Comment in the code.

    Parameters
    ----------
    mark : str
        marker

    Examples
    --------
    >>> from pyccel.ast.core import SeparatorComment
    >>> SeparatorComment(n=40)
    # ........................................
    """
    __slots__ = ()

    def __init__(self, n):
        text = """.""" * n
        super().__init__(text)

class AnnotatedComment(Basic):

    """Represents a Annotated Comment in the code.

    Parameters
    ----------
    accel : str
       accelerator id. One among {'acc'}

    txt: str
        statement to print

    Examples
    --------
    >>> from pyccel.ast.core import AnnotatedComment
    >>> AnnotatedComment('acc', 'parallel')
    AnnotatedComment(acc, parallel)
    """
    __slots__ = ('_accel','_txt')
    _attribute_nodes = ()

    def __init__(self, accel, txt):
        self._accel = accel
        self._txt = txt
        super().__init__()

    @property
    def accel(self):
        return self._accel

    @property
    def txt(self):
        return self._txt

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (self.accel, self.txt)
        return args

class CommentBlock(Basic):

    """ Represents a Block of Comments

    Parameters
    ----------
    txt : str

    """
    __slots__ = ('_header','_comments')
    _attribute_nodes = ()

    def __init__(self, txt, header = 'CommentBlock'):
        if not isinstance(txt, str):
            raise TypeError('txt must be of type str')
        txt = txt.replace('"','')
        txts = txt.split('\n')

        self._header = header
        self._comments = txts

        super().__init__()

    @property
    def comments(self):
        return self._comments

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, header):
        self._header = header


class Assert(Basic):

    """Represents a assert statement in the code.

    Parameters
    ----------
    test: PyccelAstNode
        boolean expression to check

    Examples
    --------
    """
    __slots__ = ('_test',)
    _attribute_nodes = ('_test',)
    #TODO add type check in the semantic stage
    def __init__(self, test):
        #if not isinstance(test, (bool, Relational, sp_Boolean)):
        #    raise TypeError('test %s is of type %s, but must be a Relational, Boolean, or a built-in bool.'
        #                     % (test, type(test)))

        self._test = test
        super().__init__()

    @property
    def test(self):
        return self._test


class Pass(Basic):

    """Basic class for pass instruction."""
    __slots__ = ()
    _attribute_nodes = ()

class Exit(Basic):

    """Basic class for exits."""
    __slots__ = ()
    _attribute_nodes = ()

#TODO: [EB 26.01.2021] Do we need this unused class?
class ErrorExit(Exit):

    """Exit with error."""
    __slots__ = ()

class IfSection(Basic):
    """Represents a condition and associated code block
    in an if statement in the code.

    Parameters
    ----------
    cond : PyccelAstNode
           A boolean expression indicating whether or not the block
           should be executed
    body : CodeBlock
           The code to be executed in the condition is satisfied

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import Assign, IfSection, CodeBlock
    >>> n = PyccelSymbol('n')
    >>> IfSection((n>1), CodeBlock([Assign(n,n-1)]))
    IfSection((n>1), CodeBlock([Assign(n,n-1)]))
    """
    __slots__ = ('_condition','_block')
    _attribute_nodes = ('_condition','_block')

    def __init__(self, cond, body):

        if PyccelAstNode.stage == 'semantic' and cond.dtype is not NativeBool():
            cond = PythonBool(cond)
        if isinstance(body, (list, tuple)):
            body = CodeBlock(body)
        elif isinstance(body, CodeBlock):
            body = body
        else:
            raise TypeError('body is not iterable or CodeBlock')

        self._condition = cond
        self._block     = body

        super().__init__()

    @property
    def condition(self):
        return self._condition

    @property
    def body(self):
        return self._block

    def __iter__(self):
        return iter((self.condition, self.body))

class If(Basic):

    """Represents a if statement in the code.

    Parameters
    ----------
    args : IfSection
           All arguments are sections of the complete If block

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import Assign, If
    >>> n = PyccelSymbol('n')
    >>> i1 = IfSection((n>1), [Assign(n,n-1)])
    >>> i2 = IfSection(True, [Assign(n,n+1)])
    >>> If(i1, i2)
    If(IfSection((n>1), [Assign(n,n-1)]), IfSection(True, [Assign(n,n+1)]))
    """
    __slots__ = ('_blocks',)
    _attribute_nodes = ('_blocks',)

    # TODO add type check in the semantic stage

    def __init__(self, *args):

        if not all(isinstance(a, IfSection) for a in args):
            raise TypeError("An If must be composed of IfSections")

        self._blocks = args

        super().__init__()

    @property
    def blocks(self):
        return self._blocks

    @property
    def bodies(self):
        return [b.body for b in self._blocks]

class StarredArguments(Basic):
    __slots__ = ('_starred_obj',)
    _attribute_nodes = ('_starred_obj',)
    def __init__(self, args):
        self._starred_obj = args
        super().__init__()

    @property
    def args_var(self):
        return self._starred_obj


# ...

# ...

def get_initial_value(expr, var):
    """Returns the first assigned value to var in the Expression expr.

    Parameters
    ----------
    expr: Expression
        any AST valid expression

    var: str, Variable, DottedName, list, tuple
        variable name
    """

    # ...

    def is_None(expr):
        """Returns True if expr is None or Nil()."""

        return isinstance(expr, Nil) or expr is None

    # ...

    # ...

    if isinstance(var, str):
        return get_initial_value(expr, [var])
    elif isinstance(var, DottedName):

        return get_initial_value(expr, [str(var)])
    elif isinstance(var, Variable):

        return get_initial_value(expr, [var.name])
    elif not isinstance(var, (list, tuple)):

        raise TypeError('Expecting var to be str, list, tuple or Variable, given {0}'.format(type(var)))

    # ...

    # ...

    if isinstance(expr, ValuedVariable):
        if expr.variable.name in var:
            return expr.value
    elif isinstance(expr, Variable):

        # expr.cls_base if of type ClassDef

        if expr.cls_base:
            return get_initial_value(expr.cls_base, var)
    elif isinstance(expr, Assign):

        if str(expr.lhs) in var:
            return expr.rhs
    elif isinstance(expr, FunctionDef):

        value = get_initial_value(expr.body, var)
        if not is_None(value):
            r = get_initial_value(expr.arguments, value)
            if 'self._linear' in var:
                print ('>>>> ', var, value, r)
            if not r is None:
                return r
        return value

    elif isinstance(expr, ConstructorCall):

        return get_initial_value(expr.func, var)
    elif isinstance(expr, (list, tuple)):

        for i in expr:
            value = get_initial_value(i, var)

            # here we make a difference between None and Nil,
            # since the output of our function can be None

            if not value is None:
                return value
    elif isinstance(expr, ClassDef):

        methods = expr.methods_as_dict
        init_method = methods['__init__']
        return get_initial_value(init_method, var)

    # ...

    return Nil()


# ...

# ... TODO: improve and make it recursive

def get_iterable_ranges(it, var_name=None):
    """Returns ranges of an iterable object."""

    if isinstance(it, Variable):
        if it.cls_base is None:
            raise TypeError('iterable must be an iterable Variable object'
                            )

        # ...

        def _construct_arg_Range(name):
            if not isinstance(name, DottedName):
                raise TypeError('Expecting a DottedName, given  {0}'.format(type(name)))

            if not var_name:
                return DottedName(it.name.name[0], name.name[1])
            else:
                return DottedName(var_name, name.name[1])

        # ...

        cls_base = it.cls_base

        if isinstance(cls_base, PythonRange):
            if not isinstance(it.name, DottedName):
                raise TypeError('Expecting a DottedName, given  {0}'.format(type(it.name)))

            args = []
            for i in [cls_base.start, cls_base.stop, cls_base.step]:
                if isinstance(i, Variable):
                    arg_name = _construct_arg_Range(i.name)
                    arg = i.clone(arg_name)
                elif isinstance(i, IndexedElement):
                    arg_name = _construct_arg_Range(i.base.name)
                    base = i.base.clone(arg_name)
                    indices = i.indices
                    arg = base[indices]
                else:
                    raise TypeError('Wrong type, given {0}'.format(type(i)))
                args += [arg]

            return [PythonRange(*args)]

    elif isinstance(it, ConstructorCall):
        cls_base = it.this.cls_base

        # arguments[0] is 'self'

        args = []
        kwargs = {}
        for a in it.arguments[1:]:
            if isinstance(a, dict):

                # we add '_' tp be conform with the private variables convention

                kwargs['{0}'.format(a['key'])] = a['value']
            else:
                args.append(a)

        # TODO improve

        params = args

#        for k,v in kwargs:
#            params.append(k)

    methods = cls_base.methods_as_dict
    init_method = methods['__init__']

    args = init_method.arguments[1:]
    args = [str(i) for i in args]

    # ...

    it_method = methods['__iter__']
    targets = []
    starts = []
    for stmt in it_method.body:
        if isinstance(stmt, Assign):
            targets.append(stmt.lhs)
            starts.append(stmt.lhs)

    names = []
    for i in starts:
        if isinstance(i, IndexedElement):
            names.append(str(i.base))
        else:
            names.append(str(i))
    names = list(set(names))

    inits = {}
    for stmt in init_method.body:
        if isinstance(stmt, Assign):
            if str(stmt.lhs) in names:
                expr = stmt.rhs
                for (a_old, a_new) in zip(args, params):
                    dtype = datatype(stmt.rhs.dtype)
                    v_old = Variable(dtype, a_old)
                    if isinstance(a_new, (IndexedElement, str, Variable)):
                        v_new = Variable(dtype, a_new)
                    else:
                        v_new = a_new
                    expr = subs(expr, v_old, v_new)
                    inits[str(stmt.lhs)] = expr

    _starts = []
    for i in starts:
        if isinstance(i, IndexedElement):
            _starts.append(i.base)
        else:
            _starts.append(i)
    starts = [inits[str(i)] for i in _starts]

    # ...

    def _find_stopping_criterium(stmts):
        for stmt in stmts:
            if isinstance(stmt, If):
                if not len(stmt.args) == 2:
                    raise ValueError('Wrong __next__ pattern')

                (ct, et) = stmt.args[0]
                (cf, ef) = stmt.args[1]

                for i in et:
                    if isinstance(i, Raise):
                        return cf

                for i in ef:
                    if isinstance(i, Raise):
                        return ct

                raise TypeError('Wrong type for __next__ pattern')

        return None

    # ...

    # ...

    def doit(expr, targets):
        if isinstance(expr, Relational):
            if str(expr.lhs) in targets and expr.rel_op in ['<', '<=']:
                return expr.rhs
            elif str(expr.rhs) in targets and expr.rel_op in ['>', '>='
                    ]:
                return expr.lhs
            else:
                return None
        elif isinstance(expr, sp_And):
            return [doit(a, targets) for a in expr.args]
        else:
            raise TypeError('Expecting And logical expression.')

    # ...

    # ...

    next_method = methods['__next__']
    ends = []
    cond = _find_stopping_criterium(next_method.body)

    # TODO treate case of cond with 'and' operation
    # TODO we should avoid using str
    #      must change target from DottedName to Variable

    targets = [str(i) for i in targets]
    ends = doit(cond, targets)

    # TODO not use str

    if not isinstance(ends, (list, tuple)):
        ends = [ends]

    names = []
    for i in ends:
        if isinstance(i, IndexedElement):
            names.append(str(i.base))
        else:
            names.append(str(i))
    names = list(set(names))

    inits = {}
    for stmt in init_method.body:
        if isinstance(stmt, Assign):
            if str(stmt.lhs) in names:
                expr = stmt.rhs
                for (a_old, a_new) in zip(args, params):
                    dtype = datatype(stmt.rhs.dtype)
                    v_old = Variable(dtype, a_old)
                    if isinstance(a_new, (IndexedElement, str, Variable)):
                        v_new = Variable(dtype, a_new)
                    else:
                        v_new = a_new
                    expr = subs(expr, v_old, v_new)
                    inits[str(stmt.lhs)] = expr

    _ends = []
    for i in ends:
        if isinstance(i, IndexedElement):
            _ends.append(i.base)
        else:
            _ends.append(i)
    ends = [inits[str(i)] for i in _ends]


    if not len(ends) == len(starts):
        raise ValueError('wrong number of starts/ends')

    # ...

    return [PythonRange(s, e, 1) for (s, e) in zip(starts, ends)]

class ParserResult(Basic):
    __slots__ = ('_program','_module','_prog_name','_mod_name')
    _attribute_nodes = ('_program','_module')

    def __init__(
        self,
        program   = None,
        module    = None,
        mod_name  = None,
        prog_name = None,
        ):

        if program is not None  and not isinstance(program, CodeBlock):
            raise TypeError('Program must be a CodeBlock')

        if module is not None  and not isinstance(module, CodeBlock):
            raise TypeError('Module must be a CodeBlock')

        if program is not None and module is not None:
            if mod_name is None:
                raise TypeError('Please provide module name')
            elif not isinstance(mod_name, str):
                raise TypeError('Module name must be a string')
            if prog_name is None:
                raise TypeError('Please provide program name')
            elif not isinstance(prog_name, str):
                raise TypeError('Program name must be a string')

        self._program   = program
        self._module    = module
        self._prog_name = prog_name
        self._mod_name  = mod_name
        super().__init__()


    @property
    def program(self):
        return self._program

    @property
    def module(self):
        return self._module

    @property
    def prog_name(self):
        return self._prog_name

    @property
    def mod_name(self):
        return self._mod_name

    def has_additional_module(self):
        return self.program is not None and self.module is not None

    def is_program(self):
        return self.program is not None

    def get_focus(self):
        if self.is_program():
            return self.program
        else:
            return self.module

    def __reduce_ex__(self, i):
        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments.
        """
        kwargs = dict(
        program = self.program,
        module  = self.module,
        prog_name = self.prog_name,
        mod_name  = self.mod_name)
        return (apply, (self.__class__, (), kwargs))


class TaskFunctionCall(FunctionCall):
    pass

class Task(Basic):
    """
    """
    __slots__ = ('_stmt','_inputs','_outputs','_preceders', '_shareds', '_should_wait')
    _attribute_nodes = ('_stmt','_inputs','_preceders')

    def __init__(self, stmt, inputs = (), outputs = dict(), shareds = set(), preceders = ()):
        self._stmt        = stmt
        self._inputs      = inputs
        self._outputs     = outputs
        self._preceders   = preceders
        self._shareds     = shareds
        self._should_wait = False
        super().__init__()

    @property
    def stmt(self):
        return self._stmt

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def preceders(self):
        return self._preceders

    @property
    def shareds(self):
        return self._shareds

    @property
    def should_wait(self):
        return self._should_wait

    @should_wait.setter
    def should_wait(self, value):
        self._should_wait = value

class TaskMaster(Task):
    __slots__ = ('_num_threads',)

    def __init__(self, num_threads, *args, **kwargs):
        self._num_threads = num_threads
        super().__init__(*args, **kwargs)

    @property
    def num_threads(self):
        return self._num_threads

#==============================================================================
def process_shape(shape):
    if not hasattr(shape,'__iter__'):
        shape = [shape]

    new_shape = []
    for s in shape:
        if isinstance(s,(LiteralInteger, Variable, Slice, PyccelAstNode, FunctionCall)):
            new_shape.append(s)
        elif isinstance(s, int):
            new_shape.append(LiteralInteger(s))
        else:
            raise TypeError('shape elements cannot be '+str(type(s))+'. They must be one of the following types: LiteralInteger, Variable, Slice, PyccelAstNode, int, FunctionCall')
    return tuple(new_shape)


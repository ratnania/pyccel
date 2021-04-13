# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" This module contains all classes which are used to handle memory block labels at
different stages of pyccel. Memory block labels are usually either Variables or Indexed
variables
"""
import inspect

from pyccel.errors.errors import Errors

from .basic     import Basic, PyccelAstNode
from .datatypes import (datatype, DataType,
                        NativeInteger, NativeBool, NativeReal,
                        NativeComplex, NativeGeneric,
                        default_precision)
from .internals import PyccelArraySize, Slice
from .literals  import LiteralInteger, Nil
from .operators import (PyccelMinus, PyccelDiv, PyccelMul,
                        PyccelUnarySub, PyccelAdd)

errors = Errors()

__all__ = (
    'DottedName',
    'DottedVariable',
    'IndexedElement',
    'TupleVariable',
    'ValuedVariable',
    'Variable',
    'VariableAddress'
)

class Variable(PyccelAstNode):

    """Represents a typed variable.

    Parameters
    ----------
    dtype : str, DataType
        The type of the variable. Can be either a DataType,
        or a str (bool, int, real).

    name : str, list, DottedName
        The name of the variable represented. This can be either a string
        or a dotted name, when using a Class attribute.

    rank : int
        used for arrays. [Default value: 0]

    allocatable: bool
        used for arrays, if we need to allocate memory [Default value: False]

    is_stack_array: bool
        used for arrays, if memory should be allocated on the stack [Default value: False]

    is_pointer: bool
        if object is a pointer [Default value: False]

    is_target: bool
        if object is pointed to by another variable [Default value: False]

    is_optional: bool
        if object is an optional argument of a function [Default value: False]

    shape: int or list
        shape of the array. [Default value: None]

    cls_base: class
        class base if variable is an object or an object member [Default value: None]

    order : str
        used for arrays. Indicates whether the data is stored in C or Fortran format in memory [Default value: 'C']

    precision : str
        Precision of the data type [Default value: depends on the datatype]

    is_argument: bool
        if object is the argument of a function [Default value: False]

    is_kwonly: bool
        if object is an argument which can only be specified using its keyword

    is_const: bool
        if object is a const argument of a function [Default value: False]

    is_temp: bool
        Indicates if this symbol represents a temporary variable created by Pyccel,
        and was not present in the original Python code [default value : False].

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> Variable('int', 'n')
    n
    >>> n = 4
    >>> Variable('real', 'x', rank=2, shape=(n,2), allocatable=True)
    x
    >>> Variable('int', DottedName('matrix', 'n_rows'))
    matrix.n_rows
    """
    __slots__ = ('_name', '_alloc_shape', '_allocatable', '_is_const', '_is_pointer',
            '_is_stack_array', '_is_target', '_is_optional', '_allows_negative_indexes',
            '_cls_base', '_is_argument', '_is_kwonly', '_is_temp','_dtype','_precision',
            '_rank','_shape','_order')
    _attribute_nodes = ()

    def __init__(
        self,
        dtype,
        name,
        *,
        rank=0,
        allocatable=False,
        is_stack_array = False,
        is_pointer=False,
        is_const=False,
        is_target=False,
        is_optional=False,
        shape=None,
        cls_base=None,
        order='C',
        precision=0,
        is_argument=False,
        is_kwonly=False,
        is_temp =False,
        allows_negative_indexes=False
        ):
        super().__init__()

        # ------------ Variable Properties ---------------
        # if class attribute
        if isinstance(name, str):
            name = name.split(""".""")
            if len(name) == 1:
                name = name[0]
            else:
                name = DottedName(*name)

        if name == '':
            raise ValueError("Variable name can't be empty")

        if not isinstance(name, (str, DottedName)):
            raise TypeError('Expecting a string or DottedName, given {0}'.format(type(name)))
        self._name = name

        if not isinstance(allocatable, bool):
            raise TypeError('allocatable must be a boolean.')
        self.allocatable = allocatable

        if not isinstance(is_const, bool):
            raise TypeError('is_const must be a boolean.')
        self._is_const = is_const

        if not isinstance(is_stack_array, bool):
            raise TypeError('is_stack_array must be a boolean.')
        self._is_stack_array = is_stack_array

        if not isinstance(is_pointer, bool):
            raise TypeError('is_pointer must be a boolean.')
        self.is_pointer = is_pointer

        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self.is_target = is_target

        if not isinstance(is_optional, bool):
            raise TypeError('is_optional must be a boolean.')
        self._is_optional = is_optional

        if not isinstance(allows_negative_indexes, bool):
            raise TypeError('allows_negative_indexes must be a boolean.')
        self._allows_negative_indexes = allows_negative_indexes

        self._cls_base       = cls_base
        self._order          = order
        self._is_argument    = is_argument
        self._is_kwonly      = is_kwonly
        self._is_temp        = is_temp

        # ------------ PyccelAstNode Properties ---------------
        if isinstance(dtype, str) or str(dtype) == '*':

            dtype = datatype(str(dtype))
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        if not isinstance(rank, int):
            raise TypeError('rank must be an instance of int.')

        if rank == 0:
            shape = ()

        if shape is None:
            shape = tuple(None for i in range(rank))

        if not precision:
            if isinstance(dtype, NativeInteger):
                precision = default_precision['int']
            elif isinstance(dtype, NativeReal):
                precision = default_precision['real']
            elif isinstance(dtype, NativeComplex):
                precision = default_precision['complex']
            elif isinstance(dtype, NativeBool):
                precision = default_precision['bool']
        if not isinstance(precision,int) and precision is not None:
            raise TypeError('precision must be an integer or None.')

        self._alloc_shape = shape
        self._dtype = dtype
        self._rank  = rank
        self._shape = self.process_shape(shape)
        self._precision = precision

    def process_shape(self, shape):
        """ Simplify the provided shape and ensure it
        has the expected format

        The provided shape is the shape used to create
        the object. In most cases where the shape is
        required we do not require this expression
        (which can be quite long). This function therefore
        replaces those expressions with calls to
        PyccelArraySize
        """
        if not hasattr(shape,'__iter__'):
            shape = [shape]

        new_shape = []
        for i,s in enumerate(shape):
            if self.is_pointer:
                # Shape of a pointer can change
                new_shape.append(PyccelArraySize(self, LiteralInteger(i)))
            elif isinstance(s,(LiteralInteger, PyccelArraySize)):
                new_shape.append(s)
            elif isinstance(s, int):
                new_shape.append(LiteralInteger(s))
            elif s is None or isinstance(s, PyccelAstNode):
                new_shape.append(PyccelArraySize(self, LiteralInteger(i)))
            else:
                raise TypeError('shape elements cannot be '+str(type(s))+'. They must be one of the following types: LiteralInteger,'
                                'Variable, Slice, PyccelAstNode, int, Function')
        return tuple(new_shape)

    @property
    def name(self):
        """ Name of the variable
        """
        return self._name

    @property
    def alloc_shape(self):
        """ Shape of the variable at allocation

        The shape used in pyccel is usually simplified to contain
        only Literals and PyccelArraySizes but the shape for
        the allocation of x cannot be `Shape(x)`
        """
        return self._alloc_shape

    @property
    def allocatable(self):
        """ Indicates whether a Variable has a dynamic size
        """
        return self._allocatable

    @allocatable.setter
    def allocatable(self, allocatable):
        if not isinstance(allocatable, bool):
            raise TypeError('allocatable must be a boolean.')
        self._allocatable = allocatable

    @property
    def cls_base(self):
        """ Class from which the Variable inherits
        """
        return self._cls_base

    @property
    def is_const(self):
        """ Indicates if the Variable is constant
        within its context
        """
        return self._is_const

    @property
    def is_pointer(self):
        """ Indicates if the Variable is a label for
        something which points to another object.
        In other words, the Variable does not own its data
        """
        return self._is_pointer

    @is_pointer.setter
    def is_pointer(self, is_pointer):
        if not isinstance(is_pointer, bool):
            raise TypeError('is_pointer must be a boolean.')
        self._is_pointer = is_pointer

    @property
    def is_temp(self):
        """
        Indicates if this symbol represents a temporary variable created by Pyccel,
		and was not present in the original Python code [default value : False].
        """
        return self._is_temp

    @property
    def is_target(self):
        """ Indicates if the data in this Variable is
        shared with (pointed at by) another Variable
        """
        return self._is_target

    @is_target.setter
    def is_target(self, is_target):
        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self._is_target = is_target

    @property
    def is_optional(self):
        """ Indicates if the Variable is optional
        in this context
        """
        return self._is_optional

    @property
    def is_stack_array(self):
        """ Indicates whether an array is allocated
        on the stack
        """
        return self._is_stack_array

    @is_stack_array.setter
    def is_stack_array(self, is_stack_array):
        self._is_stack_array = is_stack_array

    @property
    def allows_negative_indexes(self):
        """ Indicates whether negative values can be
        used to index this Variable
        """
        return self._allows_negative_indexes

    @allows_negative_indexes.setter
    def allows_negative_indexes(self, allows_negative_indexes):
        self._allows_negative_indexes = allows_negative_indexes

    @property
    def is_argument(self):
        """ Indicates whether the Variable is
        a function argument in this context
        """
        return self._is_argument

    @property
    def is_kwonly(self):
        """ If the Variable is an argument then this
        indicates whether the argument is a keyword
        only argument
        """
        return self._is_kwonly

    @property
    def is_ndarray(self):
        """user friendly method to check if the variable is an ndarray:
            1. have a rank > 0
            2. dtype is one among {int, bool, real, complex}
        """

        if self.rank == 0:
            return False
        return isinstance(self.dtype, (NativeInteger, NativeBool,
                          NativeReal, NativeComplex))

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return 'Variable({}, dtype={})'.format(repr(self.name), repr(self.dtype))

    def __eq__(self, other):
        if type(self) is type(other):
            return self._name == other.name
        return False

    def __hash__(self):
        return hash((type(self).__name__, self._name))

    def inspect(self):
        """inspects the variable."""

        print('>>> Variable')
        print( '  name           = {}'.format(self.name))
        print( '  dtype          = {}'.format(self.dtype))
        print( '  precision      = {}'.format(self.precision))
        print( '  rank           = {}'.format(self.rank))
        print( '  order          = {}'.format(self.order))
        print( '  allocatable    = {}'.format(self.allocatable))
        print( '  shape          = {}'.format(self.shape))
        print( '  cls_base       = {}'.format(self.cls_base))
        print( '  is_pointer     = {}'.format(self.is_pointer))
        print( '  is_target      = {}'.format(self.is_target))
        print( '  is_optional    = {}'.format(self.is_optional))
        print( '<<<')

    def clone(self, name, new_class = None, **kwargs):
        """
        Create a new Variable object of the chosen class
        with the provided name and options

        Parameters
        ==========
        name      : str
                    The name of the new Variable
        new_class : type
                    The class of the new Variable
                    The default is the same class
        kwargs    : dict
                    Dictionary containing any keyword-value
                    pairs which are valid constructor keywords
        """

        if (new_class is None):
            cls = self.__class__
        else:
            cls = new_class

        args = inspect.signature(Variable.__init__)
        new_kwargs = {k:getattr(self, '_'+k) \
                            for k in args.parameters.keys() \
                            if '_'+k in dir(self)}
        new_kwargs.update(kwargs)
        new_kwargs['name'] = name

        return cls(**new_kwargs)

    def rename(self, newname):
        """ Forbidden method for renaming the variable
        """
        # The name is part of the hash so it must never change
        raise RuntimeError('Cannot modify hash definition')

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
        args = (
            self.dtype,
            self.name)
        kwargs = {
            'rank' : self.rank,
            'allocatable': self.allocatable,
            'is_pointer':self.is_pointer,
            'is_optional':self.is_optional,
            'shape':self.shape,
            'cls_base':self.cls_base,
            }

        apply = lambda func, args, kwargs: func(*args, **kwargs)
        out =  (apply, (Variable, args, kwargs))
        return out

    def __getitem__(self, *args):

        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]

        if self.rank < len(args):
            raise IndexError('Rank mismatch.')

        return IndexedElement(self, *args)

    def invalidate_node(self):
        # Don't invalidate Variables
        pass

class DottedName(Basic):

    """
    Represents a dotted object.

    Examples
    --------
    >>> from pyccel.ast.core import DottedName
    >>> DottedName('matrix', 'n_rows')
    matrix.n_rows
    >>> DottedName('pyccel', 'stdlib', 'parallel')
    pyccel.stdlib.parallel
    """
    __slots__ = ('_name',)
    _attribute_nodes = ()

    def __init__(self, *args):

        self._name = args
        super().__init__()

    @property
    def name(self):
        """ The different components of the name
        (these were separated by dots)
        """
        return self._name

    def __str__(self):
        return """.""".join(str(n) for n in self.name)

class ValuedVariable(Variable):

    """Represents a valued variable in the code.

    Parameters
    ----------
    variable: Variable
        A single variable
    value: Variable, or instance of Native types
        value associated to the variable

    Examples
    --------
    >>> from pyccel.ast.core import ValuedVariable
    >>> n  = ValuedVariable('int', 'n', value=4)
    >>> n
    n := 4
    """
    __slots__ = ('_value',)
    _attribute_nodes = ('_value',)

    def __init__(self, *args, **kwargs):

        # if value is not given, we set it to Nil
        self._value = kwargs.pop('value', Nil())
        super().__init__(*args, **kwargs)

    @property
    def value(self):
        """ Default value of the variable
        """
        return self._value

    def __str__(self):
        name = str(self.name)
        value = str(self.value)
        return '{0}={1}'.format(name, value)

class TupleVariable(Variable):

    """Represents a tuple variable in the code.

    Parameters
    ----------
    arg_vars: Iterable
        Multiple variables contained within the tuple

    Examples
    --------
    >>> from pyccel.ast.core import TupleVariable, Variable
    >>> v1 = Variable('int','v1')
    >>> v2 = Variable('bool','v2')
    >>> n  = TupleVariable([v1, v2],'n')
    >>> n
    n
    """
    __slots__ = ('_vars','_inconsistent_shape','_is_homogeneous')
    _attribute_nodes = ('_vars',)

    def __init__(self, arg_vars, dtype, name, *args, **kwargs):
        self._vars = tuple(arg_vars)
        self._inconsistent_shape = not all(arg_vars[0].shape==a.shape   for a in arg_vars[1:])
        self._is_homogeneous = not dtype is NativeGeneric()
        super().__init__(dtype, name, *args, **kwargs)

    def get_vars(self):
        """ Get the variables saved internally in the tuple
        (used for inhomogeneous variables)
        """
        return tuple(self[i] for i in range(len(self._vars)))

    def get_var(self, variable_idx):
        """ Get the n-th variable saved internally in the
        tuple (used for inhomogeneous variables)

        Parameters
        ==========
        variable_idx : int/LiteralInteger
                       The index of the variable which we
                       wish to collect
        """
        assert(not self._is_homogeneous)
        return self._vars[variable_idx]

    def rename_var(self, variable_idx, new_name):
        """ Rename the n-th variable saved internally in the
        tuple (used for inhomogeneous variables)

        Parameters
        ==========
        variable_idx : int/LiteralInteger
                       The index of the variable which we
                       wish to collect
        new_name     : str
                       The new name of the variable
        """
        assert(not self._is_homogeneous)
        self._vars[variable_idx].rename(new_name)

    def __getitem__(self, idx):
        if self._is_homogeneous:
            return Variable.__getitem__(self, idx)
        else:
            if isinstance(idx, tuple):
                sub_idx = idx[1:]
                idx = idx[0]
            else:
                sub_idx = []

            var = self.get_var(idx)

            if len(sub_idx) > 0:
                return var[sub_idx]
            else:
                return var

    def __iter__(self):
        return self._vars.__iter__()

    def __len__(self):
        return len(self._vars)

    @property
    def inconsistent_shape(self):
        """ Indicates whether all objects in the tuple have the
        same shape
        """
        return self._inconsistent_shape

    @property
    def is_homogeneous(self):
        """ Indicates if all objects in the tuple have the
        same datatype
        # TODO: Indicates if all objects in the tuple have the
        same properties
        """
        return self._is_homogeneous

    @is_homogeneous.setter
    def is_homogeneous(self, is_homogeneous):
        self._is_homogeneous = is_homogeneous

    @Variable.allocatable.setter
    def allocatable(self, allocatable):
        if not isinstance(allocatable, bool):
            raise TypeError('allocatable must be a boolean.')
        self._allocatable = allocatable
        for var in self._vars:
            var.allocatable = allocatable

    @Variable.is_pointer.setter
    def is_pointer(self, is_pointer):
        if not isinstance(is_pointer, bool):
            raise TypeError('is_pointer must be a boolean.')
        self._is_pointer = is_pointer
        for var in self._vars:
            var.is_pointer = is_pointer

    @Variable.is_target.setter
    def is_target(self, is_target):
        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self._is_target = is_target
        for var in self._vars:
            var.is_target = is_target

class Constant(ValuedVariable):

    """

    Examples
    --------

    """
    __slots__ = ()
    # The value of a constant is not a translated object
    _attribute_nodes = ()



class IndexedElement(PyccelAstNode):

    """
    Represents a mathematical object with indices.

    Examples
    --------
    >>> from pyccel.ast.core import Variable, IndexedElement
    >>> A = Variable('A', dtype='int', shape=(2,3), rank=2)
    >>> i = Variable('i', dtype='int')
    >>> j = Variable('j', dtype='int')
    >>> IndexedElement(A, i, j)
    IndexedElement(A, i, j)
    >>> IndexedElement(A, i, j) == A[i, j]
    True
    """
    __slots__ = ('_label', '_indices','_dtype','_precision','_shape','_rank','_order')
    _attribute_nodes = ('_label', '_indices')

    def __init__(
        self,
        base,
        *args,
        **kw_args
        ):

        if not args:
            raise IndexError('Indexed needs at least one index.')

        self._label = base

        if PyccelAstNode.stage == 'syntactic':
            self._indices = args
            super().__init__()
            return

        self._dtype = base.dtype
        self._order = base.order
        self._precision = base.precision

        shape = base.shape
        rank  = base.rank

        # Add empty slices to fully index the object
        if len(args) < rank:
            args = args + tuple([Slice(None, None)]*(rank-len(args)))

        if any(not isinstance(a, (int, PyccelAstNode, Slice)) for a in args):
            errors.report("Index is not of valid type",
                    symbol = args, severity = 'fatal')

        self._indices = tuple(LiteralInteger(a) if isinstance(a, int) else a for a in args)
        super().__init__()

        # Calculate new shape

        if shape is not None:
            new_shape = []
            from .mathext import MathCeil
            for a,s in zip(args, shape):
                if isinstance(a, Slice):
                    start = a.start
                    stop  = a.stop if a.stop is not None else s
                    step  = a.step
                    if isinstance(start, PyccelUnarySub):
                        start = PyccelAdd(s, start, simplify=True)
                    if isinstance(stop, PyccelUnarySub):
                        stop = PyccelAdd(s, stop, simplify=True)

                    _shape = stop if start is None else PyccelMinus(stop, start, simplify=True)
                    if step is not None:
                        if isinstance(step, PyccelUnarySub):
                            start = s if a.start is None else start
                            _shape = start if a.stop is None else PyccelMinus(start, stop, simplify=True)
                            step = PyccelUnarySub(step)

                        _shape = MathCeil(PyccelDiv(_shape, step, simplify=True))
                    new_shape.append(_shape)
            self._shape = tuple(new_shape)
            self._rank  = len(new_shape)
        else:
            new_rank = rank
            for i in range(rank):
                if not isinstance(args[i], Slice):
                    new_rank -= 1
            self._rank = new_rank

    @property
    def base(self):
        """ The object which is indexed
        """
        return self._label

    @property
    def indices(self):
        """ A tuple of indices used to index the variable
        """
        return self._indices

    def __str__(self):
        return '{}[{}]'.format(self.base, ','.join(str(i) for i in self.indices))

    def __repr__(self):
        return '{}[{}]'.format(repr(self.base), ','.join(repr(i) for i in self.indices))

    def __getitem__(self, *args):

        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]

        if self.rank < len(args):
            raise IndexError('Rank mismatch.')

        new_indexes = []
        j = 0
        for i in self.indices:
            if isinstance(i, Slice) and j<len(args):
                if j == 0:
                    i = args[j]
                elif i.step == 1:
                    i = PyccelAdd(i.start, args[j], simplify = True)
                else:
                    i = PyccelAdd(i.start, PyccelMul(i.step, args[j], simplify=True), simplify = True)
                j += 1
            new_indexes.append(i)
        return IndexedElement(self.base, *new_indexes)

class VariableAddress(PyccelAstNode):

    """Represents the address of a variable.
    E.g. In C
    VariableAddress(Variable('int','a'))                     is  &a
    VariableAddress(Variable('int','a', is_pointer=True))    is   a
    """
    __slots__ = ('_variable','_dtype','_precision','_shape','_rank','_order')
    _attribute_nodes = ('_variable',)

    def __init__(self, variable):
        if not isinstance(variable, Variable):
            raise TypeError('variable must be a variable')
        self._variable = variable

        self._shape     = variable.shape
        self._rank      = variable.rank
        self._dtype     = variable.dtype
        self._precision = variable.precision
        self._order     = variable.order
        super().__init__()

    @property
    def variable(self):
        """ The variable whose address is of interest
        """
        return self._variable

class DottedVariable(Variable):

    """
    Represents a dotted variable. This is usually
    a variable which is a member of a class

    E.g.
    a = AClass()
    a.b = 3

    In this case b is a DottedVariable where the lhs
    is a
    """
    __slots__ = ('_lhs',)
    _attribute_nodes = ('_lhs',)

    def __init__(self, *args, lhs, **kwargs):
        self._lhs = lhs
        super().__init__(*args, **kwargs)

    @property
    def lhs(self):
        """ The object before the final dot in the
        dotted variable

        e.g. for the DottedVariable:
        a.b
        The lhs is a
        """
        return self._lhs

    def __eq__(self, other):
        if type(self) is type(other):
            return self.name == other.name and self.lhs == other.lhs

        return False

    def __hash__(self):
        return hash((type(self).__name__, self.name, self.lhs))

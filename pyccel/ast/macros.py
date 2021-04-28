# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains all classes and functions used for handling macros.
"""
from .basic          import PyccelAstNode
from .datatypes      import default_precision
from .datatypes      import NativeInteger, NativeGeneric
from .internals      import PyccelSymbol
from .variable       import Variable

__all__ = (
    'Macro',
    'MacroCount',
    'MacroShape',
    'MacroType',
    'construct_macro'
)

#==============================================================================
class Macro(PyccelAstNode):
    """."""
    __slots__ = ('_argument',)
    _name = '__UNDEFINED__'
    _attribute_nodes = ()

    def __init__(self, argument):
        if not isinstance(argument, (PyccelSymbol, Variable)):
            raise TypeError("Argument must be a Pyccelsymbol or a Variable not {}".format(type(argument)))

        self._argument = argument
        super().__init__()

    @property
    def argument(self):
        return self._argument

    @property
    def name(self):
        return self._name

#==============================================================================
class MacroShape(Macro):
    """."""
    __slots__ = ('_index','_rank','_shape')
    _name      = 'shape'
    _dtype     = NativeInteger()
    _precision = default_precision['integer']
    _order     = None

    def __init__(self, argument, index=None):
        if index is not None:
            self._rank = 0
            self._shape = ()
        elif PyccelAstNode.stage != "syntactic":
            self._rank      = int(argument.rank>1)
            self._shape     = (argument.rank,)
        else:
            self._rank      = 1
            self._shape     = ()
        self._index = index
        super().__init__(argument)

    @property
    def index(self):
        return self._index

    def __str__(self):
        if self.index is None:
            return 'MacroShape({})'.format(str(self.argument))
        else:
            return 'MacroShape({}, {})'.format(str(self.argument),
                                               str(self.index))

#==============================================================================
class MacroType(Macro):
    """."""
    __slots__ = ()
    _name      = 'dtype'
    _dtype     = NativeGeneric()
    _precision = 0
    _rank      = 0
    _shape     = ()
    _order     = None

    def __str__(self):
        return 'MacroType({})'.format(str(self.argument))

#==============================================================================
class MacroCount(Macro):
    """."""
    __slots__ = ()
    _name      = 'count'
    _dtype     = NativeInteger()
    _precision = default_precision['integer']
    _rank      = 0
    _shape     = ()
    _order     = None

    def __str__(self):
        return 'MacroCount({})'.format(str(self.argument))





def construct_macro(name, argument, parameter=None):
    """."""
    # TODO add available macros: shape, len, dtype
    if not isinstance(name, str):
        raise TypeError('name must be of type str')

    if name == 'shape':
        return MacroShape(argument, index=parameter)
    elif name == 'dtype':
        return MacroType(argument)
    elif name == 'count':
        return MacroCount(argument)


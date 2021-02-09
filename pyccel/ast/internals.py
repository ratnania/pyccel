# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
File containing basic classes which are used throughout pyccel.
To avoid circular imports this file should only import from basic, datatypes, and literals
"""
from .basic     import Basic, PyccelAstNode
from .datatypes import NativeInteger, default_precision
from .literals  import LiteralInteger

__all__ = (
    'PyccelInternalFunction',
    'PyccelArraySize'
)


class PyccelInternalFunction(PyccelAstNode):
    """ Abstract class used by function calls
    which are translated to Pyccel objects
    """
    def __init__(self, *args):
        PyccelAstNode.__init__(self)
        self._args   = tuple(args)

    @property
    def args(self):
        """ The arguments passed to the function
        """
        return self._args


class PyccelArraySize(PyccelInternalFunction):
    """
    Class representing a call to a function which would
    return the shape of an object in a given dimension

    Parameters
    ==========
    arg   : PyccelAstNode
            A PyccelAstNode of unknown shape
    index : int
            The dimension along which the shape is
            provided
    """

    def __init__(self, arg, index):
        if not isinstance(arg, (list,
                                tuple,
                                PyccelAstNode)):
            raise TypeError('Unknown type of  %s.' % type(arg))

        if isinstance(index, int):
            index = LiteralInteger(index)
        elif not isinstance(index, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(index))

        PyccelInternalFunction.__init__(self, arg, index)
        self._arg   = arg
        self._index = index
        self._dtype = NativeInteger()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['integer']

    @property
    def arg(self):
        """ Object whose size is investigated
        """
        return self._arg

    @property
    def index(self):
        """ Dimension along which the size is calculated
        """
        return self._index

    def __str__(self):
        return 'Shape({},{})'.format(str(self.arg), str(self.index))

class Slice(Basic):

    """Represents a slice in the code.

    Parameters
    ----------
    start : Symbol or int
        starting index

    stop : Symbol or int
        ending index

    step : Symbol or int default None

    Examples
    --------
    >>> from sympy import symbols
    >>> from pyccel.ast.core import Slice
    >>> start, end, step = symbols('start, stop, step', integer=True)
    >>> Slice(start, stop)
    start : stop
    >>> Slice(None, stop)
     : stop
    >>> Slice(start, None)
    start :
    >>> Slice(start, stop, step)
    start : stop : step
    """

    def __new__(cls, start, stop, step = None):
        return Basic.__new__(cls, start, stop, step)

    def __init__(self, start, stop, step = None):
        Basic.__init__(self)
        self._start = start
        self._stop = stop
        self._step = step
        if PyccelAstNode.stage == 'syntactic':
            return
        if start is not None and not (hasattr(start, 'dtype') and isinstance(start.dtype, NativeInteger)):
            raise TypeError('Slice start must be Integer or None')
        if stop is not None and not (hasattr(stop, 'dtype') and isinstance(stop.dtype, NativeInteger)):
            raise TypeError('Slice stop must be Integer or None')
        if step is not None and not (hasattr(step, 'dtype') and isinstance(step.dtype, NativeInteger)):
            raise TypeError('Slice step must be Integer or None')

    @property
    def start(self):
        """ Index where the slicing of the object starts
        """
        return self._start

    @property
    def stop(self):
        """ Index until which the slicing takes place
        """
        return self._stop

    @property
    def step(self):
        """ The difference between each index of the
        objects in the slice
        """
        return self._step

    def _sympystr(self, printer):
        """ sympy equivalent of __str__"""
        sstr = printer.doprint
        if self.start is None:
            start = ''
        else:
            start = sstr(self.start)
        if self.stop is None:
            stop = ''
        else:
            stop = sstr(self.stop)
        return '{0} : {1}'.format(start, stop)

    def __str__(self):
        if self.start is None:
            start = ''
        else:
            start = str(self.start)
        if self.stop is None:
            stop = ''
        else:
            stop = str(self.stop)
        return '{0} : {1}'.format(start, stop)


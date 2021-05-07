# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201, missing-function-docstring

# TODO: - Pow
#       - If
#       - FunctionalFor

"""
This module provides us with functions and objects that allow us to compute
the arithmetic complexity a of a program.

Example
-------

"""


from sympy import count_ops as sympy_count_ops
from sympy import Symbol

from pyccel.ast.core     import Comment, EmptyNode
from pyccel.ast.numpyext  import NumpyZeros, NumpyOnes
from pyccel.ast.sympy_helper import pyccel_to_sympy
from pyccel.complexity.basic import Complexity

__all__ = ["OpComplexity"]

# ...
ADD = Symbol('ADD')
SUB = Symbol('SUB')
MUL = Symbol('MUL')
DIV = Symbol('DIV')
IDIV = Symbol('IDIV')
ABS = Symbol('ABS')

op_registry = {
    '+': ADD,
    '-': SUB,
    '*': MUL,
    '/': DIV,
#    '%': MOD,
    }
# ...

# ==============================================================================
class OpComplexity(Complexity):
    """class for Operation complexity computation."""

    def _cost_Variable(self, expr, **settings):
        return 0

    def _cost_IndexedElement(self, expr, **settings):
        return 0

    def _cost_PyccelArraySize(self, expr, **settings):
        return 0

    def _cost_NumpyZeros(self, expr, **settings):
        return 0

    def _cost_NumpyOnes(self, expr, **settings):
        return 0

    def _cost_Allocate(self, expr, **settings):
        return 0

    def _cost_Deallocate(self, expr, **settings):
        return 0

    def _cost_NumpyFloor(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('FLOOR')

    def _cost_NumpyExp(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('EXP')

    def _cost_NumpyLog(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('LOG')

    def _cost_NumpySqrt(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('SQRT')

    def _cost_NumpySin(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('SIN')

    def _cost_NumpyCos(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('COS')

    def _cost_NumpyTan(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('TAN')

    def _cost_NumpyArcsin(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('ARCSIN')

    def _cost_NumpyArccos(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('ARCCOS')

    def _cost_NumpyArctan(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('ARCTAN')

    def _cost_NumpyArctan2(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('ARCTAN2')

    def _cost_NumpySinh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('SINH')

    def _cost_NumpyCosh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('COSH')

    def _cost_NumpyTanh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('TANH')

    def _cost_NumpyArcsinh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('ARCSINH')

    def _cost_NumpyArccosh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('ARCCOSH')

    def _cost_NumpyArctanh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + Symbol('ARCTANH')

    def _cost_PyccelAdd(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + ADD

    def _cost_PyccelMinus(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + SUB

    def _cost_PyccelDiv(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + DIV

    def _cost_PyccelFloorDiv(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + IDIV

    def _cost_PyccelMul(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + MUL

    def _cost_PythonAbs(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 0
        else:
            return ops + ABS

    def _cost_PyccelPow(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            # TODO is this correct?
            return ops +expr.args[1]
        else:
            return (int(expr.args[1])-1) * MUL

    def _cost_Assign(self, expr, **settings):
        if isinstance(expr.rhs, (Comment, EmptyNode)):
            return 0

        # TODO add other numpy array constructors
        if isinstance(expr.rhs, (NumpyZeros, NumpyOnes)):
            shape = [pyccel_to_sympy(i, self._symbol_map, self._used_names) for i in expr.rhs.shape]
            self._shapes[expr.lhs] = shape

            return 0

        ntimes = self._compute_size_lhs(expr)

        return ntimes * ( self._cost(expr.rhs, **settings) )

    def _cost_AugAssign(self, expr, **settings):
        # TODO add other numpy array constructors
        if isinstance(expr.rhs, (NumpyZeros, NumpyOnes, Comment, EmptyNode)):
            return 0

        # ...
        if self.mode:
            op = 1
        else:
            op = op_registry[expr.op]
        # ...

        ntimes = self._compute_size_lhs(expr)

        return ntimes * ( op + self._cost(expr.rhs, **settings) )

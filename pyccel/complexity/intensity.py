# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201, missing-function-docstring

# TODO - use SHAPE(...) as args for Poly => ex1.py breaks down for a function
#        like array_int32_1d_scalar_add
#      - for the moment, we should put only tests that are working (because of
#      SHAPE(...))

"""
This module provides us with functions and objects that allow us to compute
the computational intensity

Example
-------

"""

import sympy
from collections import OrderedDict

from sympy import sympify, Symbol
from sympy import Poly, LT

from pyccel.complexity.arithmetic import OpComplexity, ADD, SUB, MUL, DIV, IDIV, ABS
from pyccel.complexity.memory import MemComplexity, READ, WRITE

from sympy.abc import x
FLOOR = Symbol('FLOOR')
EXP = Symbol('EXP')
LOG = Symbol('LOG')
SQRT = Symbol('SQRT')
SIN = Symbol('SIN')
COS = Symbol('COS')
TAN = Symbol('TAN')
ARCSIN = Symbol('ARCSIN')
ARCCOS = Symbol('ARCCOS')
ARCTAN = Symbol('ARCTAN')
SINH = Symbol('SINH')
COSH = Symbol('COSH')
TANH = Symbol('TANH')
ARCSINH = Symbol('ARCSINH')
ARCCOSH = Symbol('ARCCOSH')
ARCTANH = Symbol('ARCTANH')
ARCTAN2 = Symbol('ARCTAN2')
_cost_symbols = {ADD, SUB, MUL, DIV, IDIV, ABS,
                 READ, WRITE, SIN, FLOOR, EXP, LOG, SQRT, COS, TAN, ARCSIN, ARCCOS, ARCTAN, SINH, COSH, TANH, ARCSINH, ARCCOSH, ARCTANH, ARCTAN2}

__all__ = ["ComputationalIntensity"]


# ==============================================================================
def _leading_term(expr, *args):
    """
    Returns the leading term in a sympy Polynomial.

    expr: sympy.Expr
        any sympy expression

    args: list
        list of input symbols for univariate/multivariate polynomials
    """
    expr = sympify(str(expr))
    if expr.atoms(sympy.Function):
        for shape_expr in expr.atoms(sympy.Function):
            expr = expr.subs(shape_expr, "shape_" + str(shape_expr.args[0]) + '_' + str(shape_expr.args[1]))
        P = Poly(expr, *args)
        lt = LT(P)
        for var in lt.free_symbols:
            var = str(var)
            if "shape" in var:
                lt = lt.subs(var, "SHAPE(" + var.split("_")[1] + "," + var.split("_")[2] + ")")
    else:
        P = Poly(expr, *args)
        lt = LT(P)
    return lt
# ==============================================================================
def checker(expr, temp = []):
    if isinstance(expr, sympy.core.power.Pow):
        if (expr.args[-1] < 0):
            temp.append(expr)
    for i in expr.args:
        checker(i)
    return temp

# ==============================================================================
def _intensity(f, m):
    if f * m == 0:
        return 0
    f = sympify(str(f))
    m = sympify(str(m))

    args = f.free_symbols.union(m.free_symbols) - _cost_symbols
    if not args:
        args = {x}
    multiply_args = sympify("1")
    for i in checker(f):
        multiply_args = multiply_args * i.args[0]
        args = args - {i.args[0]}
    for i in checker(m):
        multiply_args = multiply_args * i.args[0]
        args = args - {i.args[0]}
    f = f * multiply_args
    m = m * multiply_args
    args = list(args)

    # TODO _leading_term does not retrieve the right output
    lt_f = _leading_term(f, *args)
    lt_m = _leading_term(m, *args)

    return lt_f/lt_m

# ==============================================================================
class ComputationalIntensity(object):
    def __init__(self, filename_or_text):
        self._arithmetic = OpComplexity(filename_or_text)
        self._memory     = MemComplexity(filename_or_text)
        self._costs = OrderedDict()

    @property
    def arithmetic(self):
        return self._arithmetic

    @property
    def memory(self):
        return self._memory

    @property
    def costs(self):
        return self._costs

    def cost(self, mode=None):
        f = self.arithmetic.cost(mode=mode)
        m = self.memory.cost(mode=mode)

        f_costs = self.arithmetic.costs
        m_costs = self.memory.costs

        q = _intensity(f, m)

        # ...
        self._costs = OrderedDict()
        for i,fi in f_costs.items():
            mi = m_costs[i]
            qi = _intensity(fi, mi)
            self._costs[i] = qi
        # ...

        return q

# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201, missing-function-docstring

from collections import OrderedDict

from sympy import summation
from sympy import Function
from sympy import Symbol
from sympy import Poly, LT
from sympy import simplify as sp_simplify

from pyccel.parser.parser import Parser
from pyccel.ast.sympy_helper import pyccel_to_sympy
from pyccel.ast.variable import IndexedElement
from pyccel.ast.internals import Slice

#koka try deleting visual
__all__ = ["Complexity"]

SHAPE = Function('SHAPE')

# ==============================================================================
class Complexity(object):
    """Abstract class for complexity computation."""
    def __init__(self, filename_or_text):
        """Constructor for the Complexity class.

        filename_or_text: str
            name of the file containing the abstract grammar or input code to
            parse as a string.
        """

        pyccel = Parser(filename_or_text)
        self._ast = pyccel.parse()
        settings = {}
        self._ast = pyccel.annotate(**settings).ast

        # ...
        functions = OrderedDict()
        if pyccel.namespace.functions:
            functions = pyccel.namespace.functions

        for son in pyccel.sons:
            functions.update(son.namespace.functions)

        self._functions = functions
        # ...

        self._costs = OrderedDict()
        self._shapes = OrderedDict()
        self._symbol_map = {}
        self._used_names = set()
        self._visual = True
        self._mode = None

    @property
    def ast(self):
        """Returns the Abstract Syntax Tree."""
        return self._ast

    @property
    def functions(self):
        """Returns declared functions."""
        return self._functions

    @property
    def costs(self):
        """Returns costs of declared functions."""
        return self._costs

    @property
    def shapes(self):
        """Returns shapes of allocated arrays."""
        return self._shapes

    @property
    def mode(self):
        """
        Returns the complexity computation mode.

        possible values are (None, simple)

        complexity in simple mode will only return the number of operations done.
        """
        return self._mode

    @property
    def visual(self):
        return self._visual

    def cost(self, visual=True, mode=None, simplify=True, bigo=None):
        """
        Computes the complexity of the given code.

        verbose: bool
            talk more

        mode: string
            possible values are (None, simple)

        simplify: bool
            if true, calls sympy simplify function

        bigo: list
            a list containing variable names (str) for which we want to take the
            highest degree (expansion serie)
        """
        # ...
        self._visual = visual
        if mode and mode != 'simple':
            raise ValueError('not valid value for mode')
        self._mode = mode
        # ...

        # ...
        costs = OrderedDict()

        # ... first we treat declared functions
        if self.functions:
            for fname, d in self.functions.items():
                expr =  self._cost(d)

                if simplify:
                    expr = sp_simplify(expr)

                if bigo:
                    for n in bigo:
                        expr = LT(expr, Symbol(n))

                costs[fname] = expr

        self._costs.update(costs)
        # ...

        # ... then we compute the complexity for the main program
        expr = self._cost(self.ast)

        if simplify:
            expr = sp_simplify(expr)

        if bigo:
            for n in bigo:
                expr = LT(expr, Symbol(n))
        # ...

        return expr

    def _cost(self, expr, **settings):
        if expr is None:
            return 0

#        print('>>> ', expr, type(expr))

        classes = type(expr).__mro__
        for cls in classes:
            method = '_cost_' + cls.__name__
            if hasattr(self, method):
                obj = getattr(self, method)(expr, **settings)
                return obj
        raise NotImplementedError('{} not available for {}'.format(method, type(expr)))

    def _cost_CodeBlock(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.body)

    def _cost_Comment(self, expr, **settings):
        return 0

    def _cost_EmptyNode(self, expr, **settings):
        return 0

    def _cost_Tuple(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr)

    def _cost_list(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr)

    def _cost_tuple(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr)

    def _cost_LiteralInteger(self, expr, **settings):
        return 0

    def _cost_LiteralFloat(self, expr, **settings):
        return 0

    def _cost_For(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.body.body)

        # ...
        i = expr.target
        i = pyccel_to_sympy(i, self._symbol_map, self._used_names)

        b = expr.iterable.start
        b = pyccel_to_sympy(b, self._symbol_map, self._used_names)

        e = expr.iterable.stop
        e = pyccel_to_sympy(e, self._symbol_map, self._used_names)

        step = expr.iterable.step
        step = pyccel_to_sympy(step, self._symbol_map, self._used_names)
        # ...

        end = ((e-b) / step)-1
        # translate so that we always start from 0
        return summation(ops, (i, 0, end))

    def _cost_PyccelOperator(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_FunctionDef(self, expr, **settings):
        return self._cost(expr.body, **settings)

    def _cost_FunctionCall(self, expr, **settings):
        if self.costs is None:
            raise ValueError('costs dict is None')

        fname = expr.func_name

        if not fname in self.costs.keys():
            raise ValueError('Cannot find the cost of the function {}'.format(fname))

        return self.costs[fname]

    def _cost_PyccelAssociativeParenthesis(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr._args)

    def _cost_Return(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in [expr.stmt, expr.expr])

    def _cost_PyccelArraySize(self, expr, **settings):
        # x = size(z) has a READ right?
        return 0

    def _compute_size_lhs(self, expr):
        ntimes = 1

        if isinstance(expr.lhs, IndexedElement):
            shape = None
            if expr.lhs.base in self.shapes.keys():
                shape = self.shapes[expr.lhs.base]

            indices = [(e,i) for e,i in enumerate(expr.lhs.indices) if isinstance(i, Slice)]
            for e,i in indices:
                # ...
                start = 0
                if not i.start == None:
                    start = i.start.python_value
                # ...

                # ...
                stop = SHAPE(pyccel_to_sympy(expr.lhs.base, self._symbol_map, self._used_names), e)
                if not i.stop == None:
                    stop = i.stop.python_value

                elif not(shape is None):
                        stop = shape[e]
                # ...

                # ...
                step = 1
                if not i.step == None:
                    step = i.step.python_value
                # ...

                if step != 1:
                    raise NotImplementedError('only step == 1 is treated')

                # TODO uncomment this
                #      this was commented because we get floor(...)
                ntimes *= (stop - start) #// step

        return ntimes

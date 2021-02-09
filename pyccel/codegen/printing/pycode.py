# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201
# pylint: disable=missing-function-docstring

from itertools import chain

from sympy.printing.pycode import PythonCodePrinter as SympyPythonCodePrinter
from sympy.printing.pycode import _known_functions
from sympy.printing.pycode import _known_functions_math
from sympy.printing.pycode import _known_constants_math

from pyccel.decorators import __all__ as pyccel_decorators

from pyccel.ast.utilities  import build_types_decorator
from pyccel.ast.core       import CodeBlock, Import, DottedName

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *

errors = Errors()

#==============================================================================
def _construct_header(func_name, args):
    args = build_types_decorator(args, order='F')
    args = ','.join("{}".format(i) for i in args)
    pattern = '#$ header function static {name}({args})'
    return pattern.format(name=func_name, args=args)

#==============================================================================
class PythonCodePrinter(SympyPythonCodePrinter):
    _kf = dict(chain(
        _known_functions.items(),
        [(k, '' + v) for k, v in _known_functions_math.items()]
    ))
    _kc = {k: ''+v for k, v in _known_constants_math.items()}

    def __init__(self, parser=None, settings=None):
        self.assert_contiguous = settings.pop('assert_contiguous', False)
        self.parser = parser
        SympyPythonCodePrinter.__init__(self, settings=settings)
        self._additional_imports = set()

    def get_additional_imports(self):
        """return the additional imports collected in printing stage"""
        return self._additional_imports

    def _print_Variable(self, expr):
        return self._print(expr.name)

    def _print_VariableAddress(self, expr):
        return self._print(expr.variable)

    def _print_Idx(self, expr):
        return self._print(expr.name)

    def _print_IndexedElement(self, expr):
        indices = expr.indices
        if isinstance(indices, (tuple, list)):
            # this a fix since when having a[i,j] the generated code is a[(i,j)]
            if len(indices) == 1 and isinstance(indices[0], (tuple, list)):
                indices = indices[0]

            indices = [self._print(i) for i in indices]
            indices = ','.join(i for i in indices)
        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        base = self._print(expr.base)
        return '{base}[{indices}]'.format(base=base, indices=indices)

    def _print_FunctionDef(self, expr):
        name = self._print(expr.name)
        body = self._print(expr.body)
        body = self._indent_codestring(body)
        args = ', '.join(self._print(i) for i in expr.arguments)

        imports = '\n'.join(self._print(i) for i in expr.imports)
        imports = self._indent_codestring(imports)
        code = ('def {name}({args}):\n'
                '\n{imports}\n{body}\n').format(name=name, args=args,imports=imports, body=body)
        decorators = expr.decorators
        if decorators:
            if decorators['template']:
                # Eliminate template_dict because it is useless in the printing
                expr.decorators['template'] = expr.decorators['template']['decorator_list']
            else:
                expr.decorators.pop('template')
            for n,f in decorators.items():
                if n in pyccel_decorators:
                    self._additional_imports.add(Import(DottedName('pyccel.decorators'), n))
                # TODO - All decorators must be stored in a list
                if not isinstance(f, list):
                    f = [f]
                dec = ''
                for func in f:
                    args = func.args
                    if args:
                        args = ', '.join("{}".format(self._print(i)) for i in args)
                        dec += '@{name}({args})\n'.format(name=n, args=args)

                    else:
                        dec += '@{name}\n'.format(name=n)

                code = '{dec}{code}'.format(dec=dec, code=code)
        headers = expr.headers
        if headers:
            headers = self._print(headers)
            code = '{header}\n{code}'.format(header=header, code=code)

        return code

    def _print_Return(self, expr):
        code = ''
        if expr.stmt:
            code += self._print(expr.stmt)+'\n'
        if expr.expr:
            ret = ','.join([self._print(i) for i in expr.expr])
            code += 'return {}'.format(ret)
        return code

    def _print_Program(self, expr):
        body  = self._print(expr.body)
        body = self._indent_codestring(body)
        imports  = [*expr.imports, *self._additional_imports]
        imports  = '\n'.join(self._print(i) for i in imports)

        return ('{imports}\n'
                'if __name__ == "__main__":\n'
                '{body}\n').format(imports=imports,
                                    body=body)


    def _print_AsName(self, expr):
        name = self._print(expr.name)
        target = self._print(expr.target)

        return '{name} as {target}'.format(name = name, target = target)

    def _print_PythonTuple(self, expr):
        args = ', '.join(self._print(i) for i in expr.args)
        return '('+args+')'

    def _print_PyccelArraySize(self, expr):
        arg = self._print(expr.arg)
        index = self._print(expr.index)
        return 'shape({0})[{1}]'.format(arg, index)

    def _print_Comment(self, expr):
        txt = self._print(expr.text)
        return '# {0} '.format(txt)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_DottedName(self, expr):
        return '.'.join(self._print(n) for n in expr.name)

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
        args = ', '.join(self._print(i) for i in expr.args)
        return'{func}({args})'.format(func=func.name, args=args)

    def _print_Len(self, expr):
        return 'len({})'.format(self._print(expr.arg))

    def _print_Import(self, expr):
        source = self._print(expr.source)
        if not expr.target:
            return 'import {source}'.format(source=source)
        else:
            target = ', '.join([self._print(i) for i in expr.target])
            return 'from {source} import {target}'.format(source=source, target=target)

    def _print_CodeBlock(self, expr):
        code = '\n'.join(self._print(c) for c in expr.body)
        return code

    def _print_For(self, expr):
        iterable = self._print(expr.iterable)
        target   = expr.target
        if not isinstance(target,(list, tuple)):
            target = [target]
        target = ','.join(self._print(i) for i in target)
        body   = self._print(expr.body)
        body   = self._indent_codestring(body)
        code   = ('for {0} in {1}:\n'
                '{2}\n').format(target,iterable,body)

        return code

    def _print_Assign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} = {1}'.format(lhs,rhs)

    def _print_AugAssign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        op  = self._print(expr.op._symbol)
        return'{0} {1}= {2}'.format(lhs,op,rhs)

    def _print_Range(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop)
        step  = self._print(expr.step)
        return 'range({}, {}, {})'.format(start,stop,step)

    def _print_Product(self, expr):
        args = ','.join(self._print(i) for i in expr.elements)
        return 'product({})'.format(args)

    def _print_Zeros(self, expr):
        return 'zeros('+ self._print(expr.shape)+')'

    def _print_ZerosLike(self, expr):
        return 'zeros_like('+ self._print(expr.rhs)+')'

    def _print_Max(self, expr):
        args = ', '.join(self._print(e) for e in expr.args)
        return 'max({})'.format(args)

    def _print_Min(self, expr):
        args = ', '.join(self._print(e) for e in expr.args)
        return 'min({})'.format(args)

    def _print_Slice(self, expr):
        return str(expr)

    def _print_Nil(self, expr):
        return 'None'

    def _print_Pass(self, expr):
        return 'pass'

    def _print_PyccelIs(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} is {1}'.format(lhs,rhs)

    def _print_If(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.blocks):
            if i == 0:
                lines.append("if (%s):" % self._print(c))

            elif i == len(expr.args) - 1 and c is True:
                lines.append("else:")

            else:
                lines.append("elif (%s):" % self._print(c))

            if isinstance(e, CodeBlock):
                body = self._indent_codestring(self._print(e))
                lines.append(body)
            else:
                lines.append(self._print(e))
        return "\n".join(lines)

    def _print_LiteralString(self, expr):
        return '"{}"'.format(self._print(expr.arg))

    def _print_Shape(self, expr):
        arg = self._print(expr.arg)
        if expr.index is None:
            return '{}.shape'.format(arg)

        else:
            index = self._print(expr.index)
            return '{0}.shape[{1}]'.format(arg, index)

    def _print_Print(self, expr):
        args = []
        for f in expr.expr:
            if isinstance(f, str):
                args.append("'{}'".format(f))

            elif isinstance(f, tuple):
                for i in f:
                    args.append("{}".format(self._print(i)))

            else:
                args.append("{}".format(self._print(f)))

        fs = ', '.join(i for i in args)

        return 'print({0})'.format(fs)

    def _print_Module(self, expr):
        body = '\n'.join(self._print(e) for e in expr.body)
        imports  = [*expr.imports, *self._additional_imports]
        imports  = '\n'.join(self._print(i) for i in imports)
        return ('{imports}\n\n'
                '{body}').format(
                        imports = imports,
                        body    = body)

    def _print_PyccelPow(self, expr):
        base = self._print(expr.args[0])
        e    = self._print(expr.args[1])
        return '{} ** {}'.format(base, e)

    def _print_PyccelAdd(self, expr):
        return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        return ' - '.join(self._print(a) for a in expr.args)

    def _print_PyccelMul(self, expr):
        return ' * '.join(self._print(a) for a in expr.args)

    def _print_PyccelDiv(self, expr):
        return ' / '.join(self._print(a) for a in expr.args)

    def _print_PyccelMod(self, expr):
        return '%'.join(self._print(a) for a in expr.args)

    def _print_PyccelFloorDiv(self, expr):
        return '//'.join(self._print(a) for a in expr.args)

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelAnd(self, expr):
        return ' and '.join(self._print(a) for a in expr.args)

    def _print_PyccelOr(self, expr):
        return ' or '.join(self._print(a) for a in expr.args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} == {1} '.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} != {1} '.format(lhs, rhs)

    def _print_PyccelLt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} < {1}'.format(lhs, rhs)

    def _print_PyccelLe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} <= {1}'.format(lhs, rhs)

    def _print_PyccelGt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} > {1}'.format(lhs, rhs)

    def _print_PyccelGe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} >= {1}'.format(lhs, rhs)

    def _print_PyccelNot(self, expr):
        a = self._print(expr.args[0])
        return 'not {}'.format(a)

#==============================================================================
def pycode(expr, **settings):
    """ Converts an expr to a string of Python code
    Parameters
    ==========
    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    Examples
    ========
    >>> from sympy import tan, Symbol
    >>> from sympy.printing.pycode import pycode
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'
    """
    settings.pop('parser', None)
    return PythonCodePrinter(settings).doprint(expr)

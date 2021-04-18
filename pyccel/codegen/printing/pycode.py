# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201
# pylint: disable=missing-function-docstring

from pyccel.decorators import __all__ as pyccel_decorators

from pyccel.ast.utilities  import build_types_decorator
from pyccel.ast.core       import CodeBlock, Import, Assign, FunctionCall, For
from pyccel.ast.datatypes  import default_precision
from pyccel.ast.literals   import LiteralTrue, LiteralString
from pyccel.ast.variable   import DottedName

from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors import Errors
from pyccel.errors.messages import PYCCEL_RESTRICTION_TODO

errors = Errors()

#==============================================================================
def _construct_header(func_name, args):
    args = build_types_decorator(args, order='F')
    args = ','.join("{}".format(i) for i in args)
    pattern = '#$ header function static {name}({args})'
    return pattern.format(name=func_name, args=args)

#==============================================================================

# Dictionary mapping imported targets to their aliases used internally by pyccel
# This prevents a mismatch between printed imports and function calls
# The keys are modules from which the target is imported
# The values are a dictionary whose keys are object aliases and whose values
# are the names used in pyccel
import_target_swap = {
        'numpy' : {'double'     : 'float64',
                   'product'    : 'prod',
                   'empty_like' : 'empty',
                   'zeros_like' : 'zeros',
                   'ones_like'  : 'ones',
                   'full_like'  : 'full'},
        'numpy.random' : {'random' : 'rand'}
        }

class PythonCodePrinter(CodePrinter):
    """A printer to convert pyccel expressions to strings of Python code"""
    printmethod = "_pycode"
    language = "python"

    _default_settings = {
        'tabwidth': 4,
    }

    def __init__(self, parser=None):
        self._parser = parser
        super().__init__()
        self._additional_imports = {}

    def _indent_codestring(self, lines):
        tab = " "*self._default_settings['tabwidth']
        if lines == '':
            return lines
        else:
            # lines ends with \n
            return tab+lines[:-1].replace('\n','\n'+tab)+'\n'

    def _format_code(self, lines):
        return lines

    def get_additional_imports(self):
        """return the additional imports collected in printing stage"""
        imports = [i for tup in self._additional_imports.values() for i in tup[1]]
        return imports

    def insert_new_import(self, import_obj):
        """ Add an import of an object which may have been
        added by pyccel and therefore may not have been imported
        """
        source = str(import_obj.source)
        src_info = self._additional_imports.setdefault(source, (set(), []))
        src_info[0].update(import_obj.target)
        src_info[1].append(import_obj)

    def _print_tuple(self, expr):
        fs = ', '.join(self._print(f) for f in expr)
        return '({0})'.format(fs)

    def _print_NativeBool(self, expr):
        return 'bool'

    def _print_NativeInteger(self, expr):
        return 'int'

    def _print_NativeReal(self, expr):
        return 'float'

    def _print_NativeComplex(self, expr):
        return 'complex'

    def _print_Variable(self, expr):
        return self._print(expr.name)

    def _print_ValuedArgument(self, expr):
        return '{} = {}'.format(self._print(expr.argument), self._print(expr.value))

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

        doc_string = self._print(expr.doc_string) if expr.doc_string else ''
        doc_string = self._indent_codestring(doc_string)

        body = ''.join([doc_string, imports, body])

        code = ('def {name}({args}):\n'
                '{body}\n').format(
                        name=name,
                        args=args,
                        body=body)
        decorators = expr.decorators
        if decorators:
            if decorators['template']:
                # Eliminate template_dict because it is useless in the printing
                expr.decorators['template'] = expr.decorators['template']['decorator_list']
            else:
                expr.decorators.pop('template')
            for n,f in decorators.items():
                if n in pyccel_decorators:
                    self.insert_new_import(Import(DottedName('pyccel.decorators'), n))
                # TODO - All decorators must be stored in a list
                if not isinstance(f, list):
                    f = [f]
                dec = ''
                for func in f:
                    if isinstance(func, FunctionCall):
                        args = func.args
                    elif func == n:
                        args = []
                    else:
                        args = [LiteralString(a) for a in func]
                    if n == 'types' and len(args)==0:
                        continue
                    if args:
                        args = ', '.join(self._print(i) for i in args)
                        dec += '@{name}({args})\n'.format(name=n, args=args)

                    else:
                        dec += '@{name}\n'.format(name=n)

                code = '{dec}{code}'.format(dec=dec, code=code)
        headers = expr.headers
        if headers:
            headers = self._print(headers)
            code = '{header}\n{code}'.format(header=headers, code=code)

        return code

    def _print_FunctionAddress(self, expr):
        return expr.name

    def _print_Return(self, expr):

        rhs_list = [i.rhs for i in expr.stmt.body if isinstance(i, Assign)] if expr.stmt else []
        lhs_list = [i.lhs for i in expr.stmt.body if isinstance(i, Assign)] if expr.stmt else []
        expr_return_vars = [a for a in expr.expr if a not in lhs_list]

        return 'return {}\n'.format(','.join(self._print(i) for i in expr_return_vars + rhs_list))

    def _print_Program(self, expr):
        body  = self._print(expr.body)
        body = self._indent_codestring(body)
        imports  = [*expr.imports, *self.get_additional_imports()]
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
        if len(expr.args) == 1:
            args += ','
        return '('+args+')'

    def _print_PythonList(self, expr):
        args = ', '.join(self._print(i) for i in expr.args)
        return '['+args+']'

    def _print_PythonLen(self, expr):
        return 'len({})'.format(self._print(expr.arg))

    def _print_PythonAbs(self, expr):
        return 'abs({})'.format(self._print(expr.arg))

    def _print_PythonMin(self, expr):
        return 'min({})'.format(self._print(expr.args[0]))

    def _print_PythonMax(self, expr):
        return 'max({})'.format(self._print(expr.args[0]))

    def _print_PythonSum(self, expr):
        return 'sum({})'.format(self._print(expr.args[0]))

    def _print_PythonBool(self, expr):
        return 'bool({})'.format(self._print(expr.arg))

    def _print_PythonInt(self, expr):
        type_name = type(expr).__name__.lower()
        is_numpy  = type_name.startswith('numpy')
        precision = str(expr.precision*8) if is_numpy else ''
        name = 'int{}'.format(precision)
        if is_numpy:
            self.insert_new_import(Import('numpy',name))
        return '{}({})'.format(name, self._print(expr.arg))

    def _print_PythonFloat(self, expr):
        type_name = type(expr).__name__.lower()
        is_numpy  = type_name.startswith('numpy')
        precision = str(expr.precision*8) if is_numpy else ''
        name = 'float{}'.format(precision)
        if is_numpy:
            self.insert_new_import(Import('numpy',name))
        return '{}({})'.format(name, self._print(expr.arg))

    def _print_PythonComplex(self, expr):
        if expr.is_cast:
            return 'complex({})'.format(self._print(expr.internal_var))
        else:
            return 'complex({}, {})'.format(self._print(expr.real), self._print(expr.imag))

    def _print_NumpyComplex(self, expr):
        precision = str(expr.precision*16)
        name = 'complex{}'.format(precision)
        self.insert_new_import(Import('numpy',name))
        if expr.is_cast:
            return '{}({})'.format(name, self._print(expr.internal_var))
        else:
            return '{}({}+{}*1j)'.format(name, self._print(expr.real), self._print(expr.imag))

    def _print_PythonRange(self, expr):
        return 'range({start}, {stop}, {step})'.format(
                start = self._print(expr.start),
                stop  = self._print(expr.stop ),
                step  = self._print(expr.step ))

    def _print_PythonReal(self, expr):
        return '({}).real'.format(self._print(expr.internal_var))

    def _print_PythonImag(self, expr):
        return '({}).imag'.format(self._print(expr.internal_var))

    def _print_PythonPrint(self, expr):
        return 'print({})'.format(', '.join(self._print(a) for a in expr.expr))

    def _print_PyccelArraySize(self, expr):
        arg = self._print(expr.arg)
        index = self._print(expr.index)
        return 'shape({0})[{1}]'.format(arg, index)

    def _print_Comment(self, expr):
        txt = self._print(expr.text)
        return '# {0} \n'.format(txt)

    def _print_CommentBlock(self, expr):
        txt = '\n'.join(self._print(c) for c in expr.comments)
        return '"""{0}"""\n'.format(txt)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_DottedName(self, expr):
        return '.'.join(self._print(n) for n in expr.name)

    def _print_FunctionCall(self, expr):
        if expr.interface:
            func_name = expr.interface_name
        else:
            func_name = expr.funcdef.name
        args = ', '.join(self._print(i) for i in expr.args)
        code = '{func}({args})'.format(func=func_name, args=args)
        if expr.funcdef.results:
            return code
        else:
            return code+'\n'

    def _print_Len(self, expr):
        return 'len({})'.format(self._print(expr.arg))

    def _print_Import(self, expr):
        source = self._print(expr.source)
        if not expr.target:
            return 'import {source}\n'.format(source=source)
        else:
            if source in import_target_swap:
                # If the source contains multiple names which reference the same object
                # check if the target is referred to by another name in pyccel.
                # Print the name used by pyccel (either the value from import_target_swap
                # or the original name from the import
                target = [self._print(import_target_swap[source].get(i,i)) for i in expr.target]
            else:
                target = [self._print(i) for i in expr.target]
            target = ', '.join(target)
            return 'from {source} import {target}\n'.format(source=source, target=target)

    def _print_CodeBlock(self, expr):
        if len(expr.body)==0:
            return 'pass\n'
        else:
            code = ''.join(self._print(c) for c in expr.body)
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
                '{2}').format(target,iterable,body)

        return code

    def _print_FunctionalFor(self, expr):
        dummy_var = expr.index
        iterators = []
        body = expr.loops[1]
        while not isinstance(body, Assign):
            if isinstance(body, CodeBlock):
                body = body.body
                if len(body) > 1:
                    if any(not(isinstance(b, Assign) and b.lhs is dummy_var) for b in body[1:]):
                        raise NotImplementedError("Pyccel has introduced unnecessary statements which it cannot yet disambiguate in the python printer")
                body = body[0]
            elif isinstance(body, For):
                iterators.append(body.iterable)
                body = body.body
            else:
                raise NotImplementedError("Type {} not handled in a FunctionalFor".format(type(body)))
        body = self._print(body.rhs)
        for_loops = ' '.join(['for {} in {}'.format(self._print(idx), self._print(iters))
                        for idx, iters in zip(expr.indices, iterators)])
        lhs = self._print(expr.lhs)
        return '{} = [{} {}]\n'.format(lhs, body, for_loops)

    def _print_While(self, expr):
        cond = self._print(expr.test)
        body = self._indent_codestring(self._print(expr.body))
        return 'while {cond}:\n{body}'.format(
                cond = cond,
                body = body)

    def _print_Break(self, expr):
        return 'break\n'

    def _print_Continue(self, expr):
        return 'continue\n'

    def _print_Assign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} = {1}\n'.format(lhs,rhs)

    def _print_AliasAssign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} = {1}\n'.format(lhs,rhs)

    def _print_AugAssign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        op  = self._print(expr.op)
        return'{0} {1}= {2}\n'.format(lhs,op,rhs)

    def _print_PythonRange(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop)
        step  = self._print(expr.step)
        return 'range({}, {}, {})'.format(start,stop,step)

    def _print_Product(self, expr):
        args = ','.join(self._print(i) for i in expr.elements)
        return 'product({})'.format(args)

    def _print_Allocate(self, expr):
        return ''

    def _print_Deallocate(self, expr):
        return ''

    def _print_NumpyArray(self, expr):
        dtype = self._print(expr.dtype)
        if expr.precision != default_precision[str(expr.dtype)]:
            factor = 16 if dtype == 'complex' else 8
            dtype += str(expr.precision*factor)

        return "array({arg}, dtype={dtype}, order='{order}')".format(
                arg   = self._print(expr.arg),
                dtype = dtype,
                order = expr.order)

    def _print_NumpyAutoFill(self, expr):
        type_name = type(expr).__name__
        func_name = type_name[5:].lower()

        dtype = self._print(expr.dtype)
        if expr.precision != default_precision[str(expr.dtype)]:
            factor = 16 if dtype == 'complex' else 8
            dtype += str(expr.precision*factor)

        return "{func_name}({shape}, dtype={dtype}, order='{order}')".format(
                func_name = func_name,
                shape = self._print(expr.shape),
                dtype = dtype,
                order = expr.order)

    def _print_NumpyLinspace(self, expr):
        return "linspace({0}, {1}, {2})".format(
                self._print(expr.start),
                self._print(expr.stop),
                self._print(expr.size))

    def _print_NumpyMatmul(self, expr):
        return "matmul({0}, {1})".format(
                self._print(expr.a),
                self._print(expr.b))


    def _print_NumpyFull(self, expr):
        dtype = self._print(expr.dtype)
        if expr.precision != default_precision[str(expr.dtype)]:
            factor = 16 if dtype == 'complex' else 8
            dtype += str(expr.precision*factor)

        return "full({shape}, {fill_value}, dtype={dtype}, order='{order}')".format(
                shape = self._print(expr.shape),
                fill_value = self._print(expr.fill_value),
                dtype = dtype,
                order = expr.order)

    def _print_NumpyArange(self, expr):
        return "arange({start}, {stop}, {step}, dtype={dtype})".format(
                start = self._print(expr.start),
                stop  = self._print(expr.stop),
                step  = self._print(expr.step),
                dtype = self._print(expr.dtype))

    def _print_NumpySum(self, expr):
        return "sum({})".format(self._print(expr.arg))

    def _print_NumpyProduct(self, expr):
        return "prod({})".format(self._print(expr.arg))

    def _print_NumpyRand(self, expr):
        args = ', '.join(self._print(a) for a in expr.args)
        return "rand({})".format(args)

    def _print_NumpyRandint(self, expr):
        if expr.low:
            args = "{}, ".format(self._print(expr.low))
        else:
            args = ""
        args += "{}".format(self._print(expr.high))
        if expr.shape != ():
            size = self._print(expr.shape)
            args += ", size = {}".format(size)
        return "randint({})".format(args)

    def _print_NumpyNorm(self, expr):
        name = 'norm'
        axis = self._print(expr.axis) if expr.axis else None
        if axis:
            return  "{name}({arg},axis={axis})".format(name = name, arg  = self._print(expr.python_arg), axis=axis)
        return  "{name}({arg})".format(name = name, arg  = self._print(expr.python_arg))

    def _print_NumpyUfuncBase(self, expr):
        type_name = type(expr).__name__
        name = type_name[5:].lower()
        args = ', '.join(self._print(a) for a in expr.args)
        return "{}({})".format(name, args)

    def _print_MathFunctionBase(self, expr):
        type_name = type(expr).__name__
        name = type_name[4:].lower()
        args = ', '.join(self._print(a) for a in expr.args)
        return "{}({})".format(name, args)

    def _print_Max(self, expr):
        args = ', '.join(self._print(e) for e in expr.args)
        return 'max({})'.format(args)

    def _print_Min(self, expr):
        args = ', '.join(self._print(e) for e in expr.args)
        return 'min({})'.format(args)

    def _print_Slice(self, expr):
        start = self._print(expr.start) if expr.start else ''
        stop  = self._print(expr.stop)  if expr.stop  else ''
        step  = self._print(expr.step)  if expr.step  else ''
        return '{start}:{stop}:{step}'.format(
                start = start,
                stop  = stop,
                step  = step)

    def _print_Nil(self, expr):
        return 'None'

    def _print_Pass(self, expr):
        return 'pass\n'

    def _print_PyccelIs(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} is {1}'.format(lhs,rhs)

    def _print_PyccelIsNot(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} is not {1}'.format(lhs,rhs)

    def _print_If(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.blocks):
            if i == 0:
                lines.append("if %s:\n" % self._print(c))

            elif i == len(expr.blocks) - 1 and isinstance(c, LiteralTrue):
                lines.append("else:\n")

            else:
                lines.append("elif %s:\n" % self._print(c))

            if isinstance(e, CodeBlock):
                body = self._indent_codestring(self._print(e))
                lines.append(body)
            else:
                lines.append(self._print(e))
        return "".join(lines)

    def _print_IfTernaryOperator(self, expr):
        cond = self._print(expr.cond)
        value_true = self._print(expr.value_true)
        value_false = self._print(expr.value_false)
        return '{true} if {cond} else {false}'.format(cond = cond, true =value_true, false = value_false)

    def _print_Literal(self, expr):
        return repr(expr.python_value)

    def _print_Print(self, expr):
        args = []
        for f in expr.expr:
            if isinstance(f, str):
                args.append("'{}'".format(f))

            elif isinstance(f, tuple):
                for i in f:
                    args.append(self._print(i))

            else:
                args.append(self._print(f))

        fs = ', '.join(i for i in args)

        return 'print({0})\n'.format(fs)

    def _print_Module(self, expr):
        # Print interface functions (one function with multiple decorators describes the problem)
        interfaces = [i.functions[0] for i in expr.interfaces]
        for f,i in zip(interfaces, expr.interfaces):
            f.rename(i.name)
        interfaces = ''.join(self._print(i) for i in interfaces)
        # Collect functions which are not in an interface
        funcs = [f for f in expr.funcs if not any(f in i.functions for i in expr.interfaces)]
        funcs = ''.join(self._print(f) for f in funcs)
        classes = ''.join(self._print(c) for c in expr.classes)
        body = ''.join((interfaces, funcs, classes))
        imports  = [*expr.imports, *self.get_additional_imports()]
        imports  = ''.join(self._print(i) for i in imports)
        return ('{imports}\n'
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
        return '+{}'.format(self._print(expr.args[0]))

    def _print_PyccelUnarySub(self, expr):
        return '-{}'.format(self._print(expr.args[0]))

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

    def _print_PyccelInvert(self, expr):
        return '~{}'.format(self._print(expr.args[0]))

    def _print_PyccelRShift(self, expr):
        return '{} >> {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelLShift(self, expr):
        return '{} << {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitXor(self, expr):
        return '{} ^ {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitOr(self, expr):
        return '{} | {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitAnd(self, expr):
        return '{} & {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelSymbol(self, expr):
        return expr

#==============================================================================
def pycode(expr, assign_to=None, **settings):
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
    return PythonCodePrinter(settings).doprint(expr, assign_to)

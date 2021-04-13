# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201
# pylint: disable=missing-function-docstring

"""Print to F90 standard. Trying to follow the information provided at
www.fortran90.org as much as possible."""


import string
from itertools import chain
from collections import OrderedDict

import functools
import operator

from sympy.core.numbers import NegativeInfinity as NINF
from sympy.core.numbers import Infinity as INF

from pyccel.ast.core import get_iterable_ranges
from pyccel.ast.core import SeparatorComment, Comment
from pyccel.ast.core import ConstructorCall
from pyccel.ast.core import ErrorExit, FunctionAddress
from pyccel.ast.internals    import PyccelInternalFunction
from pyccel.ast.itertoolsext import Product
from pyccel.ast.core import (Assign, AliasAssign, Declare,
                             CodeBlock, Dlist, AsName,
                             If, IfSection)

from pyccel.ast.variable  import (Variable, TupleVariable,
                             IndexedElement,
                             DottedName, PyccelArraySize)

from pyccel.ast.operators      import PyccelAdd, PyccelMul, PyccelDiv, PyccelMinus, PyccelNot
from pyccel.ast.operators      import PyccelMod

from pyccel.ast.operators      import PyccelUnarySub, PyccelLt, PyccelGt, IfTernaryOperator

from pyccel.ast.core      import FunctionCall, DottedFunctionCall

from pyccel.ast.builtins  import (PythonEnumerate, PythonInt, PythonLen,
                                  PythonMap, PythonPrint, PythonRange,
                                  PythonZip, PythonFloat, PythonTuple)
from pyccel.ast.builtins  import PythonComplex, PythonBool, PythonAbs
from pyccel.ast.datatypes import is_pyccel_datatype
from pyccel.ast.datatypes import is_iterable_datatype, is_with_construct_datatype
from pyccel.ast.datatypes import NativeSymbol, NativeString, str_dtype
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeReal, NativeComplex
from pyccel.ast.datatypes import iso_c_binding
from pyccel.ast.datatypes import NativeRange, NativeTuple
from pyccel.ast.datatypes import CustomDataType

from pyccel.ast.internals import Slice

from pyccel.ast.literals  import LiteralInteger, LiteralFloat, Literal
from pyccel.ast.literals  import LiteralTrue
from pyccel.ast.literals  import Nil

from pyccel.ast.numpyext import NumpyEmpty
from pyccel.ast.numpyext import NumpyMod, NumpyFloat
from pyccel.ast.numpyext import NumpyRand
from pyccel.ast.numpyext import NumpyNewArray
from pyccel.ast.numpyext import Shape

from pyccel.ast.utilities import builtin_import_registery as pyccel_builtin_import_registery
from pyccel.ast.utilities import expand_to_loops

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *
from pyccel.codegen.printing.codeprinter import CodePrinter


# TODO: add examples
# TODO: use _get_statement when returning a string

__all__ = ["FCodePrinter", "fcode"]

numpy_ufunc_to_fortran = {
    'NumpyAbs'  : 'abs',
    'NumpyFabs'  : 'abs',
    'NumpyMin'  : 'minval',
    'NumpyMax'  : 'maxval',
    'NumpyFloor': 'floor',  # TODO: might require special treatment with casting
    # ---
    'NumpyExp' : 'exp',
    'NumpyLog' : 'Log',
    # 'NumpySqrt': 'Sqrt',  # sqrt is printed using _Print_NumpySqrt
    # ---
    'NumpySin'    : 'sin',
    'NumpyCos'    : 'cos',
    'NumpyTan'    : 'tan',
    'NumpyArcsin' : 'asin',
    'NumpyArccos' : 'acos',
    'NumpyArctan' : 'atan',
    'NumpyArctan2': 'atan2',
    'NumpySinh'   : 'sinh',
    'NumpyCosh'   : 'cosh',
    'NumpyTanh'   : 'tanh',
    'NumpyArcsinh': 'asinh',
    'NumpyArccosh': 'acosh',
    'NumpyArctanh': 'atanh',
}

math_function_to_fortran = {
    'MathAcos'   : 'acos',
    'MathAcosh'  : 'acosh',
    'MathAsin'   : 'asin',
    'MathAsinh'  : 'asinh',
    'MathAtan'   : 'atan',
    'MathAtan2'  : 'atan2',
    'MathAtanh'  : 'atanh',
    'MathCopysign': 'sign',
    'MathCos'    : 'cos',
    'MathCosh'   : 'cosh',
    'MathErf'    : 'erf',
    'MathErfc'   : 'erfc',
    'MathExp'    : 'exp',
    # 'MathExpm1'  : '???', # TODO
    'MathFabs'   : 'abs',
    # 'MathFmod'   : '???',  # TODO
    # 'MathFsum'   : '???',  # TODO
    'MathGamma'  : 'gamma',
    'MathHypot'  : 'hypot',
    # 'MathLdexp'  : '???',  # TODO
    'MathLgamma' : 'log_gamma',
    'MathLog'    : 'log',
    'MathLog10'  : 'log10',
    # 'MathLog1p'  : '???', # TODO
    # 'MathLog2'   : '???', # TODO
    # 'MathPow'    : '???', # TODO
    'MathSin'    : 'sin',
    'MathSinh'   : 'sinh',
    'MathSqrt'   : 'sqrt',
    'MathTan'    : 'tan',
    'MathTanh'   : 'tanh',
    # ---
    'MathFloor'    : 'floor',
    # ---
    # 'MathIsclose' : '???', # TODO
    # 'MathIsfinite': '???', # TODO
    # 'MathIsinf'   : '???', # TODO
    # --------------------------- internal functions --------------------------
    'MathFactorial' : 'pyc_factorial',
    'MathGcd'       : 'pyc_gcd',
    'MathDegrees'   : 'pyc_degrees',
    'MathRadians'   : 'pyc_radians',
    'MathLcm'       : 'pyc_lcm',
}

_default_methods = {
    '__init__': 'create',
    '__del__' : 'free',
}

python_builtin_datatypes = {
    'integer' : PythonInt,
    'real'    : PythonFloat,
    'bool'    : PythonBool,
    'complex' : PythonComplex
}

errors = Errors()

class FCodePrinter(CodePrinter):
    """A printer to convert sympy expressions to strings of Fortran code"""
    printmethod = "_fcode"
    language = "Fortran"

    _default_settings = {
        'tabwidth': 2,
    }


    def __init__(self, parser, prefix_module = None):

        if parser.filename:
            errors.set_target(parser.filename, 'file')

        super().__init__()
        self.parser = parser
        self._namespace = self.parser.namespace
        self._current_function = None
        self._current_class    = None

        self._additional_code = None
        self._additional_imports = set([])

        self.prefix_module = prefix_module

    def get_additional_imports(self):
        """return the additional imports collected in printing stage"""
        return self._additional_imports

    def set_current_class(self, name):

        self._current_class = name

    def set_current_function(self, name):

        if name:
            self._namespace = self._namespace.sons_scopes[name]
            if self._current_function:
                name = DottedName(self._current_function, name)
        else:
            self._namespace = self._namespace.parent_scope
            if isinstance(self._current_function, DottedName):

                # case of a function inside a function

                name = self._current_function.name[:-1]
                if len(name) > 1:
                    name = DottedName(*name)
                else:
                    name = name[0]
        self._current_function = name

    def get_method(self, cls_name, method_name):
        container = self._namespace
        while container:
            if cls_name in container.classes:
                cls = container.classes[cls_name]
                methods = cls.methods_as_dict
                if method_name in methods:
                    return methods[method_name]
                else:
                    interface_funcs = {f.name:f for i in cls.interfaces for f in i.functions}
                    if method_name in interface_funcs:
                        return interface_funcs[method_name]
                    errors.report(UNDEFINED_METHOD, symbol=method_name,
                        severity='fatal')
            container = container.parent_scope
        if isinstance(method_name, DottedName):
            return self.get_function(DottedName(method_name.name[1:]))
        errors.report(UNDEFINED_FUNCTION, symbol=method_name,
            severity='fatal')

    def get_function(self, name):
        container = self._namespace
        while container:
            if name in container.functions:
                return container.functions[name]
            container = container.parent_scope
        if isinstance(name, DottedName):
            return self.get_function(name.name[-1])
        errors.report(UNDEFINED_FUNCTION, symbol=name,
            severity='fatal')

    def add_vars_to_namespace(self, *new_vars):
        if self._current_function:
            if self._current_class:
                func = self.get_method(self._current_class, self._current_function)
            else:
                func = self.get_function(self._current_function)
            func.add_local_vars(*new_vars)
        else:
            for var in new_vars:
                self._namespace.variables[var.name] = var

    def _get_statement(self, codestring):
        return codestring

    def _get_comment(self, text):
        return "! {0}".format(text)

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    def _handle_fortran_specific_a_prioris(self, var_list):
        for v in var_list:
            if isinstance(v, TupleVariable):
                if v.is_pointer or v.inconsistent_shape:
                    v.is_homogeneous = False

    def print_kind(self, expr):
        """
        Prints the kind(precision) of a literal value
        """
        return iso_c_binding[self._print(expr.dtype)][expr.precision]

    # ============ Elements ============ #
    def _print_PyccelSymbol(self, expr):
        return expr

    def _print_Module(self, expr):
        self._handle_fortran_specific_a_prioris(self.parser.get_variables(self._namespace))
        name = self._print(expr.name)
        name = name.replace('.', '_')
        if not name.startswith('mod_') and self.prefix_module:
            name = '{prefix}_{name}'.format(prefix=self.prefix_module,
                                            name=name)

        imports = ''.join(self._print(i) for i in expr.imports)
        imports += 'use, intrinsic :: ISO_C_BINDING\n'

        decs    = ''.join(self._print(i) for i in expr.declarations)
        body    = ''

        # ... TODO add other elements
        private_funcs = [f.name for f in expr.funcs if f.is_private]
        private = private_funcs
        if private:
            private = ','.join(self._print(i) for i in private)
            private = 'private :: {}\n'.format(private)
        else:
            private = ''
        # ...

        # ...
        sep = self._print(SeparatorComment(40))
        interfaces = ''
        if expr.interfaces:
            interfaces = '\n'.join(self._print(i) for i in expr.interfaces)

        if expr.funcs:
            body += '\n'.join(''.join([sep, self._print(i), sep]) for i in expr.funcs)
        # ...

        # ...
        for i in expr.classes:
            # update decs with declarations from ClassDef
            c_decs, c_funcs = self._print(i)
            decs = '{0}\n{1}'.format(decs, c_decs)
            body = '{0}\n{1}\n'.format(body, c_funcs)
        # ...

        contains = 'contains\n' if (expr.funcs or expr.classes or expr.interfaces) else ''
        imports += "\n".join('use ' + lib for lib in self._additional_imports)
        parts = ['module {}\n'.format(name),
                 imports,
                 'implicit none\n',
                 private,
                 decs,
                 interfaces,
                 contains,
                 body,
                 'end module {}\n'.format(name)]

        return '\n'.join([a for a in parts if a])

    def _print_Program(self, expr):
        self._handle_fortran_specific_a_prioris(self.parser.get_variables(self._namespace))
        name    = 'prog_{0}'.format(self._print(expr.name)).replace('.', '_')
        imports = ''.join(self._print(i) for i in expr.imports)
        imports += 'use, intrinsic :: ISO_C_BINDING\n'
        body    = self._print(expr.body)

        # Print the declarations of all variables in the namespace, which include:
        #  - user-defined variables (available in Program.variables)
        #  - pyccel-generated variables added to Scope when printing 'expr.body'
        variables = self.parser.get_variables(self._namespace)
        decs = ''.join(self._print_Declare(Declare(v.dtype, v)) for v in variables)

        # Detect if we are using mpi4py
        # TODO should we find a better way to do this?
        mpi = any('mpi4py' == str(getattr(i.source, 'name', i.source)) for i in expr.imports)

        # Additional code and variable declarations for MPI usage
        # TODO: check if we should really add them like this
        if mpi:
            body = 'call mpi_init(ierr)\n'+\
                   '\nallocate(status(0:-1 + mpi_status_size)) '+\
                   '\nstatus = 0\n'+\
                   body +\
                   '\ncall mpi_finalize(ierr)'

            decs += '\ninteger :: ierr = -1' +\
                    '\ninteger, allocatable :: status (:)'
        imports += "\n".join('use ' + lib for lib in self._additional_imports)
        parts = ['program {}\n'.format(name),
                 imports,
                'implicit none\n',
                 decs,
                 body,
                'end program {}\n'.format(name)]

        return '\n'.join(a for a in parts if a)

    def _print_Import(self, expr):

        source = ''
        if expr.ignore:
            return ''

        if isinstance(expr.source, DottedName):
            source = expr.source.name[-1]
        else:
            source = self._print(expr.source)

        # importing of pyccel extensions is not printed
        if source in pyccel_builtin_import_registery:
            return ''

        if 'mpi4py' == str(getattr(expr.source,'name',expr.source)):
            return 'use mpi\n' + 'use mpiext\n'

        if len(expr.target) == 0:
            return 'use {}\n'.format(source)

        prefix = 'use {}, only:'.format(source)

        code = ''
        for i in expr.target:
            if isinstance(i, AsName):
                target = '{target} => {name}'.format(target=self._print(i.target),
                                                     name=self._print(i.name))
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=target)

            elif isinstance(i, DottedName):
                target = '_'.join(self._print(j) for j in i.name)
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=target)

            elif isinstance(i, str):
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=i)

            else:
                raise TypeError('Expecting str, PyccelSymbol, DottedName or AsName, '
                                'given {}'.format(type(i)))

            code = (code + '\n' + line) if code else line

        # in some cases, the source is given as a string (when using metavar)
        code = code.replace("'", '')
        return self._get_statement(code) + '\n'

    def _print_PythonPrint(self, expr):
        args = []
        for f in expr.expr:
            if isinstance(f, str):
                args.append("'{}'".format(f))
            elif isinstance(f, PythonTuple):
                for i in f:
                    args.append("{}".format(self._print(i)))
            elif isinstance(f, TupleVariable) and not f.is_homogeneous:
                for i in f:
                    args.append("{}".format(self._print(i)))
            elif f.dtype is NativeString() and f != expr.expr[-1]:
                args.append("{} // ' ' ".format(self._print(f)))
            else:
                args.append("{}".format(self._print(f)))

        code = ', '.join(['print *', *args])
        return self._get_statement(code) + '\n'

    def _print_SymbolicPrint(self, expr):
        # for every expression we will generate a print
        code = '\n'.join("print *, 'sympy> {}'".format(a) for a in expr.expr)
        return self._get_statement(code) + '\n'

    def _print_Comment(self, expr):
        comments = self._print(expr.text)
        return '!' + comments + '\n'

    def _print_CommentBlock(self, expr):
        txts   = expr.comments
        header = expr.header
        header_size = len(expr.header)

        ln = max(len(i) for i in txts)
        if ln<max(20, header_size+2):
            ln = 20
        top  = '!' + '_'*int((ln-header_size)/2) + header + '_'*int((ln-header_size)/2) + '!'
        ln = len(top) - 2
        bottom = '!' + '_'*ln + '!'

        txts = ['!' + txt + ' '*(ln - len(txt)) + '!' for txt in txts]

        body = '\n'.join(i for i in txts)

        return ('{0}\n'
                '{1}\n'
                '{2}\n').format(top, body, bottom)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_AnnotatedComment(self, expr):
        accel = self._print(expr.accel)
        txt   = str(expr.txt)
        if len(txt)>72:
            txts = []
            while len(txt)>72:
                txts.append(txt[:72])
                txt  = txt[72:]
            if txt:
                txts.append(txt)

            txt = '&\n!${} &'.format(accel).join(txt for txt in txts)

        return '!${0} {1}\n'.format(accel, txt)

    def _print_tuple(self, expr):
        if expr[0].rank>0:
            raise NotImplementedError(' tuple with elements of rank > 0 is not implemented')
        fs = ', '.join(self._print(f) for f in expr)
        return '[{0}]'.format(fs)

    def _print_PythonAbs(self, expr):
        """ print the python builtin function abs
        args : variable
        """
        return "abs({})".format(self._print(expr.arg))

    def _print_PythonTuple(self, expr):
        shape = tuple(reversed(expr.shape))
        if len(shape)>1:
            elements = ', '.join(self._print(i) for i in expr)
            shape    = ', '.join(self._print(i) for i in shape)
            return 'reshape(['+ elements + '], '+ '[' + shape + ']' + ')'
        fs = ', '.join(self._print(f) for f in expr)
        return '[{0}]'.format(fs)

    def _print_PythonList(self, expr):
        return self._print_PythonTuple(expr)

    def _print_TupleVariable(self, expr):
        if expr.is_homogeneous:
            return self._print_Variable(expr)
        else:
            fs = ', '.join(self._print(f) for f in expr)
            return '[{0}]'.format(fs)

    def _print_Variable(self, expr):
        return self._print(expr.name)

    def _print_ValuedVariable(self, expr):
        if expr.is_argument:
            return self._print_Variable(expr)
        else:
            return '{} = {}'.format(self._print(expr.name), self._print(expr.value))

    def _print_Constant(self, expr):
        val = LiteralFloat(expr.value)
        return self._print(val)

    def _print_DottedVariable(self, expr):
        if isinstance(expr.lhs, FunctionCall):
            base = expr.lhs.funcdef.results[0]
            if (not self._additional_code):
                self._additional_code = ''
            var_name = self.parser.get_new_name()
            var = base.clone(var_name)

            self.add_vars_to_namespace(var)

            self._additional_code = self._additional_code + self._print(Assign(var,expr.lhs)) + '\n'
            return self._print(var) + '%' +self._print(expr.name)
        else:
            return self._print(expr.lhs) + '%' +self._print(expr.name)

    def _print_DottedName(self, expr):
        return ' % '.join(self._print(n) for n in expr.name)

    def _print_Lambda(self, expr):
        return '"{args} -> {expr}"'.format(args=expr.variables, expr=expr.expr)

#    # TODO this is not used anymore since, we are calling printer inside
#    #      numpyext. must be improved!!
#    def _print_ZerosLike(self, expr):
#        lhs = self._print(expr.lhs)
#        rhs = self._print(expr.rhs)
#        if isinstance(expr.rhs, IndexedElement):
#            shape = []
#            for i in expr.rhs.indices:
#                if isinstance(i, Slice):
#                    shape.append(i)
#            rank = len(shape)
#        else:
#            rank = expr.rhs.rank
#        rs = []
#        for i in range(1, rank+1):
#            l = 'lbound({0},{1})'.format(rhs, str(i))
#            u = 'ubound({0},{1})'.format(rhs, str(i))
#            r = '{0}:{1}'.format(l, u)
#            rs.append(r)
#        shape = ', '.join(self._print(i) for i in rs)
#        init_value = self._print(expr.init_value)
#
#        code  = ('allocate({lhs}({shape}))\n'
#                 '{lhs} = {init_value}').format(lhs=lhs,
#                                                shape=shape,
#                                                init_value=init_value)
#
#        return self._get_statement(code)


    def _print_SumFunction(self, expr):
        return str(expr)

    def _print_PythonLen(self, expr):
        var = expr.arg
        idx = 1 if var.order == 'F' else var.rank
        prec = self.print_kind(expr)

        dtype = var.dtype
        if dtype is NativeString():
            return 'len({})'.format(self._print(var))
        elif var.rank == 1:
            return 'size({}, kind={})'.format(self._print(var), prec)
        else:
            return 'size({},{},{})'.format(self._print(var), self._print(idx), prec)

    def _print_PythonSum(self, expr):
        args = [self._print(arg) for arg in expr.args]
        return "sum({})".format(", ".join(args))

    def _print_PythonReal(self, expr):
        value = self._print(expr.internal_var)
        return 'real({0})'.format(value)
    def _print_PythonImag(self, expr):
        value = self._print(expr.internal_var)
        return 'aimag({0})'.format(value)



    #========================== Numpy Elements ===============================#

    def _print_NumpySum(self, expr):
        """Fortran print."""

        rhs_code = self._print(expr.arg)
        return 'sum({0})'.format(rhs_code)

    def _print_NumpyProduct(self, expr):
        """Fortran print."""

        rhs_code = self._print(expr.arg)
        return 'product({0})'.format(rhs_code)

    def _print_NumpyMatmul(self, expr):
        """Fortran print."""
        a_code = self._print(expr.a)
        b_code = self._print(expr.b)

        if expr.a.order and expr.b.order:
            if expr.a.order != expr.b.order:
                raise NotImplementedError("Mixed order matmul not supported.")

        # Fortran ordering
        if expr.a.order == 'F':
            return 'matmul({0},{1})'.format(a_code, b_code)

        # C ordering
        return 'matmul({1},{0})'.format(a_code, b_code)

    def _print_NumpyEmpty(self, expr):
        errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION, symbol=expr, severity='fatal')

    def _print_NumpyNorm(self, expr):
        """Fortran print."""
        arg = PythonAbs(expr.arg) if isinstance(expr.arg.dtype, NativeComplex) else expr.arg
        if expr.axis:
            axis = expr.axis
            if expr.order != 'F':
                axis = PyccelMinus(LiteralInteger(arg.rank), expr.axis)
            else:
                axis = LiteralInteger(expr.axis.python_value + 1)
            code = 'Norm2({},{})'.format(self._print(arg), self._print(axis))
        else:
            code = 'Norm2({})'.format(self._print(arg))

        return code

    def _print_NumpyLinspace(self, expr):

        template = '[({start} + {index}*{step},{index} = {zero},{end})]'

        init_value = template.format(
            start = self._print(expr.start),
            step  = self._print(expr.step ),
            index = self._print(expr.index),
            zero  = self._print(LiteralInteger(0)),
            end   = self._print(PyccelMinus(expr.size, LiteralInteger(1), simplify = True)),
        )
        code = init_value

        return code

    def _print_NumpyWhere(self, expr):

        ind   = self._print(expr.index)
        mask  = self._print(expr.mask)

        stmt  = 'pack([({ind},{ind}=0,size({mask})-1)],{mask})'.format(ind=ind,mask=mask)

        return stmt

    def _print_NumpyArray(self, expr):
        """Fortran print."""

        # If Numpy array is stored with column-major ordering, transpose values
        # use reshape with order for rank > 2
        if expr.order == 'F':
            if expr.rank == 2:
                rhs_code = self._print(expr.arg)
                rhs_code = 'transpose({})'.format(rhs_code)
            elif expr.rank > 2:
                args     = [self._print(a) for a in expr.arg]
                new_args = []
                for ac, a in zip(args, expr.arg):
                    if a.order == 'C':
                        shape    = ', '.join(self._print(i) for i in a.shape)
                        order    = ', '.join(self._print(LiteralInteger(i)) for i in range(a.rank, 0, -1))
                        ac       = 'reshape({}, [{}], order=[{}])'.format(ac, shape, order)
                    new_args.append(ac)

                args     = new_args
                rhs_code = '[' + ' ,'.join(args) + ']'
                shape    = ', '.join(self._print(i) for i in expr.shape)
                order    = [LiteralInteger(i) for i in range(1, expr.rank+1)]
                order    = order[1:]+ order[:1]
                order    = ', '.join(self._print(i) for i in order)
                rhs_code = 'reshape({}, [{}], order=[{}])'.format(rhs_code, shape, order)
        elif expr.order == 'C':
            if expr.rank > 2:
                args     = [self._print(a) for a in expr.arg]
                new_args = []
                for ac, a in zip(args, expr.arg):
                    if a.order == 'F':
                        shape    = ', '.join(self._print(i) for i in a.shape[::-1])
                        order    = ', '.join(self._print(LiteralInteger(i)) for i in range(a.rank, 0, -1))
                        ac       = 'reshape({}, [{}], order=[{}])'.format(ac, shape, order)
                    new_args.append(ac)

                args     = new_args
                rhs_code = '[' + ' ,'.join(args) + ']'
                shape    = ', '.join(self._print(i) for i in expr.shape[::-1])
                rhs_code = 'reshape({}, [{}])'.format(rhs_code, shape)
            else:
                rhs_code = self._print(expr.arg)
        return rhs_code

    def _print_NumpyFloor(self, expr):
        result_code = self._print_MathFloor(expr)
        return 'real({}, {})'.format(result_code, self.print_kind(expr))

    def _print_NumpyArange(self, expr):
        start  = self._print(expr.start)
        step   = self._print(expr.step)
        shape  = PyccelMinus(expr.shape[0], LiteralInteger(1), simplify = True)
        index  = Variable(NativeInteger(), name =  self.parser.get_new_name('i'))

        self.add_vars_to_namespace(index)

        code = '[({start} + {step} * {index}, {index} = {0}, {shape}, {1})]'
        code = code.format(self._print(LiteralInteger(0)),
                           self._print(LiteralInteger(1)),
                           start  = start,
                           step   = step,
                           index  = self._print(index),
                           shape  = self._print(shape))

        return code

    def _print_NumpyMod(self, expr):
        return self._print(PyccelMod(*expr.args))

    # ======================================================================= #
    def _print_PyccelArraySize(self, expr):
        init_value = self._print(expr.arg)
        prec = self.print_kind(expr)

        if expr.arg.order == 'C':
            index = PyccelMinus(LiteralInteger(expr.arg.rank), expr.index, simplify = True)
            index = self._print(index)
        else:
            index = PyccelAdd(expr.index, LiteralInteger(1), simplify = True)
            index = self._print(index)

        if expr.arg.rank == 1:
            return 'size({0}, kind={1})'.format(init_value, prec)

        return 'size({0}, {1}, {2})'.format(init_value, index, prec)

    def _print_PythonInt(self, expr):
        value = self._print(expr.arg)
        if (expr.arg.dtype is NativeBool()):
            code = 'MERGE(1_{0}, 0_{1}, {2})'.format(self.print_kind(expr), self.print_kind(expr),value)
        else:
            code  = 'Int({0}, {1})'.format(value, self.print_kind(expr))
        return code

    def _print_PythonFloat(self, expr):
        value = self._print(expr.arg)
        if (expr.arg.dtype is NativeBool()):
            code = 'MERGE(1.0_{0}, 0.0_{1}, {2})'.format(self.print_kind(expr), self.print_kind(expr),value)
        else:
            code  = 'Real({0}, {1})'.format(value, self.print_kind(expr))
        return code

    def _print_MathFloor(self, expr):
        arg = expr.args[0]
        arg_code = self._print(arg)

        # math.floor on integer argument is identity,
        # but we need parentheses around expressions
        if arg.dtype is NativeInteger():
            return '({})'.format(arg_code)

        prec = expr.precision
        prec_code = self._print(prec)
        return 'floor({}, kind={})'.format(arg_code, prec_code)

    def _print_PythonComplex(self, expr):
        if expr.is_cast:
            code = 'cmplx({0}, kind={1})'.format(expr.internal_var,
                                self.print_kind(expr))
        else:
            real = self._print(expr.real)
            imag = self._print(expr.imag)
            code = 'cmplx({0}, {1}, {2})'.format(real, imag,
                                self.print_kind(expr))
        return code

    def _print_PythonBool(self, expr):
        if isinstance(expr.arg.dtype, NativeBool):
            return 'logical({}, kind = {prec})'.format(self._print(expr.arg), prec = self.print_kind(expr))
        else:
            return '{} /= 0'.format(self._print(expr.arg))

    def _print_NumpyRand(self, expr):
        if expr.rank != 0:
            errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')

        if (not self._additional_code):
            self._additional_code = ''
        var_name = self.parser.get_new_name()
        var = Variable(expr.dtype, var_name, is_stack_array = all([s.is_constant for s in expr.shape]),
                shape = expr.shape, precision = expr.precision,
                order = expr.order, rank = expr.rank)

        self.add_vars_to_namespace(var)

        self._additional_code = self._additional_code + self._print(Assign(var,expr)) + '\n'
        return self._print(var)

    def _print_NumpyRandint(self, expr):
        if expr.rank != 0:
            errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')
        if expr.low is None:
            randreal = self._print(PyccelMul(expr.high, NumpyRand(), simplify = True))
        else:
            randreal = self._print(PyccelAdd(PyccelMul(PyccelMinus(expr.high, expr.low, simplify = True), NumpyRand(), simplify=True), expr.low, simplify = True))

        prec_code = self.print_kind(expr)
        return 'floor({}, kind={})'.format(randreal, prec_code)

    def _print_NumpyFull(self, expr):

        # Create statement for initialization
        init_value = self._print(expr.fill_value)
        return init_value

    def _print_PythonMin(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            code = 'minval({0})'.format(self._print(arg))
        else:
            code = ','.join(self._print(arg) for arg in args)
            code = 'min('+code+')'
        return self._get_statement(code)

    def _print_PythonMax(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            code = 'maxval({0})'.format(self._print(arg))
        else:
            code = ','.join(self._print(arg) for arg in args)
            code = 'max('+code+')'
        return self._get_statement(code)

    def _print_Dot(self, expr):
        return self._get_statement('dot_product(%s,%s)'%(self._print(expr.expr_l), self._print(expr.expr_r)))

    def _print_Ceil(self, expr):
        return self._get_statement('ceiling(%s)'%(self._print(expr.rhs)))

    def _print_Mod(self, expr):
        args = ','.join(self._print(i) for i in expr.args)
        return 'modulo({})'.format(args)

    def _print_Sign(self, expr):
        # TODO use the appropriate precision from rhs
        return self._get_statement('sign(1.0d0,%s)'%(self._print(expr.rhs)))

    # ... MACROS
    def _print_MacroShape(self, expr):
        var = expr.argument
        if not isinstance(var, (Variable, IndexedElement)):
            raise TypeError('Expecting a variable, given {}'.format(type(var)))
        shape = None
        if isinstance(var, Variable):
            shape = var.shape

        if shape is None:
            rank = var.rank
            shape = []
            for i in range(0, rank):
                l = 'lbound({var},{i})'.format(var=self._print(var),
                                               i=self._print(i+1))
                u = 'ubound({var},{i})'.format(var=self._print(var),
                                               i=self._print(i+1))
                s = '{u}-{l}+1'.format(u=u, l=l)
                shape.append(s)

        if len(shape) == 1:
            shape = shape[0]


        elif not(expr.index is None):
            if expr.index < len(shape):
                shape = shape[expr.index]
            else:
                shape = '1'

        code = '{}'.format(self._print(shape))

        return self._get_statement(code)
    # ...
    def _print_MacroType(self, expr):
        dtype = self._print(expr.argument.dtype)
        prec  = expr.argument.precision

        if dtype == 'integer':
            if prec==4:
                return 'MPI_INTEGER'
            elif prec==8:
                return 'MPI_INTEGER8'
            else:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')

        elif dtype == 'real':
            if prec==8:
                return 'MPI_DOUBLE'
            if prec==4:
                return 'MPI_FLOAT'
            else:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')

        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

    def _print_MacroCount(self, expr):

        var = expr.argument
        #TODO calculate size when type is pointer
        # it must work according to fortran documentation
        # but it raises somehow an error when it's a pointer
        # and shape is None

        if isinstance(var, Variable):
            shape = var.shape
            if not isinstance(shape,(tuple,list)):
                shape = [shape]
            rank = len(shape)
            if shape is None:
                return 'size({})'.format(self._print(var))


        elif isinstance(var, IndexedElement):
            _shape = var.base.shape
            if _shape is None:
                return 'size({})'.format(self._print(var))

            shape = []
            for (s, i) in zip(_shape, var.indices):
                if isinstance(i, Slice):
                    if i.start is None and i.stop is None:
                        shape.append(s)
                    elif i.start is None:
                        if (isinstance(i.stop, (int, LiteralInteger)) and i.stop>0) or not(isinstance(i.stop, (int, LiteralInteger))):
                            shape.append(i.stop)
                    elif i.stop is None:
                        if (isinstance(i.start, (int, LiteralInteger)) and i.start<s-1) or not(isinstance(i.start, (int, LiteralInteger))):
                            shape.append(PyccelMinus(s, i.start, simplify = True))
                    else:
                        shape.append(PyccelMinus(i.stop, PyccelAdd(i.start, LiteralInteger(1), simplify = True), simplify = True))

            rank = len(shape)

        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        if rank == 0:
            return '1'

        return str(functools.reduce(operator.mul, shape ))

    def _print_Declare(self, expr):
        # ... ignored declarations
        # we don't print the declaration if iterable object
        if is_iterable_datatype(expr.dtype):
            return ''

        if is_with_construct_datatype(expr.dtype):
            return ''

        if isinstance(expr.dtype, NativeSymbol):
            return ''

        if isinstance(expr.dtype, NativeRange):
            return ''

        # meta-variables
        if (isinstance(expr.variable, Variable) and
            expr.variable.name.startswith('__')):
            return ''
        # ...

        if isinstance(expr.variable, TupleVariable) and not expr.variable.is_homogeneous:
            return ''.join(self._print_Declare(Declare(v.dtype,v,intent=expr.intent, static=expr.static)) for v in expr.variable)

        # ... TODO improve
        # Group the variables by intent
        var = expr.variable
        rank        = var.rank
        allocatable = var.allocatable
        shape       = var.alloc_shape
        is_pointer = var.is_pointer
        is_target = var.is_target
        is_const = var.is_const
        is_stack_array = var.is_stack_array
        is_optional = var.is_optional
        is_static = expr.static
        intent = expr.intent

        if isinstance(shape, (tuple,PythonTuple)) and len(shape) ==1:
            shape = shape[0]
        # ...

        # ... print datatype
        if isinstance(expr.dtype, CustomDataType):
            dtype = expr.dtype

            name   = dtype.__class__.__name__
            prefix = dtype.prefix
            alias  = dtype.alias

            if dtype.is_polymorphic or expr.passed_from_dotted:
                sig = 'class'
            else:
                sig = 'type'

            if alias is None:
                name = name.replace(prefix, '')
            else:
                name = alias
            dtype = '{0}({1})'.format(sig, name)
        else:
            if isinstance(expr.dtype, NativeTuple):
                # Non-homogenous NativeTuples must be stored in TupleVariable
                if not expr.variable.is_homogeneous:
                    errors.report(LIST_OF_TUPLES,
                                  symbol=expr.variable, severity='error')
                    expr_dtype = NativeInteger()
                else:
                    expr_dtype = expr.variable.homogeneous_dtype
            else:
                expr_dtype = expr.dtype
            dtype = self._print(expr_dtype)

        # ...
            if isinstance(expr_dtype, NativeString):

                if expr.intent:
                    dtype = dtype[:9] +'(len =*)'
                    #TODO improve ,this is the case of character as argument
            else:
                dtype += '({0})'.format(self.print_kind(expr.variable))

        code_value = ''
        if expr.value:
            code_value = ' = {0}'.format(expr.value)

        vstr = self._print(expr.variable.name)

        # arrays are 0-based in pyccel, to avoid ambiguity with range
        s = '0'
        if not(is_static) and (allocatable or (var.shape is None)):
            s = ''

        # Default empty strings
        intentstr      = ''
        allocatablestr = ''
        optionalstr    = ''
        rankstr        = ''

        # Compute intent string
        if intent:
            if intent == 'in' and rank == 0 and not (is_static and is_optional):
                intentstr = ', value'
                if is_const:
                    intentstr += ', intent(in)'
            else:
                intentstr = ', intent({})'.format(intent)

        # Compute allocatable string
        if not is_static:
            if is_pointer:
                allocatablestr = ', pointer'

            elif allocatable and not intent:
                allocatablestr = ', allocatable'

            # ISSUES #177: var is allocatable and target
            if is_target:
                allocatablestr = '{}, target'.format(allocatablestr)

        # Compute optional string
        if is_optional:
            optionalstr = ', optional'

        # Compute rank string
        # TODO: improve
        if ((rank == 1) and (isinstance(shape, (int, LiteralInteger, Variable, PyccelAdd))) and
            (not(allocatable or is_pointer) or is_static or is_stack_array)):
            rankstr = '({0}:{1})'.format(self._print(s), self._print(PyccelMinus(shape, LiteralInteger(1), simplify = True)))

        elif ((rank > 0) and (isinstance(shape, (PythonTuple, tuple))) and
            (not(allocatable or is_pointer) or is_static or is_stack_array)):
            #TODO fix bug when we include shape of type list

            if var.order == 'C':
                rankstr = ','.join('{0}:{1}'.format(self._print(s),
                                                      self._print(PyccelMinus(i, LiteralInteger(1), simplify = True))) for i in shape[::-1])
            else:
                rankstr =  ','.join('{0}:{1}'.format(self._print(s),
                                                     self._print(PyccelMinus(i, LiteralInteger(1), simplify = True))) for i in shape)
            rankstr = '({rank})'.format(rank=rankstr)

        elif (rank > 0) and allocatable and intent:
            rankstr = '({})'.format(','.join(['0:'] * rank))

        elif (rank > 0) and (allocatable or is_pointer):
            rankstr = '({})'.format(','.join( [':'] * rank))

#        else:
#            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
#                severity='fatal')

        # Construct declaration
        left  = dtype + intentstr + allocatablestr + optionalstr
        right = vstr + rankstr + code_value
        return '{} :: {}\n'.format(left, right)

    def _print_AliasAssign(self, expr):
        code = ''
        lhs = expr.lhs
        rhs = expr.rhs

        if isinstance(lhs, TupleVariable) and not lhs.is_homogeneous:
            return self._print(CodeBlock([AliasAssign(l, r) for l,r in zip(lhs,rhs)]))

        if isinstance(rhs, Dlist):
            pattern = 'allocate({lhs}(0:{length}-1))\n{lhs} = {init_value}\n'
            code = pattern.format(lhs=self._print(lhs),
                                  length=self._print(rhs.length),
                                  init_value=self._print(rhs.val))
            return code

        # TODO improve
        op = '=>'
        shape_code = ''
        if lhs.rank > 0:
            shape_code = ', '.join('0:' for i in range(lhs.rank))
            shape_code = '({s_c})'.format(s_c = shape_code)

        code += '{lhs}{s_c} {op} {rhs}'.format(lhs=self._print(expr.lhs),
                                          s_c = shape_code,
                                          op=op,
                                          rhs=self._print(expr.rhs))

        return self._get_statement(code) + '\n'

    def _print_CodeBlock(self, expr):
        if not expr.unravelled:
            body_exprs, new_vars = expand_to_loops(expr, self.parser.get_new_variable, language_has_vectors = True)
            self.add_vars_to_namespace(*new_vars)
        else:
            body_exprs = expr.body
        body_stmts = []
        for b in body_exprs :
            line = self._print(b)
            if (self._additional_code):
                body_stmts.append(self._additional_code)
                self._additional_code = None
            body_stmts.append(line)
        return ''.join(self._print(b) for b in body_stmts)

    # TODO the ifs as they are are, is not optimal => use elif
    def _print_SymbolicAssign(self, expr):
        errors.report(FOUND_SYMBOLIC_ASSIGN,
                      symbol=expr.lhs, severity='warning')

        stmt = Comment(str(expr))
        return self._print_Comment(stmt)

    def _print_NumpyReal(self, expr):
        value = self._print(expr.internal_var)
        code = 'Real({0}, {1})'.format(value, self.print_kind(expr))
        return code

    def _print_Assign(self, expr):

        lhs_code = self._print(expr.lhs)
        rhs = expr.rhs
        # we don't print Range
        # TODO treat the case of iterable classes
        if isinstance(rhs, NINF):
            rhs_code = '-Huge({0})'.format(lhs_code)
            return '{0} = {1}\n'.format(lhs_code, rhs_code)

        if isinstance(rhs, INF):
            rhs_code = 'Huge({0})'.format(lhs_code)
            return '{0} = {1}\n'.format(lhs_code, rhs_code)

        if isinstance(rhs, (PythonRange, Product)):
            return ''

        if isinstance(rhs, NumpyRand):
            return 'call random_number({0})\n'.format(self._print(expr.lhs))

        if isinstance(rhs, NumpyEmpty):
            return ''

        if isinstance(rhs, ConstructorCall):
            func = rhs.func
            name = str(func.name)

            # TODO uncomment later

#            # we don't print the constructor call if iterable object
#            if this.dtype.is_iterable:
#                return ''
#
#            # we don't print the constructor call if with construct object
#            if this.dtype.is_with_construct:
#                return ''

            if name == "__init__":
                name = "create"
            rhs_code = self._print(name)
            rhs_code = '{0} % {1}'.format(lhs_code, rhs_code)

            code_args = ', '.join(self._print(i) for i in rhs.arguments)
            return 'call {0}({1})\n'.format(rhs_code, code_args)

        if isinstance(rhs, FunctionCall):

            # in the case of a function that returns a list,
            # we should append them to the procedure arguments
            if isinstance(expr.lhs, (tuple, list, PythonTuple)):

                rhs_code = rhs.funcdef.name
                args = rhs.args
                code_args = [self._print(i) for i in args]
                func = rhs.funcdef
                output_names = func.results
                lhs_code = [self._print(name) + ' = ' + self._print(i) for (name,i) in zip(output_names,expr.lhs)]

                call_args = ', '.join(code_args + lhs_code)

                code = 'call {0}({1})\n'.format(rhs_code, call_args)
                return self._get_statement(code)

        if (isinstance(expr.lhs, Variable) and
              expr.lhs.dtype == NativeSymbol()):
            return ''

        # Right-hand side code
        rhs_code = self._print(rhs)

        code = ''
        # if (expr.status == 'unallocated') and not (expr.like is None):
        #     stmt = ZerosLike(lhs=lhs_code, rhs=expr.like)
        #     code += self._print(stmt)
        #     code += '\n'
        code += '{0} = {1}'.format(lhs_code, rhs_code)
#        else:
#            code_args = ''
#            func = expr.rhs
#            # func here is of instance FunctionCall
#            cls_name = func.func.cls_name
#            keys = func.func.arguments

#            # for MPI statements, we need to add the lhs as the last argument
#            # TODO improve
#            if isinstance(func.func, MPI):
#                if not func.arguments:
#                    code_args = lhs_code
#                else:
#                    code_args = ', '.join(self._print(i) for i in func.arguments)
#                    code_args = '{0}, {1}'.format(code_args, lhs_code)
#            else:
#                _ij_print = lambda i, j: '{0}={1}'.format(self._print(i), \
#                                                         self._print(j))
#
#                code_args = ', '.join(_ij_print(i, j) \
#                                      for i, j in zip(keys, func.arguments))
#            if (not func.arguments is None) and (len(func.arguments) > 0):
#                if (not cls_name):
#                    code_args = ', '.join(self._print(i) for i in func.arguments)
#                    code_args = '{0}, {1}'.format(code_args, lhs_code)
#                else:
#            print('code_args > {0}'.format(code_args))
#            code = 'call {0}({1})'.format(rhs_code, code_args)
        return self._get_statement(code) + '\n'

#------------------------------------------------------------------------------
    def _print_Allocate(self, expr):

        # Transpose indices because of Fortran column-major ordering
        shape = expr.shape if expr.order == 'F' else expr.shape[::-1]

        var_code = self._print(expr.variable)
        size_code = ', '.join(self._print(i) for i in shape)
        shape_code = ', '.join('0:' + self._print(PyccelMinus(i, LiteralInteger(1), simplify = True)) for i in shape)
        code = ''

        if expr.status == 'unallocated':
            code += 'allocate({0}({1}))\n'.format(var_code, shape_code)

        elif expr.status == 'unknown':
            code += 'if (allocated({})) then\n'.format(var_code)
            code += '  if (any(size({}) /= [{}])) then\n'.format(var_code, size_code)
            code += '    deallocate({})\n'     .format(var_code)
            code += '    allocate({0}({1}))\n'.format(var_code, shape_code)
            code += '  end if\n'
            code += 'else\n'
            code += '  allocate({0}({1}))\n'.format(var_code, shape_code)
            code += 'end if\n'

        elif expr.status == 'allocated':
            code += 'if (any(size({}) /= [{}])) then\n'.format(var_code, size_code)
            code += '  deallocate({})\n'     .format(var_code)
            code += '  allocate({0}({1}))\n'.format(var_code, shape_code)
            code += 'end if\n'

        return code

#-----------------------------------------------------------------------------
    def _print_Deallocate(self, expr):
        return ''
#------------------------------------------------------------------------------

    def _print_NativeBool(self, expr):
        return 'logical'

    def _print_NativeInteger(self, expr):
        return 'integer'

    def _print_NativeReal(self, expr):
        return 'real'

    def _print_NativeComplex(self, expr):
        return 'complex'

    def _print_NativeString(self, expr):
        return 'character(len=280)'
        #TODO fix improve later

    def _print_DataType(self, expr):
        return self._print(expr.name)

    def _print_LiteralString(self, expr):
        sp_chars = ['\a', '\b', '\f', '\r', '\t', '\v', "'", '\n']
        sub_str = ''
        formatted_str = []
        for c in expr.arg:
            if c in sp_chars:
                if sub_str != '':
                    formatted_str.append("'{}'".format(sub_str))
                    sub_str = ''
                formatted_str.append('ACHAR({})'.format(ord(c)))
            else:
                sub_str += c
        if sub_str != '':
            formatted_str.append("'{}'".format(sub_str))
        return ' // '.join(formatted_str)

    def _print_Interface(self, expr):
        # ... we don't print 'hidden' functions
        name = self._print(expr.name)
        if expr.is_argument:
            funcs_sigs = []
            for f in expr.functions:
                self._handle_fortran_specific_a_prioris(list(f.arguments) + list(f.results))
                parts = self.function_signature(f, f.name)
                parts = ["{}({}) {}\n".format(parts['sig'], parts['arg_code'], parts['func_end']),
                        'use, intrinsic :: ISO_C_BINDING\n',
                        parts['arg_decs'],
                        'end {} {}\n'.format(parts['func_type'], f.name)]
                funcs_sigs.append(''.join(a for a in parts))
            interface = 'interface\n' + '\n'.join(a for a in funcs_sigs) + 'end interface\n'
            return interface

        if expr.functions[0].cls_name:
            for k, m in list(_default_methods.items()):
                name = name.replace(k, m)
            cls_name = expr.cls_name
            if not (cls_name == '__UNDEFINED__'):
                name = '{0}_{1}'.format(cls_name, name)
        else:
            for i in _default_methods:
                # because we may have a class Point with init: Point___init__
                if i in name:
                    name = name.replace(i, _default_methods[i])
        interface = 'interface ' + name +'\n'
        for f in expr.functions:
            interface += 'module procedure ' + str(f.name)+'\n'
        interface += 'end interface\n'
        return interface



   # def _print_With(self, expr):
   #     test = 'call '+self._print(expr.test) + '%__enter__()'
   #     body = self._print(expr.body)
   #     end = 'call '+self._print(expr.test) + '%__exit__()'
   #     code = ('{test}\n'
   #            '{body}\n'
   #            '{end}').format(test=test, body=body, end=end)
        #TODO return code later
  #      expr.block
  #      return ''

    def _print_Block(self, expr):

        decs=[]
        for i in expr.variables:
            dec = Declare(i.dtype, i)
            decs += [dec]
        body = expr.body

        body_code = self._print(body)
        prelude   = ''.join(self._print(i) for i in decs)


        #case of no local variables
        if len(decs) == 0:
            return body_code

        return ('{name} : Block\n'
                '{prelude}\n'
                 '{body}\n'
                'end Block {name}\n').format(name=expr.name, prelude=prelude, body=body_code)

    def _print_BindCFunctionDef(self, expr):
        name = self._print(expr.name)
        results   = list(expr.results)
        arguments = list(expr.arguments)
        if any([isinstance(a, FunctionAddress) for a in arguments]):
            # Functions with function addresses as arguments cannot be
            # exposed to python so there is no need to print their signature
            return ''
        arguments_inout = expr.arguments_inout
        args_decs = OrderedDict()
        for i,arg in enumerate(arguments):
            if arguments_inout[i]:
                intent='inout'
            else:
                intent='in'

            dec = Declare(arg.dtype, arg, intent=intent , static=True)
            args_decs[arg] = dec

        for result in results:
            dec = Declare(result.dtype, result, intent='out', static=True)
            args_decs[result] = dec

        if len(results) != 1:
            func_type = 'subroutine'
            func_end  = ''
        else:
            func_type = 'function'
            result = results.pop()
            func_end = 'result({0})'.format(result.name)
            dec = Declare(result.dtype, result, static=True)
            args_decs[result] = dec
        # ...

        interfaces = '\n'.join(self._print(i) for i in expr.interfaces)
        arg_code  = ', '.join(self._print(i) for i in chain( arguments, results ))
        imports   = ''.join(self._print(i) for i in expr.imports)
        prelude   = ''.join(self._print(i) for i in args_decs.values())
        body_code = self._print(expr.body)
        doc_string = self._print(expr.doc_string) if expr.doc_string else ''

        parts = [doc_string,
                '{0} {1}({2}) bind(c) {3}\n'.format(func_type, name, arg_code, func_end),
                 imports,
                'implicit none\n',
                 prelude,
                 interfaces,
                 body_code,
                 'end {} {}\n'.format(func_type, name)]
        return '\n'.join(p for p in parts if p)

    def _print_FunctionAddress(self, expr):
        return expr.name

    def function_signature(self, expr, name):
        is_pure      = expr.is_pure
        is_elemental = expr.is_elemental
        out_args = []
        args_decs = OrderedDict()

        func_end  = ''
        rec = 'recursive' if expr.is_recursive else ''
        if len(expr.results) != 1:
            func_type = 'subroutine'
            out_args = list(expr.results)
            for result in out_args:
                if result in expr.arguments:
                    dec = Declare(result.dtype, result, intent='inout')
                else:
                    dec = Declare(result.dtype, result, intent='out')
                args_decs[result] = dec

            functions = expr.functions

        else:
           #todo: if return is a function
            func_type = 'function'
            result = expr.results[0]
            functions = expr.functions

            func_end = 'result({0})'.format(result.name)

            dec = Declare(result.dtype, result)
            args_decs[result] = dec
        # ...

        for i,arg in enumerate(expr.arguments):
            if isinstance(arg, Variable):
                if i == 0 and expr.cls_name:
                    dec = Declare(arg.dtype, arg, intent='inout', passed_from_dotted = True)
                elif expr.arguments_inout[i]:
                    dec = Declare(arg.dtype, arg, intent='inout')
                else:
                    dec = Declare(arg.dtype, arg, intent='in')
                args_decs[arg] = dec

        #remove parametres intent(inout) from out_args to prevent repetition
        for i in expr.arguments:
            if i in out_args:
                out_args.remove(i)

        # treate case of pure function
        sig = '{0} {1} {2}'.format(rec, func_type, name)
        if is_pure:
            sig = 'pure {}'.format(sig)

        # treate case of elemental function
        if is_elemental:
            sig = 'elemental {}'.format(sig)

        arg_code  = ', '.join(self._print(i) for i in chain( expr.arguments, out_args ))

        arg_decs = ''.join(self._print(i) for i in args_decs.values())

        parts = {
                'sig' : sig,
                'arg_code' : arg_code,
                'func_end' : func_end,
                'arg_decs' : arg_decs,
                'func_type' : func_type
        }
        return parts

    def _print_FunctionDef(self, expr):
        self._handle_fortran_specific_a_prioris(list(expr.local_vars) +
                                                list(expr.arguments)  +
                                                list(expr.results))

        name = self._print(expr.name)
        self.set_current_function(name)

        if expr.cls_name:
            for k, m in list(_default_methods.items()):
                name = name.replace(k, m)

            cls_name = expr.cls_name
            if not (cls_name == '__UNDEFINED__'):
                name = '{0}_{1}'.format(cls_name, name)
        else:
            for i in _default_methods:
                # because we may have a class Point with init: Point___init__
                if i in name:
                    name = name.replace(i, _default_methods[i])

        sig_parts = self.function_signature(expr, name)
        prelude = sig_parts.pop('arg_decs')
        decs = OrderedDict()
        functions = expr.functions
        func_interfaces = '\n'.join(self._print(i) for i in expr.interfaces)
        body_code = self._print(expr.body)
        doc_string = self._print(expr.doc_string) if expr.doc_string else ''

        for i in expr.local_vars:
            dec = Declare(i.dtype, i)
            decs[i] = dec

        vars_to_print = self.parser.get_variables(self._namespace)
        for v in vars_to_print:
            if (v not in expr.local_vars) and (v not in expr.results) and (v not in expr.arguments):
                decs[v] = Declare(v.dtype,v)
        prelude += ''.join(self._print(i) for i in decs.values())
        if len(functions)>0:
            functions_code = '\n'.join(self._print(i) for  i in functions)
            body_code = body_code +'\ncontains\n' + functions_code

        imports = ''.join(self._print(i) for i in expr.imports)

        self.set_current_function(None)

        parts = [doc_string,
                "{}({}) {}\n".format(sig_parts['sig'], sig_parts['arg_code'], sig_parts['func_end']),
                imports,
                'implicit none\n',
                prelude,
                func_interfaces,
                body_code,
                'end {} {}\n'.format(sig_parts['func_type'], name)]

        return '\n'.join(a for a in parts if a)

    def _print_Pass(self, expr):
        return '! pass\n'

    def _print_Nil(self, expr):
        return ''

    def _print_Return(self, expr):
        code = ''
        if expr.stmt:
            code += self._print(expr.stmt)
        code +='return\n'
        return code

    def _print_Del(self, expr):
        # TODO: treate class case
        code = ''
        for var in expr.variables:
            if isinstance(var, Variable):
                dtype = var.dtype
                if is_pyccel_datatype(dtype):
                    code = 'call {0} % free()'.format(self._print(var))
                else:
                    code = 'deallocate({0}){1}'.format(self._print(var), code)
            else:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')
        return code + '\n'

    def _print_ClassDef(self, expr):
        # ... we don't print 'hidden' classes
        if expr.hide:
            return '', ''
        # ...

        name = self._print(expr.name)
        self.set_current_class(name)
        base = None # TODO: add base in ClassDef

        decs = ''.join(self._print(Declare(i.dtype, i)) for i in expr.attributes)

        aliases = []
        names   = []
        ls = [self._print(i.name) for i in expr.methods]
        for i in ls:
            j = _default_methods.get(i,i)
            aliases.append(j)
            names.append('{0}_{1}'.format(name, self._print(j)))
        methods = ''.join('procedure :: {0} => {1}\n'.format(i, j) for i, j in zip(aliases, names))
        for i in expr.interfaces:
            names = ','.join('{0}_{1}'.format(name, self._print(j.name)) for j in i.functions)
            methods += 'generic, public :: {0} => {1}\n'.format(self._print(i.name), names)
            methods += 'procedure :: {0}\n'.format(names)



        options = ', '.join(i for i in expr.options)

        sig = 'type, {0}'.format(options)
        if not(base is None):
            sig = '{0}, extends({1})'.format(sig, base)

        code = ('{0} :: {1}').format(sig, name)
        if len(decs) > 0:
            code = ('{0}\n'
                    '{1}').format(code, decs)
        if len(methods) > 0:
            code = ('{0}\n'
                    'contains\n'
                    '{1}').format(code, methods)
        decs = ('{0}\n'
                'end type {1}').format(code, name)

        sep = self._print(SeparatorComment(40))
        # we rename all methods because of the aliasing
        cls_methods = [i.clone('{0}'.format(i.name)) for i in expr.methods]
        for i in expr.interfaces:
            cls_methods +=  [j.clone('{0}'.format(j.name)) for j in i.functions]

        methods = ''
        for i in cls_methods:
            methods = ('{methods}\n'
                     '{sep}\n'
                     '{f}\n'
                     '{sep}\n').format(methods=methods, sep=sep, f=self._print(i))

        self.set_current_class(None)

        return decs, methods

    def _print_Break(self, expr):
        return 'exit\n'

    def _print_Continue(self, expr):
        return 'cycle\n'

    def _print_AugAssign(self, expr):
        new_expr = expr.to_basic_assign()
        return self._print(new_expr)

    def _print_PythonRange(self, expr):
        start = self._print(expr.start)
        step  = self._print(expr.step)

        test_step = expr.step
        if isinstance(test_step, PyccelUnarySub):
            test_step = expr.step.args[0]

        # testing if the step is a value or an expression
        if isinstance(test_step, Literal):
            if isinstance(expr.step, PyccelUnarySub):
                stop = PyccelAdd(expr.stop, LiteralInteger(1), simplify = True)
            else:
                stop = PyccelMinus(expr.stop, LiteralInteger(1), simplify = True)
        else:
            stop = IfTernaryOperator(PyccelGt(expr.step, LiteralInteger(0)),
                                     PyccelMinus(expr.stop, LiteralInteger(1), simplify = True),
                                     PyccelAdd(expr.stop, LiteralInteger(1), simplify = True))

        stop = self._print(stop)
        return '{0}, {1}, {2}'.format(start, stop, step)

    def _print_FunctionalFor(self, expr):
        loops = ''.join(self._print(i) for i in expr.loops)
        return loops

    def _print_For(self, expr):
        prolog = ''
        epilog = ''

        # ...

        def _do_range(target, iterable, prolog, epilog):
            if not isinstance(iterable, PythonRange):
                # Only iterable currently supported is PythonRange
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')

            tar        = self._print(target)
            range_code = self._print(iterable)

            prolog += 'do {0} = {1}\n'.format(tar, range_code)
            epilog = 'end do\n' + epilog

            return prolog, epilog
        # ...

        if not isinstance(expr.iterable, (PythonRange, Product , PythonZip,
                            PythonEnumerate, PythonMap)):
            # Only iterable currently supported are PythonRange or Product
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        if isinstance(expr.iterable, PythonRange):
            prolog, epilog = _do_range(expr.target, expr.iterable, \
                                       prolog, epilog)

        elif isinstance(expr.iterable, Product):
            for i, a in zip(expr.target, expr.iterable.elements):
                if isinstance(a, PythonRange):
                    itr_ = a
                else:
                    itr_ = PythonRange(a.shape[0])
                prolog, epilog = _do_range(i, itr_, \
                                           prolog, epilog)

        elif isinstance(expr.iterable, PythonZip):
            itr_ = PythonRange(expr.iterable.length)
            prolog, epilog = _do_range(expr.target, itr_, \
                                       prolog, epilog)

        elif isinstance(expr.iterable, PythonEnumerate):
            itr_ = PythonRange(PythonLen(expr.iterable.element))
            prolog, epilog = _do_range(expr.target, itr_, \
                                       prolog, epilog)

        elif isinstance(expr.iterable, PythonMap):
            itr_ = PythonRange(PythonLen(expr.iterable.args[1]))
            prolog, epilog = _do_range(expr.target, itr_, \
                                       prolog, epilog)

        body = self._print(expr.body)

        if expr.nowait_expr:
            epilog += expr.nowait_expr
        return ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)

    # .....................................................
    #               Print OpenMP AnnotatedComment
    # .....................................................

    def _print_OmpAnnotatedComment(self, expr):
        clauses = ''
        if expr.combined:
            combined = expr.combined.replace("for", "do")
            clauses = ' ' + combined

        omp_expr = '!$omp {}'.format(expr.name.replace("for", "do"))
        clauses += str(expr.txt).replace("cancel for", "cancel do")
        omp_expr = '{}{}\n'.format(omp_expr, clauses)

        return omp_expr

    def _print_Omp_End_Clause(self, expr):
        omp_expr = str(expr.txt)
        if "section" in omp_expr and "sections" not in omp_expr:
            return ''
        omp_expr = omp_expr.replace("for", "do")
        if expr.has_nowait:
            omp_expr += ' nowait'
        omp_expr = '!$omp {}\n'.format(omp_expr)
        return omp_expr

    # .....................................................
    def _print_OMP_Parallel(self, expr):
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        body    = ''.join(self._print(i) for i in expr.body)

        # ... TODO adapt get_statement to have continuation with OpenMP
        prolog = '!$omp parallel {clauses}\n'.format(clauses=clauses)
        epilog = '!$omp end parallel\n'
        # ...

        # ...
        code = ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_OMP_For(self, expr):
        # ...
        loop    = self._print(expr.loop)
        clauses = ' '.join(self._print(i)  for i in expr.clauses)

        nowait  = ''
        if not(expr.nowait is None):
            nowait = 'nowait'
        # ...
        # ... TODO adapt get_statement to have continuation with OpenMP
        prolog = '!$omp do {clauses}\n'.format(clauses=clauses)
        epilog = '!$omp end do {0}\n'.format(nowait)
        # ...

        # ...
        code = ('{prolog}'
                '{loop}'
                '{epilog}').format(prolog=prolog, loop=loop, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_OMP_NumThread(self, expr):
        return 'num_threads({})'.format(self._print(expr.num_threads))

    def _print_OMP_Default(self, expr):
        status = expr.status
        if status:
            status = self._print(expr.status)
        else:
            status = ''
        return 'default({})'.format(status)

    def _print_OMP_ProcBind(self, expr):
        status = expr.status
        if status:
            status = self._print(expr.status)
        else:
            status = ''
        return 'proc_bind({})'.format(status)

    def _print_OMP_Private(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'private({})'.format(args)

    def _print_OMP_Shared(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'shared({})'.format(args)

    def _print_OMP_FirstPrivate(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'firstprivate({})'.format(args)

    def _print_OMP_LastPrivate(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'lastprivate({})'.format(args)

    def _print_OMP_Copyin(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copyin({})'.format(args)

    def _print_OMP_Reduction(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        op   = self._print(expr.operation)
        return "reduction({0}: {1})".format(op, args)

    def _print_OMP_Schedule(self, expr):
        kind = self._print(expr.kind)

        chunk_size = ''
        if expr.chunk_size:
            chunk_size = ', {0}'.format(self._print(expr.chunk_size))

        return 'schedule({0}{1})'.format(kind, chunk_size)

    def _print_OMP_Ordered(self, expr):
        n_loops = ''
        if expr.n_loops:
            n_loops = '({0})'.format(self._print(expr.n_loops))

        return 'ordered{0}'.format(n_loops)

    def _print_OMP_Collapse(self, expr):
        n_loops = '{0}'.format(self._print(expr.n_loops))

        return 'collapse({0})'.format(n_loops)

    def _print_OMP_Linear(self, expr):
        variables= ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        step = self._print(expr.step)
        return "linear({0}: {1})".format(variables, step)

    def _print_OMP_If(self, expr):
        return 'if({})'.format(self._print(expr.test))
    # .....................................................

    # .....................................................
    #                   OpenACC statements
    # .....................................................
    def _print_ACC_Parallel(self, expr):
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        body    = ''.join(self._print(i) for i in expr.body)

        # ... TODO adapt get_statement to have continuation with OpenACC
        prolog = '!$acc parallel {clauses}\n'.format(clauses=clauses)
        epilog = '!$acc end parallel\n'
        # ...

        # ...
        code = ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_ACC_For(self, expr):
        # ...
        loop    = self._print(expr.loop)
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        # ...

        # ... TODO adapt get_statement to have continuation with OpenACC
        prolog = '!$acc loop {clauses}\n'.format(clauses=clauses)
        epilog = '!$acc end loop\n'
        # ...

        # ...
        code = ('{prolog}'
                '{loop}'
                '{epilog}').format(prolog=prolog, loop=loop, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_ACC_Async(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'async({})'.format(args)

    def _print_ACC_Auto(self, expr):
        return 'auto'

    def _print_ACC_Bind(self, expr):
        return 'bind({})'.format(self._print(expr.variable))

    def _print_ACC_Collapse(self, expr):
        return 'collapse({0})'.format(self._print(expr.n_loops))

    def _print_ACC_Copy(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copy({})'.format(args)

    def _print_ACC_Copyin(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copyin({})'.format(args)

    def _print_ACC_Copyout(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copyout({})'.format(args)

    def _print_ACC_Create(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'create({})'.format(args)

    def _print_ACC_Default(self, expr):
        return 'default({})'.format(self._print(expr.status))

    def _print_ACC_DefaultAsync(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'default_async({})'.format(args)

    def _print_ACC_Delete(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'delete({})'.format(args)

    def _print_ACC_Device(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'device({})'.format(args)

    def _print_ACC_DeviceNum(self, expr):
        return 'collapse({0})'.format(self._print(expr.n_device))

    def _print_ACC_DevicePtr(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'deviceptr({})'.format(args)

    def _print_ACC_DeviceResident(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'device_resident({})'.format(args)

    def _print_ACC_DeviceType(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'device_type({})'.format(args)

    def _print_ACC_Finalize(self, expr):
        return 'finalize'

    def _print_ACC_FirstPrivate(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'firstprivate({})'.format(args)

    def _print_ACC_Gang(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'gang({})'.format(args)

    def _print_ACC_Host(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'host({})'.format(args)

    def _print_ACC_If(self, expr):
        return 'if({})'.format(self._print(expr.test))

    def _print_ACC_Independent(self, expr):
        return 'independent'

    def _print_ACC_Link(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'link({})'.format(args)

    def _print_ACC_NoHost(self, expr):
        return 'nohost'

    def _print_ACC_NumGangs(self, expr):
        return 'num_gangs({0})'.format(self._print(expr.n_gang))

    def _print_ACC_NumWorkers(self, expr):
        return 'num_workers({0})'.format(self._print(expr.n_worker))

    def _print_ACC_Present(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'present({})'.format(args)

    def _print_ACC_Private(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'private({})'.format(args)

    def _print_ACC_Reduction(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        op   = self._print(expr.operation)
        return "reduction({0}: {1})".format(op, args)

    def _print_ACC_Self(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'self({})'.format(args)

    def _print_ACC_Seq(self, expr):
        return 'seq'

    def _print_ACC_Tile(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'tile({})'.format(args)

    def _print_ACC_UseDevice(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'use_device({})'.format(args)

    def _print_ACC_Vector(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'vector({})'.format(args)

    def _print_ACC_VectorLength(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'vector_length({})'.format(self._print(expr.n))

    def _print_ACC_Wait(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'wait({})'.format(args)

    def _print_ACC_Worker(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'worker({})'.format(args)
    # .....................................................

    def _print_ForIterator(self, expr):
        return self._print_For(expr)

        prolog = ''
        epilog = ''

        # ...
        def _do_range(target, iterable, prolog, epilog):
            tar        = self._print(target)
            range_code = self._print(iterable)

            prolog += 'do {0} = {1}\n'.format(tar, range_code)
            epilog = 'end do\n' + epilog

            return prolog, epilog
        # ...

        # ...
        if not isinstance(expr.iterable, (Variable, ConstructorCall)):
            raise TypeError('iterable must be Variable or ConstructorCall.')
        # ...

        # ...
        targets = expr.target
        if isinstance(expr.iterable, Variable):
            iters = expr.ranges
        elif isinstance(expr.iterable, ConstructorCall):
            iters = get_iterable_ranges(expr.iterable)
        # ...

        # ...
        for i,a in zip(targets, iters):
            prolog, epilog = _do_range(i, a, \
                                       prolog, epilog)

        body = ''.join(self._print(i) for i in expr.body)
        # ...

        return ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)


    #def _print_Block(self, expr):
    #    body    = '\n'.join(self._print(i) for i in expr.body)
    #    prelude = '\n'.join(self._print(i) for i in expr.declarations)
    #    return prelude, body

    def _print_While(self,expr):
        body = self._print(expr.body)
        return ('do while ({test})\n'
                '{body}'
                'end do\n').format(test=self._print(expr.test), body=body)

    def _print_ErrorExit(self, expr):
        # TODO treat the case of MPI
        return 'STOP'

    def _print_Assert(self, expr):
        # we first create an If statement
        # TODO: depending on a debug flag we should print 'PASSED' or not.
        DEBUG = True

        err = ErrorExit()
        args = [IfSection(PyccelNot(expr.test), [PythonPrint(["'Assert Failed'"]), err])]

        if DEBUG:
            args.append((True, PythonPrint(["'PASSED'"])))

        stmt = If(*args)
        code = self._print(stmt)
        return self._get_statement(code)

    def _print_PyccelIs(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        a = expr.args[0]
        b = expr.args[1]

        if isinstance(expr.rhs, Nil):
            return '.not. present({})'.format(lhs)

        if (a.dtype is NativeBool() and b.dtype is NativeBool()):
            return '{} .eqv. {}'.format(lhs, rhs)

        errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                      symbol=expr, severity='fatal')

    def _print_PyccelIsNot(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        a = expr.args[0]
        b = expr.args[1]

        if isinstance(expr.rhs, Nil):
            return 'present({})'.format(lhs)

        if a.dtype is NativeBool() and b.dtype is NativeBool():
            return '{} .neqv. {}'.format(lhs, rhs)

        errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                      symbol=expr, severity='fatal')

    def _print_If(self, expr):
        # ...

        lines = []

        for i, (c, e) in enumerate(expr.blocks):

            if i == 0:
                lines.append("if (%s) then\n" % self._print(c))
            elif i == len(expr.blocks) - 1 and isinstance(c, LiteralTrue):
                lines.append("else\n")
            else:
                lines.append("else if (%s) then\n" % self._print(c))

            if isinstance(e, (list, tuple, PythonTuple)):
                lines.extend(self._print(ee) for ee in e)
            else:
                lines.append(self._print(e))

        lines.append("end if\n")

        return ''.join(lines)

    def _print_IfTernaryOperator(self, expr):

        cond = PythonBool(expr.cond) if not isinstance(expr.cond.dtype, NativeBool) else expr.cond
        value_true = expr.value_true
        value_false = expr.value_false

        if value_true.dtype != value_false.dtype :
            try :
                cast_func = python_builtin_datatypes[str_dtype(expr.dtype)]
            except KeyError:
                errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
            value_true = cast_func(value_true) if value_true.dtype != expr.dtype else value_true
            value_false = cast_func(value_false) if value_false.dtype != expr.dtype else value_false
        cond = self._print(cond)
        value_true = self._print(value_true)
        value_false = self._print(value_false)
        return 'merge({true}, {false}, {cond})'.format(cond = cond, true = value_true, false = value_false)

    def _print_PyccelPow(self, expr):
        base = expr.args[0]
        e    = expr.args[1]

        base_c = self._print(base)
        e_c    = self._print(e)
        return '{} ** {}'.format(base_c, e_c)

    def _print_PyccelAdd(self, expr):
        if expr.dtype is NativeString():
            return '//'.join('trim('+self._print(a)+')' for a in expr.args)
        else:
            return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        args = [self._print(a) for a in expr.args]

        if len(args) == 1:
            return '-{}'.format(args[0])
        return ' - '.join(args)

    def _print_PyccelMul(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' * '.join(a for a in args)

    def _print_PyccelDiv(self, expr):
        if all(a.dtype is NativeInteger() for a in expr.args):
            args = [NumpyFloat(a) for a in expr.args]
        else:
            args = expr.args
        return ' / '.join(self._print(a) for a in args)

    def _print_PyccelMod(self, expr):
        is_real  = expr.dtype is NativeReal()

        def correct_type_arg(a):
            if is_real and a.dtype is NativeInteger():
                return NumpyFloat(a)
            else:
                return a

        args = [self._print(correct_type_arg(a)) for a in expr.args]

        code = args[0]
        for c in args[1:]:
            code = 'MODULO({},{})'.format(code, c)
        return code

    def _print_PyccelFloorDiv(self, expr):

        code   = self._print(expr.args[0])
        adtype = expr.args[0].dtype
        is_real  = expr.dtype is NativeReal()
        for b in expr.args[1:]:
            bdtype    = b.dtype
            if adtype is NativeInteger() and bdtype is NativeInteger():
                b = NumpyFloat(b)
            c = self._print(b)
            adtype = bdtype
            code = 'FLOOR({}/{},{})'.format(code, c, self.print_kind(expr))
            if is_real:
                code = 'real({}, {})'.format(code, self.print_kind(expr))
        return code

    def _print_PyccelRShift(self, expr):
        return 'RSHIFT({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelLShift(self, expr):
        return 'LSHIFT({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitXor(self, expr):
        if expr.dtype is NativeBool():
            return ' .neqv. '.join(self._print(a) for a in expr.args)
        return 'IEOR({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitOr(self, expr):
        if expr.dtype is NativeBool():
            return ' .or. '.join(self._print(a) for a in expr.args)
        return 'IOR({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitAnd(self, expr):
        if expr.dtype is NativeBool():
            return ' .and. '.join(self._print(a) for a in expr.args)
        return 'IAND({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelInvert(self, expr):
        return 'NOT({})'.format(self._print(expr.args[0]))

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '+{}'.format(self._print(expr.args[0]))

    def _print_PyccelUnarySub(self, expr):
        return '-{}'.format(self._print(expr.args[0]))

    def _print_PyccelAnd(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' .and. '.join(a for a in args)

    def _print_PyccelOr(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' .or. '.join(a for a in args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        a = expr.args[0].dtype
        b = expr.args[1].dtype

        if a is NativeBool() and b is NativeBool():
            return '{} .eqv. {}'.format(lhs, rhs)
        return '{0} == {1}'.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        a = expr.args[0].dtype
        b = expr.args[1].dtype

        if a is NativeBool() and b is NativeBool():
            return '{} .neqv. {}'.format(lhs, rhs)
        return '{0} /= {1}'.format(lhs, rhs)

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
        if (expr.args[0].dtype is not NativeBool()):
            return '{} == 0'.format(a)
        return '.not. {}'.format(a)

    def _print_Header(self, expr):
        return ''

    def _print_ConstructorCall(self, expr):
        func = expr.func
        name = str(func.name)
        if name == "__init__":
            name = "create"
        name = self._print(name)

        code_args = ''
        if expr.arguments is not None:
            code_args = ', '.join(self._print(i) for i in expr.arguments)
        code = '{0}({1})'.format(name, code_args)
        return self._get_statement(code)

    def _print_NumpyUfuncBase(self, expr):
        type_name = type(expr).__name__
        try:
            func_name = numpy_ufunc_to_fortran[type_name]
        except KeyError:
            self._print_not_supported(expr)
        args = [self._print(NumpyFloat(a) if a.dtype is NativeInteger() else a)\
				for a in expr.args]
        code_args = ', '.join(args)
        code = '{0}({1})'.format(func_name, code_args)
        return self._get_statement(code)

    def _print_MathFunctionBase(self, expr):
        """ Convert a Python expression with a math function call to Fortran
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Math function call

        Returns
        -------
            string
                Equivalent expression in Fortran language

        ------
        Example:
        --------
            math.cos(x)    ==> cos(x)
            math.gcd(x, y) ==> pyc_gcd(x, y) # with include of pyc_math module
        """
        type_name = type(expr).__name__
        try:
            func_name = math_function_to_fortran[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
        if func_name.startswith("pyc"):
            self._additional_imports.add('pyc_math')
        args = []
        for arg in expr.args:
            if arg.dtype != expr.dtype:
                cast_func = python_builtin_datatypes[str_dtype(expr.dtype)]
                args.append(self._print(cast_func(arg)))
            else:
                args.append(self._print(arg))
        code_args = ', '.join(args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_MathCeil(self, expr):
        """Convert a Python expression with a math ceil function call to
        Fortran function call"""
        # add necessary include
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "ceiling({})".format(code_arg)

    def _print_MathIsnan(self, expr):
        """Convert a Python expression with a math isnan function call to
        Fortran function call"""
        # add necessary include
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isnan({})".format(code_arg)

    def _print_MathTrunc(self, expr):
        """Convert a Python expression with a math trunc function call to
        Fortran function call"""
        # add necessary include
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "dint({})".format(code_arg)

    def _print_MathPow(self, expr):
        base = expr.args[0]
        e    = expr.args[1]

        base_c = self._print(base)
        e_c    = self._print(e)
        return '{} ** {}'.format(base_c, e_c)

    def _print_NumpySqrt(self, expr):
        arg = expr.args[0]
        if arg.dtype is NativeInteger() or arg.dtype is NativeBool():
            arg = NumpyFloat(arg)
        code_args = self._print(arg)
        code = 'sqrt({})'.format(code_args)
        return self._get_statement(code)

    def _print_LiteralImaginaryUnit(self, expr):
        """ purpose: print complex numbers nicely in Fortran."""
        return "cmplx(0,1, kind = {})".format(self.print_kind(expr))

    def _print_int(self, expr):
        return str(expr)

    def _print_Literal(self, expr):
        printed = repr(expr.python_value)
        return "{}_{}".format(printed, self.print_kind(expr))

    def _print_LiteralTrue(self, expr):
        return ".True._{}".format(self.print_kind(expr))

    def _print_LiteralFalse(self, expr):
        return ".False._{}".format(self.print_kind(expr))

    def _print_LiteralComplex(self, expr):
        real_str = self._print(expr.real)
        imag_str = self._print(expr.imag)
        return "({}, {})".format(real_str, imag_str)

    def _print_IndexedElement(self, expr):
        base = expr.base
        if isinstance(base, PyccelInternalFunction) and not isinstance(base, PythonTuple):
            indexed_type = base.dtype
            if isinstance(indexed_type, PythonTuple):
                base = self._print_PyccelInternalFunction(expr.base.base)
            else:
                if (not self._additional_code):
                    self._additional_code = ''
                var_name = self.parser.get_new_name()
                var = Variable(base.dtype, var_name, is_stack_array = True,
                        shape=base.shape,precision=base.precision,
                        order=base.order,rank=base.rank)

                self.add_vars_to_namespace(var)

                self._additional_code = self._additional_code + self._print(Assign(var,base)) + '\n'
                return self._print(var[expr.indices])
        elif isinstance(base, TupleVariable) and not base.is_homogeneous:
            if len(expr.indices)==1:
                return self._print(base[expr.indices[0]])
            else:
                var = base[expr.indices[0]]
                return self._print(var[expr.indices[1:]])
        else:
            base_code = self._print(base)

        inds = list(expr.indices)
        if expr.base.order == 'C':
            inds = inds[::-1]
        base_shape = Shape(expr.base)
        allow_negative_indexes = base.allows_negative_indexes

        for i, ind in enumerate(inds):
            _shape = PyccelArraySize(base, i if expr.order != 'C' else len(inds) - i - 1)
            if isinstance(ind, Slice):
                inds[i] = self._new_slice_with_processed_arguments(ind, _shape, allow_negative_indexes)
            elif isinstance(ind, PyccelUnarySub) and isinstance(ind.args[0], LiteralInteger):
                inds[i] = PyccelMinus(_shape, ind.args[0], simplify = True)
            else:
                #indices of indexedElement of len==1 shouldn't be a tuple
                if isinstance(ind, tuple) and len(ind) == 1:
                    inds[i] = ind[0]
                if allow_negative_indexes and not isinstance(ind, LiteralInteger):
                    inds[i] = IfTernaryOperator(PyccelLt(ind, LiteralInteger(0)),
                            PyccelAdd(base_shape[i], ind, simplify = True), ind)

        inds = [self._print(i) for i in inds]

        return "%s(%s)" % (base_code, ", ".join(inds))


    def _print_Idx(self, expr):
        return self._print(expr.label)

    @staticmethod
    def _new_slice_with_processed_arguments(_slice, array_size, allow_negative_index):
        """ Create new slice with informations collected from old slice and decorators

        Parameters
        ----------
            _slice : Slice
                slice needed to collect (start, stop, step)
            array_size : PyccelArraySize
                call to function size()
            allow_negative_index : Bool
                True when the decorator allow_negative_index is present
        Returns
        -------
            Slice
        """
        start = _slice.start
        stop = _slice.stop
        step = _slice.step

        # negative start and end in slice
        if isinstance(start, PyccelUnarySub) and isinstance(start.args[0], LiteralInteger):
            start = PyccelMinus(array_size, start.args[0], simplify = True)
        elif start is not None and allow_negative_index and not isinstance(start,LiteralInteger):
            start = IfTernaryOperator(PyccelLt(start, LiteralInteger(0)),
                        PyccelAdd(array_size, start, simplify = True), start)

        if isinstance(stop, PyccelUnarySub) and isinstance(stop.args[0], LiteralInteger):
            stop = PyccelMinus(array_size, stop.args[0], simplify = True)
        elif stop is not None and allow_negative_index and not isinstance(stop, LiteralInteger):
            stop = IfTernaryOperator(PyccelLt(stop, LiteralInteger(0)),
                        PyccelAdd(array_size, stop, simplify = True), stop)

        # negative step in slice
        if isinstance(step, PyccelUnarySub) and isinstance(step.args[0], LiteralInteger):
            stop = PyccelAdd(stop, LiteralInteger(1), simplify = True) if stop is not None else LiteralInteger(0)
            start = start if start is not None else PyccelMinus(array_size, LiteralInteger(1), simplify = True)

        # variable step in slice
        elif step and allow_negative_index and not isinstance(step, LiteralInteger):
            if start is None :
                start = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)),
                    LiteralInteger(0), PyccelMinus(array_size , LiteralInteger(1), simplify = True))

            if stop is None :
                stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)),
                    PyccelMinus(array_size, LiteralInteger(1), simplify = True), LiteralInteger(0))
            else :
                stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)),
                    stop, PyccelAdd(stop, LiteralInteger(1), simplify = True))

        elif stop is not None:
            stop = PyccelMinus(stop, LiteralInteger(1), simplify = True)

        return Slice(start, stop, step)

    def _print_Slice(self, expr):
        if expr.start is None or  isinstance(expr.start, Nil):
            start = ''
        else:
            start = self._print(expr.start)
        if (expr.stop is None) or isinstance(expr.stop, Nil):
            stop = ''
        else:
            stop = self._print(expr.stop)
        if expr.step is not None :
            return '{0}:{1}:{2}'.format(start, stop, self._print(expr.step))
        return '{0}:{1}'.format(start, stop)

#=======================================================================================

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
        f_name = self._print(expr.func_name if not expr.interface else expr.interface_name)
        args = [a for a in expr.args if not isinstance(a, Nil)]
        results = func.results

        if len(results) == 1:
            args = ['{}'.format(self._print(a)) for a in args]

            args = ', '.join(args)
            code = '{name}({args})'.format( name = f_name,
                                            args = args)

        elif len(results)>1:
            if (not self._additional_code):
                self._additional_code = ''
            out_vars = []
            for r in func.results:
                var_name = self.parser.get_new_name()
                var =  r.clone(name = var_name)

                self.add_vars_to_namespace(var)

                out_vars.append(var)

            self._additional_code = self._additional_code + self._print(Assign(tuple(out_vars),expr)) + '\n'
            return self._print(tuple(out_vars))
        else:
            args    = ['{}'.format(self._print(a)) for a in args]
            if not func.is_header:
                results = ['{0}={0}'.format(self._print(a)) for a in results]
            else:
                results = ['{}'.format(self._print(a)) for a in results]

            newargs = ', '.join(args+results)

            code = 'call {name}({args})\n'.format( name = f_name,
                                                 args = newargs )
        return code

#=======================================================================================

    def _print_DottedFunctionCall(self, expr):
        if isinstance(expr.prefix, FunctionCall):
            base = expr.prefix.funcdef.results[0]
            if (not self._additional_code):
                self._additional_code = ''
            var_name = self.parser.get_new_name()
            var = base.clone(var_name)

            self.add_vars_to_namespace(var)

            self._additional_code = self._additional_code + self._print(Assign(var,expr.prefix)) + '\n'
            expr = DottedFunctionCall(expr.funcdef, expr.args, var)
        return self._print_FunctionCall(expr)

#=======================================================================================

    def _print_PyccelInternalFunction(self, expr):
        if isinstance(expr, NumpyNewArray):
            return errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')
        else:
            return self._print_not_supported(expr)

#=======================================================================================

    def _pad_leading_columns(self, lines):
        result = []
        for line in lines:
            if line.startswith('!'):
                result.append("! " + line[1:].lstrip())
            else:
                result.append(line)
        return result

    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (ending with a \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
        # routine to find split point in a code line
        my_alnum = set("_+-." + string.digits + string.ascii_letters)
        my_white = set(" \t()")

        def split_pos_code(line, endpos):
            if len(line) <= endpos:
                return len(line)
            pos = endpos
            split = lambda pos: \
                (line[pos] in my_alnum and line[pos - 1] not in my_alnum) or \
                (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or \
                (line[pos] in my_white and line[pos - 1] not in my_white) or \
                (line[pos] not in my_white and line[pos - 1] in my_white)
            while not split(pos):
                pos -= 1
                if pos == 0:
                    return endpos
            return pos

        # split line by line and add the splitted lines to result
        result = []
        trailing = ' &'
        for line in lines:
            if len(line)>72 and ('"' in line[72:] or "'" in line[72:] or '!' in line[:72]):
                result.append(line)

            elif len(line)>72:
                # code line

                pos = split_pos_code(line, 72)
                hunk = line[:pos].rstrip()
                line = line[pos:].lstrip()
                if line:
                    hunk += trailing
                result.append(hunk)
                while len(line) > 0:
                    pos = split_pos_code(line, 65)
                    hunk = line[:pos].rstrip()
                    line = line[pos:].lstrip()
                    if line:
                        hunk += trailing
                    result.append("%s%s"%("      " , hunk))
            else:
                result.append(line)

        # make sure that all lines end with a carriage return
        return [l if l.endswith('\n') else l+'\n' for l in result]

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        code = [line.lstrip(' \t') for line in code]

        inc_keyword = ('do ', 'if(', 'if ', 'do\n',
                       'else', 'type', 'subroutine', 'function',
                       'interface')
        dec_keyword = ('end do', 'enddo', 'end if', 'endif',
                       'else', 'endtype', 'end type',
                       'endfunction', 'end function',
                       'endsubroutine', 'end subroutine',
                       'endinterface', 'end interface')

        increase = [int(any(map(line.startswith, inc_keyword)))
                     for line in code]
        decrease = [int(any(map(line.startswith, dec_keyword)))
                     for line in code]
        continuation = [int(any(map(line.endswith, ['&', '&\n'])))
                         for line in code]

        level = 0
        cont_padding = 0
        tabwidth = self._default_settings['tabwidth']
        new_code = []
        for i, line in enumerate(code):
            if line in('','\n'):
                new_code.append(line)
                continue
            level -= decrease[i]

            padding = " "*(level*tabwidth + cont_padding)

            line = "%s%s" % (padding, line)

            new_code.append(line)

            if continuation[i]:
                cont_padding = 2*tabwidth
            else:
                cont_padding = 0
            level += increase[i]

        return new_code


def fcode(expr, parser, assign_to=None, **settings):
    """Converts an expr to a string of Fortran code

    expr : Expr
        A pyccel expression to be converted.
    parser : Parser
        The parser used to collect the expression
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    """

    return FCodePrinter(parser, **settings).doprint(expr, assign_to)

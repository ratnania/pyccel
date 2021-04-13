# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201
# pylint: disable=missing-function-docstring
import functools
import operator

from pyccel.ast.builtins  import PythonRange, PythonComplex, PythonEnumerate
from pyccel.ast.builtins  import PythonZip, PythonMap, PythonLen, PythonPrint
from pyccel.ast.builtins  import PythonList, PythonTuple

from pyccel.ast.core      import Declare, For, CodeBlock
from pyccel.ast.core      import FuncAddressDeclare, FunctionCall, FunctionDef
from pyccel.ast.core      import Deallocate
from pyccel.ast.core      import FunctionAddress
from pyccel.ast.core      import Assign, datatype, Import, AugAssign
from pyccel.ast.core      import SeparatorComment
from pyccel.ast.core      import create_incremented_string

from pyccel.ast.operators import PyccelAdd, PyccelMul, PyccelMinus, PyccelLt, PyccelGt
from pyccel.ast.operators import PyccelAssociativeParenthesis, PyccelMod
from pyccel.ast.operators import PyccelUnarySub, IfTernaryOperator

from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeComplex
from pyccel.ast.datatypes import NativeReal, NativeTuple, NativeString

from pyccel.ast.internals import Slice

from pyccel.ast.literals  import LiteralTrue, LiteralImaginaryUnit, LiteralFloat
from pyccel.ast.literals  import LiteralString, LiteralInteger, Literal
from pyccel.ast.literals  import Nil

from pyccel.ast.numpyext import NumpyFull, NumpyArray, NumpyArange
from pyccel.ast.numpyext import NumpyReal, NumpyImag, NumpyFloat

from pyccel.ast.utilities import expand_to_loops

from pyccel.ast.variable import ValuedVariable
from pyccel.ast.variable import PyccelArraySize, Variable, VariableAddress
from pyccel.ast.variable import DottedName

from pyccel.ast.sympy_helper import pyccel_to_sympy


from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors   import Errors
from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, INCOMPATIBLE_TYPEVAR_TO_FUNC,
                                    PYCCEL_RESTRICTION_IS_ISNOT )


errors = Errors()

# TODO: add examples

__all__ = ["CCodePrinter", "ccode"]

# dictionary mapping numpy function to (argument_conditions, C_function).
# Used in CCodePrinter._print_NumpyUfuncBase(self, expr)
numpy_ufunc_to_c_real = {
    'NumpyAbs'  : 'fabs',
    'NumpyFabs'  : 'fabs',
    'NumpyMin'  : 'minval',
    'NumpyMax'  : 'maxval',
    'NumpyFloor': 'floor',  # TODO: might require special treatment with casting
    # ---
    'NumpyExp' : 'exp',
    'NumpyLog' : 'log',
    'NumpySqrt': 'sqrt',
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

numpy_ufunc_to_c_complex = {
    'NumpyAbs'  : 'cabs',
    'NumpyMin'  : 'minval',
    'NumpyMax'  : 'maxval',
    # ---
    'NumpyExp' : 'cexp',
    'NumpyLog' : 'clog',
    'NumpySqrt': 'csqrt',
    # ---
    'NumpySin'    : 'csin',
    'NumpyCos'    : 'ccos',
    'NumpyTan'    : 'ctan',
    'NumpyArcsin' : 'casin',
    'NumpyArccos' : 'cacos',
    'NumpyArctan' : 'catan',
    'NumpySinh'   : 'csinh',
    'NumpyCosh'   : 'ccosh',
    'NumpyTanh'   : 'ctanh',
    'NumpyArcsinh': 'casinh',
    'NumpyArccosh': 'cacosh',
    'NumpyArctanh': 'catanh',
}

# dictionary mapping Math function to (argument_conditions, C_function).
# Used in CCodePrinter._print_MathFunctionBase(self, expr)
# Math function ref https://docs.python.org/3/library/math.html
math_function_to_c = {
    # ---------- Number-theoretic and representation functions ------------
    'MathCeil'     : 'ceil',
    # 'MathComb'   : 'com' # TODO
    'MathCopysign': 'copysign',
    'MathFabs'   : 'fabs',
    'MathFloor'    : 'floor',
    # 'MathFmod'   : '???',  # TODO
    # 'MathRexp'   : '???'   TODO requires two output
    # 'MathFsum'   : '???',  # TODO
    # 'MathIsclose' : '???',  # TODO
    'MathIsfinite': 'isfinite', # int isfinite(real-floating x);
    'MathIsinf'   : 'isinf', # int isinf(real-floating x);
    'MathIsnan'   : 'isnan', # int isnan(real-floating x);
    # 'MathIsqrt'  : '???' TODO
    'MathLdexp'  : 'ldexp',
    # 'MathModf'  : '???' TODO return two value
    # 'MathPerm'  : '???' TODO
    # 'MathProd'  : '???' TODO
    'MathRemainder'  : 'remainder',
    'MathTrunc'  : 'trunc',

    # ----------------- Power and logarithmic functions -----------------------

    'MathExp'    : 'exp',
    'MathExpm1'  : 'expm1',
    'MathLog'    : 'log',      # take also an option arg [base]
    'MathLog1p'  : 'log1p',
    'MathLog2'  : 'log2',
    'MathLog10'  : 'log10',
    'MathPow'    : 'pow',
    'MathSqrt'   : 'sqrt',

    # --------------------- Trigonometric functions ---------------------------

    'MathAcos'   : 'acos',
    'MathAsin'   : 'asin',
    'MathAtan'   : 'atan',
    'MathAtan2'  : 'atan2',
    'MathCos'    : 'cos',
    # 'MathDist'  : '???', TODO
    'MathHypot'  : 'hypot',
    'MathSin'    : 'sin',
    'MathTan'    : 'tan',


    # -------------------------- Hyperbolic functions -------------------------

    'MathAcosh'  : 'acosh',
    'MathAsinh'  : 'asinh',
    'MathAtanh'  : 'atanh',
    'MathCosh'   : 'cosh',
    'MathSinh'   : 'sinh',
    'MathTanh'   : 'tanh',

    # --------------------------- Special functions ---------------------------

    'MathErf'    : 'erf',
    'MathErfc'   : 'erfc',
    'MathGamma'  : 'tgamma',
    'MathLgamma' : 'lgamma',

    # --------------------------- internal functions --------------------------
    'MathFactorial' : 'pyc_factorial',
    'MathGcd'       : 'pyc_gcd',
    'MathDegrees'   : 'pyc_degrees',
    'MathRadians'   : 'pyc_radians',
    'MathLcm'       : 'pyc_lcm',
}

c_library_headers = (
    "complex",
    "ctype",
    "float",
    "math",
    "stdarg",
    "stdbool",
    "stddef",
    "stdint",
    "stdio",
    "stdlib",
    "tgmath",
)

dtype_registry = {('real',8)    : 'double',
                  ('real',4)    : 'float',
                  ('complex',8) : 'double complex',
                  ('complex',4) : 'float complex',
                  ('int',4)     : 'int32_t',
                  ('int',8)     : 'int64_t',
                  ('int',2)     : 'int16_t',
                  ('int',1)     : 'int8_t',
                  ('bool',4)    : 'bool'}

ndarray_type_registry = {('real',8)    : 'nd_double',
                  ('real',4)    : 'nd_float',
                  ('complex',8) : 'nd_cdouble',
                  ('complex',4) : 'nd_cfloat',
                  ('int',8)     : 'nd_int64',
                  ('int',4)     : 'nd_int32',
                  ('int',2)     : 'nd_int16',
                  ('int',1)     : 'nd_int8',
                  ('bool',4)    : 'nd_bool'}

import_dict = {'omp_lib' : 'omp' }

class CCodePrinter(CodePrinter):
    """A printer to convert python expressions to strings of c code"""
    printmethod = "_ccode"
    language = "C"

    _default_settings = {
        'tabwidth': 4,
    }

    def __init__(self, parser, prefix_module = None):

        if parser.filename:
            errors.set_target(parser.filename, 'file')

        super().__init__()
        self.prefix_module = prefix_module
        self._additional_imports = set(['stdlib'])
        self._parser = parser
        self._additional_code = ''
        self._additional_declare = []
        self._additional_args = []
        self._temporary_args = []

    def get_additional_imports(self):
        """return the additional imports collected in printing stage"""
        return self._additional_imports

    def _get_statement(self, codestring):
        return "%s;\n" % codestring

    def _get_comment(self, text):
        return "// {0}\n".format(text)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _flatten_list(self, irregular_list):
        if isinstance(irregular_list, (PythonList, PythonTuple)):
            f_list = [element for item in irregular_list for element in self._flatten_list(item)]
            return f_list
        else:
            return [irregular_list]

    #========================== Numpy Elements ===============================#
    def copy_NumpyArray_Data(self, expr):
        """ print the assignment of a NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains the declaration of a dummy data_buffer
                       and a call to an operator which copies it to an NdArray struct
                if the ndarray is a stack_array the str will contain the initialization
        """
        rhs = expr.rhs
        lhs = expr.lhs
        if rhs.rank == 0:
            raise NotImplementedError(str(expr))
        dummy_array_name, _ = create_incremented_string(self._parser.used_names, prefix = 'array_dummy')
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
        dtype = self.find_in_ndarray_type_registry(self._print(rhs.dtype), rhs.precision)
        arg = rhs.arg
        if rhs.rank > 1:
            # flattening the args to use them in C initialization.
            arg = self._flatten_list(arg)

        if isinstance(arg, Variable):
            arg = self._print(arg)
            if expr.lhs.is_stack_array:
                cpy_data = self._init_stack_array(expr, rhs.arg)
            else:
                cpy_data = "memcpy({0}.{2}, {1}.{2}, {0}.buffer_size);\n".format(lhs, arg, dtype)
            return '%s' % (cpy_data)
        else :
            arg = ', '.join(self._print(i) for i in arg)
            dummy_array = "%s %s[] = {%s};\n" % (declare_dtype, dummy_array_name, arg)
            if expr.lhs.is_stack_array:
                cpy_data = self._init_stack_array(expr, dummy_array_name)
            else:
                cpy_data = "memcpy({0}.{2}, {1}, {0}.buffer_size);\n".format(self._print(lhs), dummy_array_name, dtype)
            return  '%s%s' % (dummy_array, cpy_data)

    def arrayFill(self, expr):
        """ print the assignment of a NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains a call to the C function array_fill,
                if the ndarray is a stack_array the str will contain the initialization
        """
        rhs = expr.rhs
        lhs = expr.lhs
        code_init = ''
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)

        if lhs.is_stack_array:
            symbol_map = {}
            used_names_tmp = self._parser.used_names.copy()
            sympy_shapes = [pyccel_to_sympy(s, symbol_map, used_names_tmp) for s in lhs.alloc_shape]

            length = functools.reduce(operator.mul, sympy_shapes)
            printable_length = functools.reduce(PyccelMul, lhs.alloc_shape)

            length_code = self._print(printable_length)

            if length.is_constant():
                buffer_array = "({declare_dtype}[{length}]){{}}".format(
                                        declare_dtype = declare_dtype,
                                        length=length_code)
            else:
                dummy_array_name, _ = create_incremented_string(self._parser.used_names,
                                                                prefix = lhs.name+'_data')
                code_init += "{dtype} {name}[{length}];\n".format(
                        dtype  = declare_dtype,
                        name   = dummy_array_name,
                        length = length_code)
                buffer_array = dummy_array_name

            code_init += self._init_stack_array(expr, buffer_array)

        if rhs.fill_value is not None:
            if isinstance(rhs.fill_value, Literal):
                code_init += 'array_fill(({0}){1}, {2});\n'.format(declare_dtype, self._print(rhs.fill_value), self._print(lhs))
            else:
                code_init += 'array_fill({0}, {1});\n'.format(self._print(rhs.fill_value), self._print(lhs))
        return code_init

    def _init_stack_array(self, expr, buffer_array):
        """ return a string which handles the assignment of a stack ndarray

        Parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
            buffer_array : String
                The data buffer
        Returns
        -------
            Returns a string that contains the initialization of a stack_array
        """

        lhs = expr.lhs
        rhs = expr.rhs
        dtype = self.find_in_ndarray_type_registry(self._print(rhs.dtype), rhs.precision)
        shape = ", ".join(self._print(i) for i in lhs.alloc_shape)
        declare_dtype = self.find_in_dtype_registry('int', 8)

        shape_init = "({declare_dtype}[]){{{shape}}}".format(declare_dtype=declare_dtype, shape=shape)
        strides_init = "({declare_dtype}[{length}]){{0}}".format(declare_dtype=declare_dtype, length=len(lhs.shape))
        if isinstance(buffer_array, Variable):
            buffer_array = "{0}.{1}".format(self._print(buffer_array), dtype)
        cpy_data = '{0} = (t_ndarray){{\n.{1}={2},\n .shape={3},\n .strides={4},\n '
        cpy_data += '.nd={5},\n .type={1},\n .is_view={6}\n}};\n'
        cpy_data = cpy_data.format(self._print(lhs), dtype, buffer_array,
                    shape_init, strides_init, len(lhs.shape), 'false')
        cpy_data += 'stack_array_init(&{});\n'.format(self._print(lhs))
        self._additional_imports.add("ndarrays")
        return cpy_data

    def fill_NumpyArange(self, expr, lhs):
        """ print the assignment of a NumpyArange
        parameters
        ----------
            expr : NumpyArange
                The node holding NumpyArange
            lhs : Variable
                 The left hand of Assign
        Return
        ------
            String
                Return string that contains the Assign code and the For loop
                responsible for filling the array values
        """
        start  = self._print(expr.start)
        stop   = self._print(expr.stop)
        step   = self._print(expr.step)
        dtype  = self.find_in_ndarray_type_registry(self._print(expr.dtype), expr.precision)

        target = Variable(expr.dtype, name =  self._parser.get_new_name('s'))
        index  = Variable(NativeInteger(), name = self._parser.get_new_name('i'))

        self._additional_declare += [index, target]
        self._additional_code += self._print(Assign(index, LiteralInteger(0))) + '\n'

        code = 'for({target} = {start}; {target} {op} {stop}; {target} += {step})'
        code += '\n{{\n{lhs}.{dtype}[{index}] = {target};\n'
        code += self._print(AugAssign(index, '+', LiteralInteger(1))) + '\n}}'
        code = code.format(target = self._print(target),
                            start = start,
                            stop  = stop,
                            op    = '<' if not isinstance(expr.step, PyccelUnarySub) else '>',
                            step  = step,
                            index = self._print(index),
                            lhs   = lhs,
                            dtype = dtype)
        return code

    # ============ Elements ============ #

    def _print_PythonAbs(self, expr):
        if expr.arg.dtype is NativeReal():
            self._additional_imports.add("math")
            func = "fabs"
        elif expr.arg.dtype is NativeComplex():
            self._additional_imports.add("complex")
            func = "cabs"
        else:
            func = "labs"
        return "{}({})".format(func, self._print(expr.arg))

    def _print_PythonMin(self, expr):
        arg = expr.args[0]
        if arg.dtype is NativeReal() and len(arg) == 2:
            self._additional_imports.add("math")
            return "fmin({}, {})".format(self._print(arg[0]),
                                         self._print(arg[1]))
        else:
            return errors.report("min in C is only supported for 2 float arguments", symbol=expr,
                    severity='fatal')

    def _print_PythonMax(self, expr):
        arg = expr.args[0]
        if arg.dtype is NativeReal() and len(arg) == 2:
            self._additional_imports.add("math")
            return "fmax({}, {})".format(self._print(arg[0]),
                                         self._print(arg[1]))
        else:
            return errors.report("max in C is only supported for 2 float arguments", symbol=expr,
                    severity='fatal')

    def _print_PythonFloat(self, expr):
        value = self._print(expr.arg)
        type_name = self.find_in_dtype_registry('real', expr.precision)
        return '({0})({1})'.format(type_name, value)

    def _print_PythonInt(self, expr):
        self._additional_imports.add('stdint')
        value = self._print(expr.arg)
        type_name = self.find_in_dtype_registry('int', expr.precision)
        return '({0})({1})'.format(type_name, value)

    def _print_PythonBool(self, expr):
        value = self._print(expr.arg)
        return '({} != 0)'.format(value)

    def _print_Literal(self, expr):
        return repr(expr.python_value)

    def _print_LiteralComplex(self, expr):
        if expr.real == LiteralFloat(0):
            return self._print(PyccelAssociativeParenthesis(PyccelMul(expr.imag, LiteralImaginaryUnit())))
        else:
            return self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real,
                            PyccelMul(expr.imag, LiteralImaginaryUnit()))))

    def _print_PythonComplex(self, expr):
        if expr.is_cast:
            value = self._print(expr.internal_var)
        else:
            value = self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real,
                            PyccelMul(expr.imag, LiteralImaginaryUnit()))))
        type_name = self.find_in_dtype_registry('complex', expr.precision)
        return '({0})({1})'.format(type_name, value)

    def _print_LiteralImaginaryUnit(self, expr):
        self._additional_imports.add("complex")
        return '_Complex_I'

    def _print_PythonLen(self, expr):
        var = expr.arg
        if var.rank > 0:
            return self._print(var.shape[0])
        else:
            return errors.report("PythonLen not implemented for type {}\n".format(type(expr.arg)) +
                    PYCCEL_RESTRICTION_TODO,
                    symbol = expr, severity='fatal')

    def _print_ModuleHeader(self, expr):
        name = expr.module.name
        # TODO: Add classes and interfaces
        funcs = '\n'.join('{};'.format(self.function_signature(f)) for f in expr.module.funcs)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *map(Import, self._additional_imports)]
        imports = ''.join(self._print(i) for i in imports)

        return ('#ifndef {name}_H\n'
                '#define {name}_H\n\n'
                '{imports}\n'
                #'{classes}\n'
                '{funcs}\n'
                #'{interfaces}\n'
                '#endif // {name}_H\n').format(
                        name    = name.upper(),
                        imports = imports,
                        funcs   = funcs)

    def _print_Module(self, expr):
        body    = ''.join(self._print(i) for i in expr.body)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [Import(expr.name), *map(Import, self._additional_imports)]
        imports = ''.join(self._print(i) for i in imports)
        return ('{imports}\n'
                '{body}\n').format(
                        imports = imports,
                        body    = body)

    def _print_Break(self, expr):
        return 'break;\n'

    def _print_Continue(self, expr):
        return 'continue;\n'

    def _print_While(self, expr):
        body = self._print(expr.body)
        cond = self._print(expr.test)
        return 'while({condi})\n{{\n{body}}}\n'.format(condi = cond, body = body)

    def _print_If(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.blocks):
            var = self._print(e)
            if i == 0:
                lines.append("if (%s)\n{\n" % self._print(c))
            elif i == len(expr.blocks) - 1 and isinstance(c, LiteralTrue):
                lines.append("else\n{\n")
            else:
                lines.append("else if (%s)\n{\n" % self._print(c))
            lines.append("%s}\n" % var)
        return "".join(lines)

    def _print_IfTernaryOperator(self, expr):
        cond = self._print(expr.cond)
        value_true = self._print(expr.value_true)
        value_false = self._print(expr.value_false)
        return '{cond} ? {true} : {false}'.format(cond = cond, true =value_true, false = value_false)

    def _print_LiteralTrue(self, expr):
        return '1'

    def _print_LiteralFalse(self, expr):
        return '0'

    def _print_PyccelAnd(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' && '.join(a for a in args)

    def _print_PyccelOr(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' || '.join(a for a in args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} == {1}'.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} != {1}'.format(lhs, rhs)

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
        return '!{}'.format(a)

    def _print_PyccelMod(self, expr):
        self._additional_imports.add("math")

        first = self._print(expr.args[0])
        second = self._print(expr.args[1])

        if expr.dtype is NativeInteger():
            return "{} % {}".format(first, second)

        if expr.args[0].dtype is NativeInteger():
            first = self._print(NumpyFloat(expr.args[0]))
        if expr.args[1].dtype is NativeInteger():
            second = self._print(NumpyFloat(expr.args[1]))
        return "fmod({}, {})".format(first, second)

    def _print_PyccelPow(self, expr):
        b = expr.args[0]
        e = expr.args[1]

        if expr.dtype is NativeComplex():
            b = self._print(b if b.dtype is NativeComplex() else PythonComplex(b))
            e = self._print(e if e.dtype is NativeComplex() else PythonComplex(e))
            self._additional_imports.add("complex")
            return 'cpow({}, {})'.format(b, e)

        self._additional_imports.add("math")
        b = self._print(b if b.dtype is NativeReal() else NumpyFloat(b))
        e = self._print(e if e.dtype is NativeReal() else NumpyFloat(e))
        code = 'pow({}, {})'.format(b, e)
        if expr.dtype is NativeInteger():
            dtype = self._print(expr.dtype)
            prec  = expr.precision
            cast_type = self.find_in_dtype_registry(dtype, prec)
            return '({}){}'.format(cast_type, code)
        return code

    def _print_Import(self, expr):
        if expr.ignore:
            return ''
        if isinstance(expr.source, DottedName):
            source = expr.source.name[-1]
        else:
            source = self._print(expr.source)

        # Get with a default value is not used here as it is
        # slower and on most occasions the import will not be in the
        # dictionary
        if source in import_dict: # pylint: disable=consider-using-get
            source = import_dict[source]

        if source is None:
            return ''
        if expr.source in c_library_headers:
            return '#include <{0}.h>\n'.format(source)
        else:
            return '#include "{0}.h"\n'.format(source)

    def _print_LiteralString(self, expr):
        format_str = format(expr.arg)
        format_str = format_str.replace("\\", "\\\\")\
                               .replace('\a', '\\a')\
                               .replace('\b', '\\b')\
                               .replace('\f', '\\f')\
                               .replace("\n", "\\n")\
                               .replace('\r', '\\r')\
                               .replace('\t', '\\t')\
                               .replace('\v', '\\v')\
                               .replace('"', '\\"')\
                               .replace("'", "\\'")
        return '"{}"'.format(format_str)

    def get_print_format_and_arg(self, var):
        type_to_format = {('real',8)    : '%.12lf',
                          ('real',4)    : '%.12f',
                          ('complex',8) : '(%.12lf + %.12lfj)',
                          ('complex',4) : '(%.12f + %.12fj)',
                          ('int',4)     : '%d',
                          ('int',8)     : '%ld',
                          ('int',2)     : '%hd',
                          ('int',1)     : '%c',
                          ('bool',4)    : '%s',
                          ('string', 0) : '%s'}
        try:
            arg_format = type_to_format[(self._print(var.dtype), var.precision)]
        except KeyError:
            errors.report("{} type is not supported currently".format(var.dtype), severity='fatal')
        if var.dtype is NativeComplex():
            arg = '{}, {}'.format(self._print(NumpyReal(var)), self._print(NumpyImag(var)))
        elif var.dtype is NativeBool():
            arg = '{} ? "True" : "False"'.format(self._print(var))
        else:
            arg = self._print(var)
        return arg_format, arg

    def extract_function_call_results(self, expr):
        tmp_list = [self.create_tmp_var(a) for a in expr.funcdef.results]
        return tmp_list

    def _print_PythonPrint(self, expr):
        self._additional_imports.add("stdio")
        end = '\n'
        sep = ' '
        code = ''
        empty_end = ValuedVariable(NativeString(), 'end', value='')
        space_end = ValuedVariable(NativeString(), 'end', value=' ')
        kwargs = [f for f in expr.expr if isinstance(f, ValuedVariable)]
        for f in kwargs:
            if isinstance(f, ValuedVariable):
                if f.name == 'sep'      :   sep = str(f.value)
                elif f.name == 'end'    :   end = str(f.value)
        args_format = []
        args = []
        orig_args = [f for f in expr.expr if not isinstance(f, ValuedVariable)]

        def formatted_args_to_printf(args_format, args, end):
            args_format = sep.join(args_format)
            args_format += end
            args_format = self._print(LiteralString(args_format))
            args_code = ', '.join([args_format, *args])
            return "printf({});\n".format(args_code)

        if len(orig_args) == 0:
            return formatted_args_to_printf(args_format, args, end)

        for i, f in enumerate(orig_args):
            if isinstance(f, FunctionCall) and isinstance(f.dtype, NativeTuple):
                tmp_list = self.extract_function_call_results(f)
                tmp_arg_format_list = []
                for a in tmp_list:
                    arg_format, arg = self.get_print_format_and_arg(a)
                    tmp_arg_format_list.append(arg_format)
                    args.append(arg)
                args_format.append('({})'.format(', '.join(tmp_arg_format_list)))
                assign = Assign(tmp_list, f)
                self._additional_code += self._print(assign)
            elif f.rank > 0:
                if args_format:
                    code += formatted_args_to_printf(args_format, args, sep)
                    args_format = []
                    args = []
                for_index = Variable(NativeInteger(), name = self._parser.get_new_name('i'))
                self._additional_declare.append(for_index)
                max_index = PyccelMinus(PythonLen(orig_args[i]), LiteralInteger(1), simplify = True)
                for_range = PythonRange(max_index)
                print_body = [ orig_args[i][for_index] ]
                if orig_args[i].rank == 1:
                    print_body.append(space_end)

                for_body  = [PythonPrint(print_body)]
                for_loop  = For(for_index, for_range, for_body)
                for_end   = ValuedVariable(NativeString(), 'end', value=']'+end if i == len(orig_args)-1 else ']')

                body = CodeBlock([PythonPrint([ LiteralString('['), empty_end]),
                                  for_loop,
                                  PythonPrint([ orig_args[i][max_index], for_end])],
                                 unravelled = True)
                code += self._print(body)
            else:
                arg_format, arg = self.get_print_format_and_arg(f)
                args_format.append(arg_format)
                args.append(arg)
        if args_format:
            code += formatted_args_to_printf(args_format, args, end)
        return code

    def find_in_dtype_registry(self, dtype, prec):
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO,
                    symbol = "{}[kind = {}]".format(dtype, prec),
                    severity='fatal')

    def find_in_ndarray_type_registry(self, dtype, prec):
        try :
            return ndarray_type_registry[(dtype, prec)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO,
                    symbol = "{}[kind = {}]".format(dtype, prec),
                    severity='fatal')

    def get_declare_type(self, expr):
        dtype = self._print(expr.dtype)
        prec  = expr.precision
        rank  = expr.rank
        if isinstance(expr.dtype, NativeInteger):
            self._additional_imports.add('stdint')
        dtype = self.find_in_dtype_registry(dtype, prec)
        if rank > 0:
            if expr.is_ndarray:
                self._additional_imports.add('ndarrays')
                return 't_ndarray '
            errors.report(PYCCEL_RESTRICTION_TODO, symbol="rank > 0",severity='fatal')

        if self.stored_in_c_pointer(expr):
            return '{0} *'.format(dtype)
        else:
            return '{0} '.format(dtype)

    def _print_FuncAddressDeclare(self, expr):
        args = list(expr.arguments)
        if len(expr.results) == 1:
            ret_type = self.get_declare_type(expr.results[0])
        elif len(expr.results) > 1:
            ret_type = self._print(datatype('int')) + ' '
            args += [a.clone(name = a.name, is_pointer =True) for a in expr.results]
        else:
            ret_type = self._print(datatype('void')) + ' '
        name = expr.name
        if not args:
            arg_code = 'void'
        else:
            # TODO: extract informations needed for printing in case of function argument which itself has a function argument
            arg_code = ', '.join('{}'.format(self._print_FuncAddressDeclare(i))
                        if isinstance(i, FunctionAddress) else '{0}{1}'.format(self.get_declare_type(i), i)
                        for i in args)
        return '{}(*{})({});\n'.format(ret_type, name, arg_code)

    def _print_Declare(self, expr):
        declaration_type = self.get_declare_type(expr.variable)
        variable = self._print(expr.variable.name)

        return '{0}{1};\n'.format(declaration_type, variable)

    def _print_NativeBool(self, expr):
        self._additional_imports.add('stdbool')
        return 'bool'

    def _print_NativeInteger(self, expr):
        return 'int'

    def _print_NativeReal(self, expr):
        return 'real'

    def _print_NativeVoid(self, expr):
        return 'void'

    def _print_NativeComplex(self, expr):
        self._additional_imports.add('complex')
        return 'complex'
    def _print_NativeString(self, expr):
        return 'string'

    def function_signature(self, expr, print_arg_names = True):
        """Extract from function definition all the information
        (name, input, output) needed to create the signature

        Parameters
        ----------
        expr            : FunctionDef
            the function defintion

        print_arg_names : Bool
            default value True and False when we don't need to print
            arguments names

        Return
        ------
        String
            Signature of the function
        """
        args = list(expr.arguments)
        if len(expr.results) == 1:
            ret_type = self.get_declare_type(expr.results[0])
        elif len(expr.results) > 1:
            ret_type = self._print(datatype('int')) + ' '
            args += [a.clone(name = a.name, is_pointer =True) for a in expr.results]
        else:
            ret_type = self._print(datatype('void')) + ' '
        name = expr.name
        if not args:
            arg_code = 'void'
        else:
            arg_code = ', '.join('{}'.format(self.function_signature(i, False))
                        if isinstance(i, FunctionAddress)
                        else '{0}'.format(self.get_declare_type(i)) + (i.name if print_arg_names else '')
                        for i in args)
        if isinstance(expr, FunctionAddress):
            return '{}(*{})({})'.format(ret_type, name, arg_code)
        else:
            return '{0}{1}({2})'.format(ret_type, name, arg_code)

    def _print_IndexedElement(self, expr):
        base = expr.base
        inds = list(expr.indices)
        base_shape = base.shape
        allow_negative_indexes = base.allows_negative_indexes
        for i, ind in enumerate(inds):
            if isinstance(ind, PyccelUnarySub) and isinstance(ind.args[0], LiteralInteger):
                inds[i] = PyccelMinus(base_shape[i], ind.args[0], simplify = True)
            else:
                #indices of indexedElement of len==1 shouldn't be a tuple
                if isinstance(ind, tuple) and len(ind) == 1:
                    inds[i].args = ind[0]
                if allow_negative_indexes and \
                        not isinstance(ind, LiteralInteger) and not isinstance(ind, Slice):
                    inds[i] = IfTernaryOperator(PyccelLt(ind, LiteralInteger(0)),
                        PyccelAdd(base_shape[i], ind, simplify = True), ind)
        #set dtype to the C struct types
        dtype = self._print(expr.dtype)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.precision)
        base_name = self._print(base.name)
        if base.is_ndarray:
            if expr.rank > 0:
                #managing the Slice input
                for i , ind in enumerate(inds):
                    if isinstance(ind, Slice):
                        inds[i] = self._new_slice_with_processed_arguments(ind, PyccelArraySize(base, i),
                            allow_negative_indexes)
                    else:
                        inds[i] = Slice(ind, PyccelAdd(ind, LiteralInteger(1), simplify = True), LiteralInteger(1))
                inds = [self._print(i) for i in inds]
                return "array_slicing(%s, %s, %s)" % (base_name, expr.rank, ", ".join(inds))
            inds = [self._cast_to(i, NativeInteger(), 8).format(self._print(i)) for i in inds]
        else:
            raise NotImplementedError(expr)
        return "%s.%s[get_index(%s, %s)]" % (base_name, dtype, base_name, ", ".join(inds))

    def _cast_to(self, expr, dtype, precision):
        """ add cast to an expression when needed
        parameters
        ----------
            expr      : PyccelAstNode
                the expression to be cast
            dtype     : Datatype
                base type of the cast
            precision : integer
                precision of the base type of the cast

        Return
        ------
            String
                Return format string that contains the desired cast type
        """
        if (expr.dtype != dtype or expr.precision != precision):
            cast=self.find_in_dtype_registry(self._print(dtype), precision)
            return '({}){{}}'.format(cast)
        return '{}'

    def _print_DottedVariable(self, expr):
        """convert dotted Variable to their C equivalent"""
        return '{}.{}'.format(self._print(expr.lhs), self._print(expr.name))

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
        start = LiteralInteger(0) if _slice.start is None else _slice.start
        stop = array_size if _slice.stop is None else _slice.stop

        # negative start and end in slice
        if isinstance(start, PyccelUnarySub) and isinstance(start.args[0], LiteralInteger):
            start = PyccelMinus(array_size, start.args[0], simplify = True)
        elif allow_negative_index and not isinstance(start, (LiteralInteger, PyccelArraySize)):
            start = IfTernaryOperator(PyccelLt(start, LiteralInteger(0)),
                            PyccelMinus(array_size, start, simplify = True), start)

        if isinstance(stop, PyccelUnarySub) and isinstance(stop.args[0], LiteralInteger):
            stop = PyccelMinus(array_size, stop.args[0], simplify = True)
        elif allow_negative_index and not isinstance(stop, (LiteralInteger, PyccelArraySize)):
            stop = IfTernaryOperator(PyccelLt(stop, LiteralInteger(0)),
                            PyccelMinus(array_size, stop, simplify = True), stop)

        # steps in slices
        step = _slice.step

        if step is None:
            step = LiteralInteger(1)

        # negative step in slice
        elif isinstance(step, PyccelUnarySub) and isinstance(step.args[0], LiteralInteger):
            start = PyccelMinus(array_size, LiteralInteger(1), simplify = True) if _slice.start is None else start
            stop = LiteralInteger(0) if _slice.stop is None else stop

        # variable step in slice
        elif allow_negative_index and step and not isinstance(step, LiteralInteger):
            og_start = start
            start = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)), start, PyccelMinus(stop, LiteralInteger(1), simplify = True))
            stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)), stop, og_start)

        return Slice(start, stop, step)

    def _print_PyccelArraySize(self, expr):
        return '{}.shape[{}]'.format(expr.arg, expr.index)

    def _print_Allocate(self, expr):
        free_code = ''
        #free the array if its already allocated and checking if its not null if the status is unknown
        if  (expr.status == 'unknown'):
            free_code = 'if (%s.shape != NULL)\n' % self._print(expr.variable.name)
            free_code += "{{\n{}}}\n".format(self._print(Deallocate(expr.variable)))
        elif  (expr.status == 'allocated'):
            free_code += self._print(Deallocate(expr.variable))
        self._additional_imports.add('ndarrays')
        shape = ", ".join(self._print(i) for i in expr.shape)
        dtype = self._print(expr.variable.dtype)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.variable.precision)
        shape_dtype = self.find_in_dtype_registry('int', 8)
        shape_Assign = "("+ shape_dtype +"[]){" + shape + "}"
        alloc_code = "{} = array_create({}, {}, {});\n".format(expr.variable, len(expr.shape), shape_Assign, dtype)
        return '{}{}'.format(free_code, alloc_code)

    def _print_Deallocate(self, expr):
        if expr.variable.is_pointer:
            return 'free_pointer({});\n'.format(self._print(expr.variable))
        return 'free_array({});\n'.format(self._print(expr.variable))

    def _print_Slice(self, expr):
        start = self._print(expr.start)
        stop = self._print(expr.stop)
        step = self._print(expr.step)
        return 'new_slice({}, {}, {})'.format(start, stop, step)

    def _print_NumpyUfuncBase(self, expr):
        """ Convert a Python expression with a Numpy function call to C
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Numpy function call

        Returns
        -------
            string
                Equivalent expression in C language

        Example
        -------
            numpy.cos(x) ==> cos(x)

        """
        # add necessary include
        self._additional_imports.add('math')
        type_name = type(expr).__name__
        try:
            func_name = numpy_ufunc_to_c_real[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
        args = []
        for arg in expr.args:
            if arg.dtype is NativeComplex():
                self._additional_imports.add('complex')
                try:
                    func_name = numpy_ufunc_to_c_complex[type_name]
                    args.append(self._print(arg))
                except KeyError:
                    errors.report(INCOMPATIBLE_TYPEVAR_TO_FUNC.format(type_name) ,severity='fatal')
            elif arg.dtype is not NativeReal():
                args.append(self._print(NumpyFloat(arg)))
            else :
                args.append(self._print(arg))
        code_args = ', '.join(args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_MathFunctionBase(self, expr):
        """ Convert a Python expression with a math function call to C
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Math function call

        Returns
        -------
            string
                Equivalent expression in C language

        ------
        Example:
        --------
            math.sin(x) ==> sin(x)

        """
        # add necessary include
        type_name = type(expr).__name__
        try:
            func_name = math_function_to_c[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')

        if func_name.startswith("pyc"):
            self._additional_imports.add('pyc_math')
        else:
            if expr.dtype is NativeComplex():
                self._additional_imports.add('cmath')
            else:
                self._additional_imports.add('math')
        args = []
        for arg in expr.args:
            if arg.dtype != NativeReal() and not func_name.startswith("pyc"):
                args.append(self._print(NumpyFloat(arg)))
            else:
                args.append(self._print(arg))
        code_args = ', '.join(args)
        if expr.dtype == NativeInteger():
            cast_type = self.find_in_dtype_registry('int', expr.precision)
            return '({0}){1}({2})'.format(cast_type, func_name, code_args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_MathIsfinite(self, expr):
        """Convert a Python expression with a math isfinite function call to C
        function call"""
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isfinite({})".format(code_arg)

    def _print_MathIsinf(self, expr):
        """Convert a Python expression with a math isinf function call to C
        function call"""
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isinf({})".format(code_arg)

    def _print_MathIsnan(self, expr):
        """Convert a Python expression with a math isnan function call to C
        function call"""
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isnan({})".format(code_arg)

    def _print_MathTrunc(self, expr):
        """Convert a Python expression with a math trunc function call to C
        function call"""
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "trunc({})".format(code_arg)

    def _print_FunctionAddress(self, expr):
        return expr.name

    def _print_Rand(self, expr):
        raise NotImplementedError("Rand not implemented")

    def _print_NumpyRandint(self, expr):
        raise NotImplementedError("Randint not implemented")

    def _print_NumpyMod(self, expr):
        return self._print(PyccelMod(*expr.args))

    def _print_Interface(self, expr):
        return ""

    def _print_FunctionDef(self, expr):

        if len(expr.results) > 1:
            self._additional_args.append(expr.results)
        body  = self._print(expr.body)
        decs  = [Declare(i.dtype, i) if isinstance(i, Variable) else FuncAddressDeclare(i) for i in expr.local_vars]
        if len(expr.results) <= 1 :
            for i in expr.results:
                if isinstance(i, Variable) and not i.is_temp:
                    decs += [Declare(i.dtype, i)]
                elif not isinstance(i, Variable):
                    decs += [FuncAddressDeclare(i)]
        decs += [Declare(i.dtype, i) for i in self._additional_declare]
        decs  = ''.join(self._print(i) for i in decs)
        self._additional_declare.clear()

        sep = self._print(SeparatorComment(40))
        if self._additional_args :
            self._additional_args.pop()
        imports = ''.join(self._print(i) for i in expr.imports)
        doc_string = self._print(expr.doc_string) if expr.doc_string else ''

        parts = [sep,
                 doc_string,
                '{signature}\n{{\n'.format(signature=self.function_signature(expr)),
                 imports,
                 decs,
                 body,
                 '}\n',
                 sep]

        return ''.join(p for p in parts if p)

    def stored_in_c_pointer(self, a):
        if not isinstance(a, Variable):
            return False
        return (a.is_pointer and not a.is_ndarray) or a.is_optional or any(a in b for b in self._additional_args)

    def create_tmp_var(self, match_var):
        tmp_var_name = self._parser.get_new_name('tmp')
        tmp_var = Variable(name = tmp_var_name, dtype = match_var.dtype)
        self._additional_declare.append(tmp_var)
        return tmp_var

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
         # Ensure the correct syntax is used for pointers
        args = []
        for a, f in zip(expr.args, func.arguments):
            if isinstance(a, Variable) and self.stored_in_c_pointer(f):
                args.append(VariableAddress(a))
            elif f.is_optional and not isinstance(a, Nil):
                tmp_var = self.create_tmp_var(f)
                assign = Assign(tmp_var, a)
                self._additional_code += self._print(assign) + '\n'
                args.append(VariableAddress(tmp_var))

            else :
                args.append(a)

        args += self._temporary_args
        self._temporary_args = []
        args = ', '.join(['{}'.format(self._print(a)) for a in args])
        if not func.results:
            return '{}({});\n'.format(func.name, args)
        return '{}({})'.format(func.name, args)

    def _print_Constant(self, expr):
        """ Convert a Python expression with a math constant call to C
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Math constant

        Returns
        -------
            string
                String represent the value of the constant

        Example
        -------
            math.pi ==> 3.14159265358979

        """
        val = LiteralFloat(expr.value)
        return self._print(val)

    def _print_Return(self, expr):
        code = ''
        args = [VariableAddress(a) if self.stored_in_c_pointer(a) else a for a in expr.expr]

        if len(args) > 1:
            if expr.stmt:
                return self._print(expr.stmt)+'\n'+'return 0;\n'
            return 'return 0;\n'

        if expr.stmt:
            # get Assign nodes from the CodeBlock object expr.stmt.
            last_assign = expr.stmt.get_attribute_nodes(Assign, excluded_nodes=FunctionCall)
            deallocate_nodes = expr.stmt.get_attribute_nodes(Deallocate, excluded_nodes=(Assign,))
            vars_in_deallocate_nodes = [i.variable for i in deallocate_nodes]

            # Check the Assign objects list in case of
            # the user assigns a variable to an object contains IndexedElement object.
            if not last_assign:
                return 'return {0};\n'.format(self._print(args[0]))

            # make sure that stmt contains one assign node.
            assert(len(last_assign)==1)
            variables = last_assign[0].rhs.get_attribute_nodes(Variable, excluded_nodes=(FunctionDef,))
            unneeded_var = not any(b in vars_in_deallocate_nodes for b in variables)
            if unneeded_var:
                code = ''.join(self._print(a) for a in expr.stmt.body if a is not last_assign[0])
                return code + '\nreturn {};\n'.format(self._print(last_assign[0].rhs))
            else:
                code = ''+self._print(expr.stmt)
                self._additional_declare.append(last_assign[0].lhs)
        return code + 'return {0};\n'.format(self._print(args[0]))

    def _print_Pass(self, expr):
        return '// pass\n'

    def _print_Nil(self, expr):
        return 'NULL'

    def _print_PyccelAdd(self, expr):
        return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        args = [self._print(a) for a in expr.args]
        if len(args) == 1:
            return '-{}'.format(args[0])
        return ' - '.join(args)

    def _print_PyccelMul(self, expr):
        return ' * '.join(self._print(a) for a in expr.args)

    def _print_PyccelDiv(self, expr):
        if all(a.dtype is NativeInteger() for a in expr.args):
            args = [NumpyFloat(a) for a in expr.args]
        else:
            args = expr.args
        return  ' / '.join(self._print(a) for a in args)

    def _print_PyccelFloorDiv(self, expr):
        self._additional_imports.add("math")
        # the result type of the floor division is dependent on the arguments
        # type, if all arguments are integers the result is integer otherwise
        # the result type is float
        need_to_cast = all(a.dtype is NativeInteger() for a in expr.args)
        code = ' / '.join(self._print(a if a.dtype is NativeReal() else NumpyFloat(a)) for a in expr.args)
        if (need_to_cast):
            cast_type = self.find_in_dtype_registry('int', expr.precision)
            return "({})floor({})".format(cast_type, code)
        return "floor({})".format(code)

    def _print_PyccelRShift(self, expr):
        return ' >> '.join(self._print(a) for a in expr.args)

    def _print_PyccelLShift(self, expr):
        return ' << '.join(self._print(a) for a in expr.args)

    def _print_PyccelBitXor(self, expr):
        if expr.dtype is NativeBool():
            return '{0} != {1}'.format(self._print(expr.args[0]), self._print(expr.args[1]))
        return ' ^ '.join(self._print(a) for a in expr.args)

    def _print_PyccelBitOr(self, expr):
        if expr.dtype is NativeBool():
            return ' || '.join(self._print(a) for a in expr.args)
        return ' | '.join(self._print(a) for a in expr.args)

    def _print_PyccelBitAnd(self, expr):
        if expr.dtype is NativeBool():
            return ' && '.join(self._print(a) for a in expr.args)
        return ' & '.join(self._print(a) for a in expr.args)

    def _print_PyccelInvert(self, expr):
        return '~{}'.format(self._print(expr.args[0]))

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '+{}'.format(self._print(expr.args[0]))

    def _print_PyccelUnarySub(self, expr):
        return '-{}'.format(self._print(expr.args[0]))

    def _print_AugAssign(self, expr):
        lhs_code = self._print(expr.lhs)
        op = expr.op
        rhs_code = self._print(expr.rhs)
        return "{0} {1}= {2};\n".format(lhs_code, op, rhs_code)

    def _print_Assign(self, expr):
        if isinstance(expr.rhs, FunctionCall) and isinstance(expr.rhs.dtype, NativeTuple):
            self._temporary_args = [VariableAddress(a) for a in expr.lhs]
            return '{};\n'.format(self._print(expr.rhs))
        if isinstance(expr.rhs, (NumpyArray)):
            return self.copy_NumpyArray_Data(expr)
        if isinstance(expr.rhs, (NumpyFull)):
            return self.arrayFill(expr)
        if isinstance(expr.rhs, NumpyArange):
            return self.fill_NumpyArange(expr.rhs, expr.lhs)
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return '{} = {};\n'.format(lhs, rhs)

    def _print_AliasAssign(self, expr):
        lhs_var = expr.lhs
        rhs_var = expr.rhs

        lhs = VariableAddress(lhs_var)
        rhs = VariableAddress(rhs_var) if isinstance(rhs_var, Variable) else rhs_var

        lhs = self._print(lhs)
        rhs = self._print(rhs)

        # the below condition handles the case of reassinging a pointer to an array view.
        # setting the pointer's is_view attribute to false so it can be ignored by the free_pointer function.
        if isinstance(lhs_var, Variable) and lhs_var.is_ndarray \
                and isinstance(rhs_var, Variable) and rhs_var.is_ndarray:
            return 'alias_assign(&{}, {});\n'.format(lhs, rhs)

        return '{} = {};\n'.format(lhs, rhs)

    def _print_For(self, expr):
        counter = self._print(expr.target)
        body  = self._print(expr.body)
        if isinstance(expr.iterable, PythonRange):
            iterable = expr.iterable
        elif isinstance(expr.iterable, PythonEnumerate):
            iterable = PythonRange(PythonLen(expr.iterable.element))
        elif isinstance(expr.iterable, PythonZip):
            iterable = PythonRange(expr.iterable.length)
        elif isinstance(expr.iterable, PythonMap):
            iterable = PythonRange(PythonLen(expr.iterable.args[1]))
        else:
            raise NotImplementedError("Only iterables currently supported are Range, Enumerate, Zip and Map")
        start = self._print(iterable.start)
        stop  = self._print(iterable.stop )
        step  = self._print(iterable.step )

        test_step = iterable.step
        if isinstance(test_step, PyccelUnarySub):
            test_step = iterable.step.args[0]

        # testing if the step is a value or an expression
        if isinstance(test_step, Literal):
            op = '>' if isinstance(iterable.step, PyccelUnarySub) else '<'
            return ('for ({counter} = {start}; {counter} {op} {stop}; {counter} += '
                        '{step})\n{{\n{body}}}\n').format(counter=counter, start=start, op=op,
                                                          stop=stop, step=step, body=body)
        else:
            return (
                'for ({counter} = {start}; ({step} > 0) ? ({counter} < {stop}) : ({counter} > {stop}); {counter} += '
                '{step})\n{{\n{body}}}\n').format(counter=counter, start=start,
                                                  stop=stop, step=step, body=body)

    def _print_FunctionalFor(self, expr):
        loops = ''.join(self._print(i) for i in expr.loops)
        return loops

    def _print_CodeBlock(self, expr):
        if not expr.unravelled:
            body_exprs, new_vars = expand_to_loops(expr, self._parser.get_new_variable, language_has_vectors = False)
            self._additional_declare.extend(new_vars)
        else:
            body_exprs = expr.body
        body_stmts = []
        for b in body_exprs :
            code = self._print(b)
            code = self._additional_code + code
            self._additional_code = ''
            body_stmts.append(code)
        return ''.join(self._print(b) for b in body_stmts)

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Exp1(self, expr):
        return "M_E"

    def _print_Pi(self, expr):
        return 'M_PI'

    def _print_Infinity(self, expr):
        return 'HUGE_VAL'

    def _print_NegativeInfinity(self, expr):
        return '-HUGE_VAL'

    def _print_PythonReal(self, expr):
        return 'creal({})'.format(self._print(expr.internal_var))

    def _print_PythonImag(self, expr):
        return 'cimag({})'.format(self._print(expr.internal_var))

    def _handle_is_operator(self, Op, expr):

        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        a = expr.args[0]
        b = expr.args[1]

        if Nil() in expr.args:
            lhs = VariableAddress(expr.lhs) if isinstance(expr.lhs, Variable) else expr.lhs
            rhs = VariableAddress(expr.rhs) if isinstance(expr.rhs, Variable) else expr.rhs

            lhs = self._print(lhs)
            rhs = self._print(rhs)
            return '{} {} {}'.format(lhs, Op, rhs)

        if (a.dtype is NativeBool() and b.dtype is NativeBool()):
            return '{} {} {}'.format(lhs, Op, rhs)
        else:
            errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                          symbol=expr, severity='fatal')

    def _print_PyccelIsNot(self, expr):
        return self._handle_is_operator("!=", expr)

    def _print_PyccelIs(self, expr):
        return self._handle_is_operator("==", expr)

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond is not True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assign):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) {\n" % self._print(c))
                elif i == len(expr.args) - 1 and c is True:
                    lines.append("else {\n")
                else:
                    lines.append("else if (%s) {\n" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}\n")
            return "".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c), self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    def _print_Variable(self, expr):
        if self.stored_in_c_pointer(expr):
            return '(*{0})'.format(expr.name)
        else:
            return expr.name

    def _print_VariableAddress(self, expr):
        if self.stored_in_c_pointer(expr.variable) or expr.variable.rank > 0:
            return '{}'.format(expr.variable.name)
        else:
            return '&{}'.format(expr.variable.name)

    def _print_Comment(self, expr):
        comments = self._print(expr.text)

        return '/*' + comments + '*/\n'

    def _print_PyccelSymbol(self, expr):
        return expr

    def _print_CommentBlock(self, expr):
        txts = expr.comments
        header = expr.header
        header_size = len(expr.header)

        ln = max(len(i) for i in txts)
        if ln<max(20, header_size+4):
            ln = 20
        top  = '/*' + '_'*int((ln-header_size)/2) + header + '_'*int((ln-header_size)/2) + '*/\n'
        ln = len(top)-4
        bottom = '/*' + '_'*ln + '*/\n'

        txts = ['/*' + t + ' '*(ln - len(t)) + '*/\n' for t in txts]

        body = ''.join(i for i in txts)

        return ''.join([top, body, bottom])

    def _print_EmptyNode(self, expr):
        return ''

    #=================== OMP ==================

    def _print_OmpAnnotatedComment(self, expr):
        clauses = ''
        if expr.combined:
            clauses = ' ' + expr.combined
        clauses += str(expr.txt)
        if expr.has_nowait:
            clauses = clauses + ' nowait'
        omp_expr = '#pragma omp {}{}\n'.format(expr.name, clauses)

        if expr.is_multiline:
            if expr.combined is None:
                omp_expr += '{\n'
            elif (expr.combined and "for" not in expr.combined):
                if ("masked taskloop" not in expr.combined) and ("distribute" not in expr.combined):
                    omp_expr += '{\n'

        return omp_expr

    def _print_Omp_End_Clause(self, expr):
        return '}\n'
    #=====================================

    def _print_Program(self, expr):
        body  = self._print(expr.body)
        decs     = [self._print(i) for i in expr.declarations]
        decs    += [self._print(Declare(i.dtype, i)) for i in self._additional_declare]
        decs    = ''.join(self._print(i) for i in decs)
        self._additional_declare.clear()

        # PythonPrint imports last to be sure that all additional_imports have been collected
        imports  = [*expr.imports, *map(Import, self._additional_imports)]
        imports  = ''.join(self._print(i) for i in imports)
        return ('{imports}'
                'int main()\n{{\n'
                '{decs}'
                '{body}'
                'return 0;\n'
                '}}').format(imports=imports,
                                    decs=decs,
                                    body=body)



    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = " "*self._default_settings["tabwidth"]
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line == '' or line == '\n':
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty

def ccode(expr, parser, assign_to=None, **settings):
    """Converts an expr to a string of c code

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
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    """
    return CCodePrinter(parser, **settings).doprint(expr, assign_to)

# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201
# pylint: disable=missing-function-docstring

"""
Lua code printer

The `LuaCodePrinter` converts SymPy expressions into Lua expressions.

A complete code generator, which uses `lua_code` extensively, can be found
in `sympy.utilities.codegen`. The `codegen` module can be used to generate
complete source code files.

"""

# TODO: add examples

# TODO: find a solution when using math functions. For example, for the moment, we are using
#       math.pow rather than first declaring pow as local and then calling
#       immediatly pow.

# Possible Improvement
#
#



from sympy.core import S
from sympy.printing.precedence import precedence
from sympy.sets.fancysets import Range

from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors import Errors
from pyccel.errors.messages import PYCCEL_RESTRICTION_TODO

__all__ = ["LuaCodePrinter", "lua_code"]


# f64 method in Lua
known_functions = {
    "": "is_nan",
    "": "is_infinite",
    "": "is_finite",
    "": "is_normal",
    "": "classify",
    "floor": "floor",
    "ceiling": "ceil",
    "": "round",
    "": "trunc",
    "": "fract",
    "Abs": "abs",
    "sign": "signum",
    "": "is_sign_positive",
    "": "is_sign_negative",
    "": "mul_add",
    "exp": [(lambda exp: True, "exp", 2)],   # e ** x
    "log": "ln",
    "": "log",          # number.log(base)
    "": "log2",
    "": "log10",
    "": "to_degrees",
    "": "to_radians",
    "Max": "max",
    "Min": "min",
    "": "hypot",        # (x**2 + y**2) ** 0.5
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "": "sin_cos",
    "": "exp_m1",       # e ** x - 1
    "": "ln_1p",        # ln(1 + x)
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
}

# i64 method in Lua
# known_functions_i64 = {
#     "": "min_value",
#     "": "max_value",
#     "": "from_str_radix",
#     "": "count_ones",
#     "": "count_zeros",
#     "": "leading_zeros",
#     "": "trainling_zeros",
#     "": "rotate_left",
#     "": "rotate_right",
#     "": "swap_bytes",
#     "": "from_be",
#     "": "from_le",
#     "": "to_be",    # to big endian
#     "": "to_le",    # to little endian
#     "": "checked_add",
#     "": "checked_sub",
#     "": "checked_mul",
#     "": "checked_div",
#     "": "checked_rem",
#     "": "checked_neg",
#     "": "checked_shl",
#     "": "checked_shr",
#     "": "checked_abs",
#     "": "saturating_add",
#     "": "saturating_sub",
#     "": "saturating_mul",
#     "": "wrapping_add",
#     "": "wrapping_sub",
#     "": "wrapping_mul",
#     "": "wrapping_div",
#     "": "wrapping_rem",
#     "": "wrapping_neg",
#     "": "wrapping_shl",
#     "": "wrapping_shr",
#     "": "wrapping_abs",
#     "": "overflowing_add",
#     "": "overflowing_sub",
#     "": "overflowing_mul",
#     "": "overflowing_div",
#     "": "overflowing_rem",
#     "": "overflowing_neg",
#     "": "overflowing_shl",
#     "": "overflowing_shr",
#     "": "overflowing_abs",
#     "Pow": "pow",
#     "Abs": "abs",
#     "sign": "signum",
#     "": "is_positive",
#     "": "is_negnative",
# }

# These are the core reserved words in the Lua language. Taken from:
# http://doc.lua-lang.org/grammar.html#keywords

reserved_words = ['local',
                  ### the remaining are copy/paste from Rust.
                  # TODO must be removed
                  'abstract',
                  'alignof',
                  'as',
                  'become',
                  'box',
                  'break',
                  'const',
                  'continue',
                  'crate',
                  'do',
                  'else',
                  'enum',
                  'extern',
                  'false',
                  'final',
                  'fn',
                  'for',
                  'if',
                  'impl',
                  'in',
                  'let',
                  'loop',
                  'macro',
                  'match',
                  'mod',
                  'move',
                  'mut',
                  'offsetof',
                  'override',
                  'priv',
                  'proc',
                  'pub',
                  'pure',
                  'ref',
                  'return',
                  'Self',
                  'self',
                  'sizeof',
                  'static',
                  'struct',
                  'super',
                  'trait',
                  'true',
                  'type',
                  'typeof',
                  'unsafe',
                  'unsized',
                  'use',
                  'virtual',
                  'where',
                  'while',
                  'yield']

errors = Errors()

class LuaCodePrinter(CodePrinter):
    """A printer to convert python expressions to strings of Lua code"""
    printmethod = "_lua_code"
    language = "Lua"

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 15,
        'user_functions': {},
        'local_vars': {},
        'human': True,
        'contract': True,
        'dereference': set(),
        'error_on_reserved': False,
        'reserved_word_suffix': '_',
        'inline': False,
    }

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.reserved_words = set(reserved_words)
        self.local_vars = settings.get('local_vars', {})

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s" % codestring

    def _get_comment(self, text):
        return "-- %s" % text

    def _declare_number_const(self, name, value):
        return "const %s: f64 = %s" % (name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for %(var)s in %(start)s..%(end)s {"
        for i in indices:
            # Lua arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'var': self._print(i),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    def _print_caller_var(self, expr):
        if len(expr.args) > 1:
            # for something like `sin(x + y + z)`,
            # make sure we can get '(x + y + z).sin()'
            # instead of 'x + y + z.sin()'
            return '(' + self._print(expr) + ')'
        elif expr.is_number:
            return self._print(expr, _type=True)
        else:
            return self._print(expr)

    def _print_Function(self, expr):
        """
        """
        # TODO improve. for the moment, in ValeCodeGen we subs the expression to
        # add the call to the function
        return str(expr)

    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        if expr.exp == -1:
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        elif expr.exp == 0.5:
            return 'math.sqrt(%s)' % self._print(expr.base)
        else:
            return 'math.pow(%s,%s)' % (self.parenthesize(expr.base, PREC),
                                 self.parenthesize(expr.exp, PREC))

    def _print_Float(self, expr, _type=False):
        ret = super(LuaCodePrinter, self)._print_Float(expr)
        if _type:
            return ret + '_f64'
        else:
            return ret

    def _print_Integer(self, expr, _type=False):
        # TODO use ceiling since Lua does not have an Integer type.
        ret = super(LuaCodePrinter, self)._print_Integer(expr)
        if _type:
            return ret + '_i32'
        else:
            return ret

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d_f64/%d.0' % (p, q)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(list(range(expr.rank))):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

    def _print_Idx(self, expr):
        return expr.label.name

    def _print_Dummy(self, expr):
        return expr.name

    def _print_Exp1(self, expr, _type=False):
        return "E"

    def _print_Pi(self, expr, _type=False):
        return 'PI'

    def _print_Infinity(self, expr, _type=False):
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr, _type=False):
        return 'NEG_INFINITY'

    def _print_BooleanTrue(self, expr, _type=False):
        return "true"

    def _print_BooleanFalse(self, expr, _type=False):
        return "false"

    def _print_bool(self, expr, _type=False):
        return str(expr).lower()

    def _print_NaN(self, expr, _type=False):
        return "NAN"

    def _print_Piecewise(self, expr):
        if not expr.args[-1].cond:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []

        for i, (e, cond) in enumerate(expr.args):
            if i == 0:
                lines.append("if (%s) {" % self._print(cond))
            elif i == len(expr.args) - 1 and cond:
                lines[-1] += " else {"
            else:
                lines[-1] += " else if (%s) {" % self._print(cond)
            code0 = self._print(e)
            lines.append(code0)
            lines.append("}")

        if self._settings['inline']:
            return " ".join(lines)
        else:
            return "\n".join(lines)

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        _piecewise = Piecewise((expr.args[1], expr.args[0]), (expr.args[2], True))
        return self._print(_piecewise)

    def _print_MatrixBase(self, A):
        if A.cols == 1:
            return "[%s]" % ", ".join(self._print(a) for a in A)
        else:
            msg = "Full Matrix Support in Lua need Crates (https://crates.io/keywords/matrix)."
            errors.report(msg, symbol=A,
                severity='fatal')

    # FIXME: Str/CodePrinter could define each of these to call the _print
    # method from higher up the class hierarchy (see _print_NumberSymbol).
    # Then subclasses like us would not need to repeat all this.
    _print_Matrix = \
        _print_MatrixElement = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        _print_MatrixBase

    def _print_Symbol(self, expr):

        name = super(LuaCodePrinter, self)._print_Symbol(expr)
        return name

    # ============ Elements ============ #

    def _print_Module(self, expr):
        return '\n\n'.join(self._print(i) for i in expr.body)

    def _print_Import(self, expr):
        return '#include "{0}"'.format(expr.fil)

    def _print_Declare(self, expr):
        dtype = self._print(expr.dtype)
        variables = ', '.join(self._print(i.name) for i in expr.variables)
        return '{0} {1}'.format(dtype, variables)

    def _print_NativeBool(self, expr):
        return 'bool'

    def _print_NativeInteger(self, expr):
        return 'int'

    def _print_NativeFloat(self, expr):
        return 'float'

    def _print_NativeDouble(self, expr):
        return 'double'

    def _print_NativeVoid(self, expr):
        return 'void'

    def _print_FunctionDef(self, expr):
        if len(expr.results) == 1:
            ret_type = self._print(expr.results[0].dtype)
        elif len(expr.results) > 1:
            # TODO: Use fortran example to add pointer arguments for multiple output
            msg = 'Multiple output arguments is not yet supported in c'
            errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')
        else:
            ret_type = self._print(datatype('void'))
        name = expr.name
        arg_code = ', '.join(self._print(i) for i in expr.arguments)
        body = '\n'.join(self._print(i) for i in expr.body)
        return '{0} {1}({2}) {{\n{3}\n}}'.format(ret_type, name, arg_code, body)

    def _print_Return(self, expr):
        return 'return {0}'.format(self._print(expr.expr))

    def _print_Assign(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)

        local_vars = list(self.local_vars)
        local_vars = [str(x) for x in local_vars]

        if lhs_code in local_vars:
            return ("local %s = %s" % (lhs_code, rhs_code))
        else:
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def _print_AugAssign(self, expr):
        lhs_code = self._print(expr.lhs)
        op = expr.op
        rhs_code = self._print(expr.rhs)
        return "{0} {1}= {2}".format(lhs_code, op, rhs_code)

    def _print_For(self, expr):
        # TODO add step
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, _ = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = '\n'.join(self._print(i) for i in expr.body)

        return ('for {target} = {start}, {stop} do'
                '\n{body}\n'
                'end').format(target=target, start=start,
                stop=stop, body=body)

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        code = [ line.lstrip(' \t') for line in code ]

        tab = "    "
        inc_token = ('do')
        dec_token = ('end')

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token))) for line in code ]

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


def lua_code(expr, assign_to=None, **settings):
    """Converts an expr to a string of Lua code

    expr : Expr
        A sympy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)].  See below for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    locals: dict
        A dictionary that contains the list of local symbols. these symbols will
        be preceeded by local for their first assignment.

    Examples

    >>> from sympy import lua_code, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> lua_code((2*tau)**Rational(7, 2))
    '8*1.4142135623731*tau.powf(7_f64/2.0)'
    >>> lua_code(sin(x), assign_to="s")
    's = x.sin();'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs", 4),
    ...           (lambda x: x.is_integer, "ABS", 4)],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> lua_code(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    '(fabs(x) + x.CEIL()).f()'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(lua_code(expr, tau))
    tau = if (x > 0) {
        x + 1
    } else {
        x
    };

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> lua_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(lua_code(mat, A))
    A = [x.powi(2), if (x > 0) {
        x + 1
    } else {
        x
    }, x.sin()];
    """

    return LuaCodePrinter(settings).doprint(expr, assign_to)


def print_lua_code(expr, **settings):
    """Prints Lua representation of the given expression."""
    print((lua_code(expr, **settings)))

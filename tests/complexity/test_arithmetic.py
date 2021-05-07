# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from sympy.abc import n, m, x, b
from sympy import simplify as sp_simplify
from sympy import Function, Symbol
from pyccel.complexity.arithmetic import OpComplexity
import os

from pyccel.complexity.arithmetic import ADD, SUB, MUL, DIV, IDIV, ABS
SHAPE = Function('SHAPE')
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
out = Symbol('out')
ne1 = Symbol('ne1')
ne2 = Symbol('ne2')
p1 = Symbol('p1')
p2 = Symbol('p2')
k1 = Symbol('k1')
k2 = Symbol('k2')

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]

# ==============================================================================
def test_complexity_ex1():

    f = path_dir + '/ex1.py'
    mode = None

    complexity = OpComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
    # complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [ADD,                                  # add
            SUB,                                  # sub
            MUL,                                  # mul
            DIV,                                  # div
            IDIV,                                 # idiv
            ADD,                                  # aug_add
            SUB,                                  # aug_sub
            MUL,                                  # aug_mul
            DIV,                                  # aug_div
            n * ADD,                              # sum_natural_numbers
            (n-1) * MUL,                          # factorial
            n * ADD,                              # fibonacci
            7 * (ADD + MUL) + 77 * SUB,           #double_loop
            m * n * SUB,                          # double_loop_on_2d_array_C
            m * n * SUB,                          # double_loop_on_2d_array_F
            SHAPE(x,0) * ADD,                     # array_int32_1d_scalar_add
            SHAPE(x, 0) * SUB,                    # array_int32_1d_scalar_sub
            SHAPE(x, 0) * MUL,                    # array_int32_1d_scalar_mul
            SHAPE(x, 0) * DIV,                    # array_int32_1d_scalar_div
            SHAPE(x, 0) * IDIV,                   # array_int32_1d_scalar_idiv
            SHAPE(x, 0) * SHAPE(x, 1) * ADD,      # array_int32_2d_scalar_add
            SHAPE(x, 0) * SHAPE(x, 1) * SUB,      # array_int32_2d_scalar_sub
            SHAPE(x, 0) * SHAPE(x, 1) * MUL,      # array_int32_2d_scalar_mul
            SHAPE(x, 0) * SHAPE(x, 1) * DIV,      # array_int32_2d_scalar_div
            SHAPE(x, 0) * SHAPE(x, 1) * IDIV,     # array_int32_2d_scalar_idiv
            SHAPE(x,0) * ADD,                     # array_int32_1d_add
            SHAPE(x, 0) * SUB,                    # array_int32_1d_sub
            SHAPE(x, 0) * MUL,                    # array_int32_1d_mul
            SHAPE(x, 0) * IDIV,                   # array_int32_1d_idiv
            SHAPE(x, 0) * SHAPE(x, 1) * ADD,      # array_int32_2d_add
            SHAPE(x, 0) * SHAPE(x, 1) * SUB,      # array_int32_2d_sub
            SHAPE(x, 0) * SHAPE(x, 1) * MUL,      # array_int32_2d_mul
            SHAPE(x, 0) * SHAPE(x, 1) * IDIV,     # array_int32_2d_idiv
            9 * SHAPE(x, 1) * ADD,                # array_int32_1d_scalar_add_stride1
            9 * 3 * ADD,                          # array_int32_1d_scalar_add_stride2
            5 * 3 * ADD,                          # array_int32_1d_scalar_add_stride3
            ADD * 5 *(SHAPE(x, 1) - 2),           # array_int32_1d_scalar_add_stride4
            (n / 5) * ADD,                        # sum_natural_numbers_range_step_int
            (n / b) * ADD,                        # sum_natural_numbers_range_step_variable
            ABS,                                  # abs_real_scalar
            FLOOR,                                # floor_real_scalar
            EXP,                                  # exp_real_scalar
            LOG,                                  # log_real_scalar
            SQRT,                                 # sqrt_real_scalar
            SIN,                                  # sin_real_scalar
            COS,                                  # cos_real_scalar
            TAN,                                  # tan_real_scalar
            ARCSIN,                               # arcsin_real_scalar
            ARCCOS,                               # arccos_real_scalar
            ARCTAN,                               # arctan_real_scalar
            SINH,                                 # sinh_real_scalar
            COSH,                                 # cosh_real_scalar
            TANH,                                 # tanh_real_scalar
            ARCSINH,                              # arcsinh_real_scalar
            ARCCOSH,                              # arccosh_real_scalar
            ARCTANH,                              # arctanh_real_scalar
            ARCTAN2,                              # arctan2_real_scalar
            SHAPE(out, 0) * SIN,                  # sin_real_array_1d
            SHAPE(out, 0) * COS,                  # cos_real_array_1d
            SHAPE(out, 0) * TAN,                  # tan_real_array_1d
            SHAPE(out, 0) * ARCSIN,               # arcsin_real_array_1d
            SHAPE(out, 0) * ARCCOS,               # arccos_real_array_1d
            SHAPE(out, 0) * ARCTAN,               # arctan_real_array_1d
            SHAPE(out, 0) * SINH,                 # sinh_real_array_1d
            SHAPE(out, 0) * COSH,                 # cosh_real_array_1d
            SHAPE(out, 0) * TANH,                 # tanh_real_array_1d
            SHAPE(out, 0) * ARCSINH,              # arcsinh_real_array_1d
            SHAPE(out, 0) * ARCCOSH,              # arccosh_real_array_1d
            SHAPE(out, 0) * ARCTANH,              # arctanh_real_array_1d
            SHAPE(out, 0) * ARCTAN2,              # arctan2_real_array_1d
            SIN + MUL + ADD + DIV,                # numpy_math_expr_real_scalar
            SHAPE(out, 0) * (SIN+MUL+ADD+DIV)     # numpy_math_expr_real_array_1d
            ]
    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1


# ==============================================================================
def test_complexity_ex2():

    f = path_dir + '/ex2.py'
    mode = None

    complexity = OpComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
    # complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [n**2 * (ADD * (n - 1)/2 + MUL * (n - 1)/2 + SUB + DIV),                                                   # f1
            n**2 * (ADD*n - ADD + DIV*n + DIV + MUL*n - MUL + 2*SUB)/2,                                               # f2
            n * (n**2 * ADD - ADD + n**2 * MUL - MUL + n * SUB * 3 +  SUB * 3 + n * DIV * 3 - DIV * 3 + SQRT * 6)/6   # f3
            ]

    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1


# ==============================================================================
def test_complexity_ex_assembly():

    f = path_dir + '/ex_assembly.py'
    mode = None

    complexity = OpComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
    # complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [ne1 * ne2 * (p1+1) * (p2+1) * (k1*k2*(ADD + 2*MUL) + (p1+1) * (p2+1) * (5*ADD + 4*SUB + 3 * k1 * k2 * (4 * MUL + ADD))) # assemble_matrix_ex01
            ]

    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1



# ==============================================================================
def test_complexity_mxm():

    f = path_dir + '/mxm.py'
    mode = None

    complexity = OpComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
    # complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [n**3*(ADD + MUL),
            n**3*(ADD + MUL)
            ]

    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1




# ==============================================================================
def test_complexity_qr():

    f = path_dir + '/qr.py'
    mode = None

    complexity = OpComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
    # complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [n*(ADD*n**2 - ADD + 3*DIV*n - 3*DIV + MUL*n**2 - MUL + 6*SQRT + 3*SUB*n + 3*SUB)/6
            ]

    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1

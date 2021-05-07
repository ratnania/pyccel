# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.complexity.intensity import ComputationalIntensity
import os

from pyccel.complexity.arithmetic import ADD, SUB, MUL, DIV, IDIV, ABS
import pytest
from sympy.abc import n,m,x,b
from sympy import Function, Symbol
from sympy import simplify as sp_simplify
SHAPE = Function('SHAPE')
READ = Symbol('READ')
WRITE = Symbol('WRITE')
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
k1 = Symbol('k1')
k2 = Symbol('k2')
p1 = Symbol('p1')
p2 = Symbol('p2')
ne1 = Symbol('ne1')
ne2 = Symbol('ne2')

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]

# ==============================================================================
def test_complexity_ex1():

    f = path_dir + '/ex1.py'
    mode = None

    complexity = ComputationalIntensity(f)
    print(complexity.cost(mode=mode))
    print('----------------------')
    i = 0
    comp = [ADD / (3 * READ + WRITE),  ## add
            SUB / (3 * READ + WRITE),  ## sub
            MUL / (3 * READ + WRITE),  ## mul
            DIV / (3 * READ + WRITE),  ## div
            IDIV / (3 * READ + WRITE),  ## idiv
            ADD / (2 * READ + WRITE),  ## aug_add
            SUB / (2 * READ + WRITE),  ## aug_sub
            MUL / (2 * READ + WRITE),  ## aug_mul
            DIV / (2 * READ + WRITE),  ## aug_div
            ADD / (2 * READ + WRITE),  # sum_natural_numbers
            ((n) * MUL) / ((2 * READ + WRITE) * n),  # factorial
            ADD / (4 * READ + 3 * WRITE),  # fibonacci
            (7 * (ADD + MUL) + 77 * SUB) / (176 * READ + 92 * WRITE),  # double_loop
            SUB / (2 * READ + WRITE),  # double_loop_on_2d_array_C
            SUB / (2 * READ + WRITE),  # double_loop_on_2d_array_F
            ADD / (READ * 2 + WRITE),  ## array_int32_1d_scalar_add
            SUB / (READ * 2 + WRITE),  ## array_int32_1d_scalar_sub
            MUL / (READ * 2 + WRITE),  ## array_int32_1d_scalar_mul
            DIV / (READ * 2 + WRITE),  ## array_int32_1d_scalar_div
            IDIV / (READ * 2 + WRITE),  ## array_int32_1d_scalar_idiv
            ADD / (READ * 2 + WRITE),  ## array_int32_2d_scalar_add
            SUB / (READ * 2 + WRITE),  ## array_int32_2d_scalar_sub
            MUL / (READ * 2 + WRITE),  ## array_int32_2d_scalar_mul
            DIV / (READ * 2 + WRITE),  ## array_int32_2d_scalar_div
            IDIV / (READ * 2 + WRITE),  ## array_int32_2d_scalar_idiv
            ADD / (READ * 2 + WRITE),  ## array_int32_1d_add
            SUB / (READ * 2 + WRITE),  ## array_int32_1d_sub
            MUL / (READ * 2 + WRITE),  ## array_int32_1d_mul
            IDIV / (READ * 2 + WRITE),  ## array_int32_1d_idiv
            ADD / (READ * 2 + WRITE),  ## array_int32_2d_add
            SUB / (READ * 2 + WRITE),  ## array_int32_2d_sub
            MUL / (READ * 2 + WRITE),  ## array_int32_2d_mul
            IDIV / (READ * 2 + WRITE),  ## array_int32_2d_idiv
            ADD / (READ * 2 + WRITE),  ## array_int32_1d_scalar_add_stride1
            ADD / (READ * 2 + WRITE),  ## array_int32_1d_scalar_add_stride2
            ADD / (READ * 2 + WRITE),  ## array_int32_1d_scalar_add_stride3
            ADD / (READ * 2 + WRITE),  ## array_int32_1d_scalar_add_stride4
            ADD / (READ * 2 + WRITE),  # sum_natural_numbers_range_step_int
            ADD / (2 * READ + WRITE),  # sum_natural_numbers_range_step_variable
            ABS / (2 * READ + WRITE),  # abs_real_scalar
            FLOOR / (2 * READ + WRITE),  # floor_real_scalar # FLOOR
            EXP / (2 * READ + WRITE),  # exp_real_scalar
            LOG / (2 * READ + WRITE),  # log_real_scalar
            SQRT / (2 * READ + WRITE),  # sqrt_real_scalar
            SIN / (2 * READ + WRITE),  # sin_real_scalar
            COS / (2 * READ + WRITE),  # cos_real_scalar
            TAN / (2 * READ + WRITE),  # tan_real_scalar
            ARCSIN / (2 * READ + WRITE),  # arcsin_real_scalar
            ARCCOS / (2 * READ + WRITE),  # arccos_real_scalar
            ARCTAN / (2 * READ + WRITE),  # arctan_real_scalar
            SINH / (2 * READ + WRITE),  # sinh_real_scalar
            COSH / (2 * READ + WRITE),  # cosh_real_scalar
            TANH / (2 * READ + WRITE),  # tanh_real_scalar
            ARCSINH / (2 * READ + WRITE),  # arcsinh_real_scalar
            ARCCOSH / (2 * READ + WRITE),  # arccosh_real_scalar
            ARCTANH / (2 * READ + WRITE),  # arctanh_real_scalar
            ARCTAN2 / (3 * READ + WRITE),  # arctan2_real_scalar
            SIN / (READ + WRITE),  ## sin_real_array_1d
            COS / (READ + WRITE),  ## cos_real_array_1d
            TAN / (READ + WRITE),  ## tan_real_array_1d
            ARCSIN / (READ + WRITE),  ## arcsin_real_array_1d
            ARCCOS / (READ + WRITE),  ## arccos_real_array_1d
            ARCTAN / (READ + WRITE),  ## arctan_real_array_1d
            SINH / (READ + WRITE),  ## sinh_real_array_1d
            COSH / (READ + WRITE),  ## cosh_real_array_1d
            TANH / (READ + WRITE),  ## tanh_real_array_1d
            ARCSINH / (READ + WRITE),  ## arcsinh_real_array_1d
            ARCCOSH / (READ + WRITE),  ## arccosh_real_array_1d
            ARCTANH / (READ + WRITE),  ## arctanh_real_array_1d
            ARCTAN2 / (2 * READ + WRITE),  ## arctan2_real_array_1d
            (SIN + MUL + ADD + DIV) / (4 * READ + WRITE),  # numpy_math_expr_real_scalar
            (SIN + MUL + ADD + DIV) / (3 * READ + WRITE)  ## numpy_math_expr_real_array_1d
            ]

    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1


# ==============================================================================
def test_complexity_ex2():

    f = path_dir + '/ex2.py'
    mode = None

    complexity = ComputationalIntensity(f)
    complexity.cost(mode=mode)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [(ADD + MUL) / (3 * READ + WRITE),  # f1
            (ADD + DIV + MUL) / (4 * READ + WRITE),  # f2
            ((ADD + MUL) / 3) / (READ + WRITE / 3)  # f3
            ]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1



# ==============================================================================
def test_complexity_ex_assembly():

    f = path_dir + '/ex_assembly.py'
    mode = None

    complexity = ComputationalIntensity(f)
    complexity.cost(mode=mode)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [(3 * (4 * MUL + ADD))/(22*READ + 9*WRITE)
            ]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1




# ==============================================================================
def test_complexity_mxm():

    f = path_dir + '/mxm.py'
    mode = None

    complexity = ComputationalIntensity(f)
    complexity.cost(mode=mode)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [(ADD + MUL)/(3*READ + WRITE),
            b*(ADD+MUL) /(2 * READ + 2 * WRITE + b * (3 * READ + WRITE))
            ]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1



# ==============================================================================
def test_complexity_qr():

    f = path_dir + '/qr.py'
    mode = None

    complexity = ComputationalIntensity(f)
    complexity.cost(mode=mode)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [((ADD + MUL)/3)/(READ + WRITE/3),
            ]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1

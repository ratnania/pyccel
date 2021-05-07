# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from sympy.abc import n,m,x,b
from sympy import simplify as sp_simplify
from sympy import Function, Symbol

from pyccel.complexity.memory import MemComplexity
import os

SHAPE = Function('SHAPE')
READ = Symbol('READ')
WRITE = Symbol('WRITE')
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

    complexity = MemComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [3 * READ + WRITE,# add
            3 * READ + WRITE,# sub
            3 * READ + WRITE,# mul
            3 * READ + WRITE,# div
            3 * READ + WRITE,# idiv
            2 * READ + WRITE,# aug_add
            2 * READ + WRITE,# aug_sub
            2 * READ + WRITE,# aug_mul
            2 * READ + WRITE,# aug_div
            READ + WRITE + n*(2*READ + WRITE),# sum_natural_numbers
            2*READ*n - READ + WRITE*n,# factorial
            READ + 2 * WRITE + n * (4 * READ + 3 * WRITE),# fibonacci
            176 * READ + 92 * WRITE,# double_loop
            2 * READ + 2 * WRITE + m * n * (2 * READ + WRITE),# double_loop_on_2d_array_C
            2 * READ + 2 * WRITE + m * n * (2 * READ + WRITE),# double_loop_on_2d_array_F
            (READ * 2 + WRITE) * SHAPE(x, 0),## array_int32_1d_scalar_add
            (READ * 2 + WRITE) * SHAPE(x, 0),## array_int32_1d_scalar_sub
            (READ * 2 + WRITE) * SHAPE(x, 0),## array_int32_1d_scalar_mul
            (READ * 2 + WRITE) * SHAPE(x, 0),## array_int32_1d_scalar_div
            (READ * 2 + WRITE) * SHAPE(x, 0),## array_int32_1d_scalar_idiv
            (READ * 2 + WRITE) * SHAPE(x, 0) * SHAPE(x, 1),## array_int32_2d_scalar_add
            (READ * 2 + WRITE) * SHAPE(x, 0) * SHAPE(x, 1),## array_int32_2d_scalar_sub
            (READ * 2 + WRITE) * SHAPE(x, 0) * SHAPE(x, 1),## array_int32_2d_scalar_mul
            (READ * 2 + WRITE) * SHAPE(x, 0) * SHAPE(x, 1),## array_int32_2d_scalar_div
            (READ * 2 + WRITE) * SHAPE(x, 0) * SHAPE(x, 1),## array_int32_2d_scalar_idiv
            (READ * 2 + WRITE) * SHAPE(x, 0),## array_int32_1d_add
            (READ * 2 + WRITE) * SHAPE(x, 0),## array_int32_1d_sub
            (READ * 2 + WRITE) * SHAPE(x, 0),## array_int32_1d_mul
            (READ * 2 + WRITE) * SHAPE(x, 0),## array_int32_1d_idiv
            (READ * 2 + WRITE) * SHAPE(x, 0) * SHAPE(x, 1),## array_int32_2d_add
            (READ * 2 + WRITE) * SHAPE(x, 0) * SHAPE(x, 1),## array_int32_2d_sub
            (READ * 2 + WRITE) * SHAPE(x, 0) * SHAPE(x, 1),## array_int32_2d_mul
            (READ * 2 + WRITE) * SHAPE(x, 0) * SHAPE(x, 1),## array_int32_2d_idiv
            (READ * 2 + WRITE) * SHAPE(x, 1) * 9,## array_int32_1d_scalar_add_stride1
            (READ * 2 + WRITE) * 9 * 3,## array_int32_1d_scalar_add_stride2
            (READ * 2 + WRITE) * 5 * 3,## array_int32_1d_scalar_add_stride3
            (READ * 2 + WRITE) * 5 * (SHAPE(x, 1) - 2),## array_int32_1d_scalar_add_stride4
            READ + WRITE + n / 5 * (READ * 2 + WRITE),# sum_natural_numbers_range_step_int
            (b * (READ + WRITE) + n * (2 * READ + WRITE))/b,# sum_natural_numbers_range_step_variable
            2 * READ + WRITE,# abs_real_scalar
            2 * READ + WRITE,# floor_real_scalar
            2 * READ + WRITE,# exp_real_scalar
            2 * READ + WRITE,# log_real_scalar
            2 * READ + WRITE,# sqrt_real_scalar
            2 * READ + WRITE,# sin_real_scalar
            2 * READ + WRITE,# cos_real_scalar
            2 * READ + WRITE,# tan_real_scalar
            2 * READ + WRITE,# arcsin_real_scalar
            2 * READ + WRITE,# arccos_real_scalar
            2 * READ + WRITE,# arctan_real_scalar
            2 * READ + WRITE,# sinh_real_scalar
            2 * READ + WRITE,# cosh_real_scalar
            2 * READ + WRITE,# tanh_real_scalar
            2 * READ + WRITE,# arcsinh_real_scalar
            2 * READ + WRITE,# arccosh_real_scalar
            2 * READ + WRITE,# arctanh_real_scalar
            3 * READ + WRITE,# arctan2_real_scalar
            (READ + WRITE) * SHAPE(out, 0),## sin_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## cos_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## tan_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## arcsin_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## arccos_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## arctan_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## sinh_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## cosh_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## tanh_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## arcsinh_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## arccosh_real_array_1d
            (READ + WRITE) * SHAPE(out, 0),## arctanh_real_array_1d
            (2 * READ + WRITE) * SHAPE(out, 0),## arctan2_real_array_1d
            4 * READ + WRITE,# numpy_math_expr_real_scalar
            (3 * READ + WRITE) * SHAPE(out, 0)## numpy_math_expr_real_array_1d
            ]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1





# ==============================================================================
def test_complexity_ex2():

    f = path_dir + '/ex2.py'
    mode = None

    complexity = MemComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [READ + WRITE + n**2*(3*READ*(n - 1) + 8*READ + WRITE*(n - 1) + 6*WRITE)/2,# f1
            READ + WRITE + n ** 2 * (4 * READ * (n - 1) + 6 * READ + WRITE * (n - 1) + 4 * WRITE) / 2,# f2
            READ * n ** 3 / 2 + 3 * READ * n ** 2 / 2 + READ + WRITE * n ** 3 / 6 + 3 * WRITE * n ** 2 / 2 + WRITE * n / 3 + WRITE# f3
            ]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1



# ==============================================================================
def test_complexity_ex_assembly():

    f = path_dir + '/ex_assembly.py'
    mode = None

    complexity = MemComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [2*READ + WRITE*k1*k2 + WRITE*(p1 + 1)*(p2 + 1) + 2*WRITE + ne1*(READ + WRITE + ne2*(READ + WRITE*k1*k2 + WRITE + (p1 + 1)**2*(p2 + 1)**2*(14*READ + 6*WRITE + k1*k2*(22*READ + 9*WRITE)) + (p1 + 1)*(p2 + 1)*(READ*(p1 + 1)*(p2 + 1) + WRITE) + (p1 + 1)*(p2 + 1)*(READ + WRITE + k1*(READ + WRITE + k2*(5*READ + 2*WRITE)))))]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1




# ==============================================================================
def test_complexity_mxm():

    f = path_dir + '/mxm.py'
    mode = None

    complexity = MemComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [READ + WRITE + n**3*(3*READ + WRITE),
            (b*(READ + 3*WRITE*b**2 + WRITE) + n**2*(2*b*(READ + WRITE) + n*(2*READ + 2*WRITE + b*(3*READ + WRITE))))/b
            ]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1



# ==============================================================================
def test_complexity_qr():

    f = path_dir + '/qr.py'
    mode = None

    complexity = MemComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [READ*n**3/2 + 3*READ*n**2/2 + READ + WRITE*n**3/6 + 3*WRITE*n**2/2 + WRITE*n/3 + WRITE,
            ]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1

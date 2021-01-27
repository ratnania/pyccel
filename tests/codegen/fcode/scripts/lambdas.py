# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

f1 = lambda x: x**2 + 1
f2 = lambda x,y: x**2 + y**2 + 1
g1 = lambda x: f1(x)**2 + 1

f1(3)

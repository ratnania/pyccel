# pylint: disable=missing-function-docstring, missing-module-docstring/

def f(x : 'int[:]'):
    y = x + 1
    return y[0]

x = 3.9
z = f(x)
print(z)

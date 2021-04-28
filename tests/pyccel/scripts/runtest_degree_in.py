# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import stack_array, pure

@pure
@stack_array('tmp')
def test_degree(degree : int):
    from numpy import empty

    tmp = empty(degree+1, dtype=float)
    for i in range(degree+1):
        tmp[i]=0.
    return 1

print(test_degree(3))

@pure
@stack_array('tmp')
def test_degree2d(degree1 : int, degree2 : int):
    from numpy import empty

    tmp = empty((degree1+1, degree2+1), dtype=float)
    for i in range(degree1+1):
        for j in range(degree2+1):
            tmp[i,j]=0.
    return 1

print(test_degree(3))
print(test_degree2d(3,4))

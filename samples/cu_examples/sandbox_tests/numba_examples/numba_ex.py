import numpy as np
from numba import cuda
from pyccel.decorators import types

@cuda.jit
@types("int[:]")
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.shape[0]:  # Check array boundaries
        an_array[pos] += 1

arr = np.arange(100000000)
d_arr = cuda.to_device(arr)

increment_by_one[1000, 100](d_arr)

result_array = d_arr.copy_to_host()
for i in range(5):
    print(d_arr[i])

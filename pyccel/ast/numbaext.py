from .basic          import Basic, PyccelAstNode
from .internals      import PyccelInternalFunction

from .core           import (ClassDef, FunctionDef,
                            process_shape, ValuedArgument)

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex

from .datatypes      import (dtype_and_precision_registry as dtype_registry,
                             default_precision, datatype, NativeInteger,
                             NativeReal, NativeComplex, NativeBool, str_dtype,
                             NativeNumeric)

from .builtins       import (PythonInt, PythonBool, PythonFloat, PythonTuple,
                             PythonComplex, PythonReal, PythonImag, PythonList)

from .variable       import (Variable, IndexedElement, Constant)

from .numpyext       import process_dtype, NumpyNewArray, NumpyArray

from .cudext         import CudaInternalVar, CudaNewArray
from .cudext         import CudaThreadIdx, CudaBlockDim, CudaBlockIdx, CudaGridDim
#==============================================================================
__all__ = (
    'NumbaThreadIdx',
    'NumbaBlockDim',
    'NumbaBlockIdx',
    'Shape'
    'NumbaGridDim'
)


#------------------------------------------------------------------------------

class NumbaNewArray(PyccelInternalFunction):
    """ Class from which all Cuda functions which imply a call to Allocate
    inherit
    """

    #--------------------------------------------------------------------------
    @staticmethod
    def _process_order(order):

        if not order:
            return None

        order = str(order).strip('\'"')
        if order not in ('C', 'F'):
            raise ValueError('unrecognized order = {}'.format(order))
        return order
#==============================================================================
class NumbaToDevice(NumbaNewArray):
    """
    Represents a call to  numba.cuda.to_device for code generation.

    obj : list, tuple, PythonList, Variable

    """
    _attribute_nodes = ('_arg',)

    def __init__(self, arg, dtype=None, order='C', copy_mode = 'device'):

        if not isinstance(arg, (PythonTuple, PythonList, Variable)):
            raise TypeError('Unknown type of  %s.' % type(arg))

        # TODO: treat inhomogenous lists and tuples when they have mixed ordering
        if isinstance(arg, (PythonTuple, PythonList)) and not arg.is_homogeneous or \
            isinstance(arg, Variable) and not arg.is_ndarray and not arg.is_stack_array:
            raise TypeError('we only accept homogeneous arguments')

        # Verify dtype and get precision
        if dtype is None:
            dtype = arg.dtype
        dtype, prec = process_dtype(dtype)
        # ... Determine ordering
        if isinstance(order, ValuedArgument):
            order = order.value
        order = str(order).strip("\'")

        if order not in ('K', 'A', 'C', 'F'):
            raise ValueError("Cannot recognize '{:s}' order".format(order))

        # TODO [YG, 18.02.2020]: set correct order based on input array
        if order in ('K', 'A'):
            order = 'C'
        # ...
        self._arg   = arg
        self._shape = process_shape(arg.shape)
        self._rank  = len(self._shape)
        self._dtype = dtype
        self._order = order
        self._precision = prec
        self._copy_mode = copy_mode
        super().__init__()

    def __str__(self):
        return str(self.arg)

    @property
    def arg(self):
        return self._arg

def NumbaCopyToHost(arg):
    """
    Represents a call to  numba.cuda.to_device for code generation.

    obj : list, tuple, PythonList, Variable

    """
    _attribute_nodes = ('_arg',)

    if not isinstance(arg, Variable):
        raise TypeError('Unknown type of  %s.' % type(arg))

    if isinstance(arg, Variable) and not arg.is_ndarray and not arg.is_stack_array and not arg.is_ondevice:
            raise TypeError('we only accept device ndarrays')
    return NumpyArray(arg)

def Shape(arg):
    if isinstance(arg.shape, PythonTuple):
        return arg.shape
    else:
        return PythonTuple(*arg.shape)
#==============================================================================
# class NumbaThreadIdx(CudaInternalVar) : pass
# class NumbaBlockDim(CudaInternalVar)  : pass
# class NumbaBlockIdx(CudaInternalVar)  : pass
# class NumbaGridDim(CudaInternalVar)   : pass

#=======================================================================================
NumbaArrayClass = ClassDef('numba.cuda.cudadrv.devicearray.DeviceNDArray',
        methods=[
            FunctionDef('shape',[],[],body=[],
                decorators={'property':'property', 'numba_wrapper':Shape}),
            FunctionDef('copy_to_host',[],[],body=[],
                decorators={'numba_wrapper':NumbaCopyToHost})])
#=======================================================================================

numba_functions = {
    'to_device'         : NumbaToDevice,
    'threadIdx'         : CudaThreadIdx,
    'blockDim'          : CudaBlockDim,
    'blockIdx'          : CudaBlockIdx,
    'shape'             : Shape,
    'copy_to_host'      : NumbaCopyToHost,
    'gridDim'           : CudaGridDim
}

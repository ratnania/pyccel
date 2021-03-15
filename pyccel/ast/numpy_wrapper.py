#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=missing-function-docstring

"""
Handling the transitions between python code and C code using (Numpy/C Api).
"""

import numpy as np

from .datatypes         import (NativeInteger, NativeReal, NativeComplex,
                                NativeBool, NativeGeneric, NativeVoid)

from .cwrapper          import PyccelPyObject, PyccelPyArrayObject

from .core              import FunctionDef, FunctionCall
from .operators         import PyccelNot


from .literals          import LiteralInteger

from .variable          import Variable

from ..errors.errors   import Errors

from pyccel.errors.messages import PYCCEL_RESTRICTION_TODO

errors = Errors()

__all__ = (
    #------- CAST FUNCTIONS ------
    'numpy_get_data',
    'pyarray_to_ndarray',
    #-------CHECK FUNCTIONS ------
    'array_checker',
    'NumpyType_Check',
    #-------HELPERS ------
    'numpy_get_dim',
)

#-------------------------------------------------------------------
#                      Numpy functions
#-------------------------------------------------------------------

# https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_TYPE
numpy_get_type = FunctionDef(name      = 'PyArray_TYPE',
                             body      = [],
                             arguments = [Variable(dtype=PyccelPyArrayObject(), name = 'o', is_pointer=True)],
                             results   = [Variable(dtype=NativeInteger(), name = 'i', precision = 4)])


# numpy array to fortrab ndarray : function definition in pyccel/stdlib/cwrapper.c
pyarray_to_c_ndarray = FunctionDef(
                name      = 'pyarray_to_c_ndarray',
                arguments = [Variable(name = 'a', dtype = PyccelPyArrayObject(), is_pointer = True)],
                body      = [],
                results   = [Variable(name = 'array', dtype = NativeGeneric())])


# numpy array to c ndarray : function definition in pyccel/stdlib/cwrapper.c
pyarray_to_f_ndarray = FunctionDef(
                name      = 'pyarray_to_f_ndarray',
                arguments = [Variable(name = 'a', dtype = PyccelPyArrayObject(), is_pointer = True)],
                body      = [],
                results   = [Variable(name = 'array', dtype = NativeGeneric())])

# numpy array to check element : function definition in pyccel/stdlib/cwrapper.c
pyarray_check = FunctionDef(
                name      = 'pyarray_check',
                arguments = [
                        Variable(name = 'a', dtype = PyccelPyArrayObject(), is_pointer = True),
                        Variable(name = 'dtype', dtype = NativeInteger()),
                        Variable(name = 'rank', dtype = NativeInteger()),
                        Variable(name = 'flag', dtype = NativeInteger())
                    ],
                body      = [], 
                results   = [Variable(name = 'b', dtype = NativeBool())])

# Return the shape in the n dimension : function definition in pyccel/stdlib/cwrapper.c
array_get_dim  = FunctionDef(name    = 'nd_ndim',
                           body      = [],
                           arguments = [Variable(dtype=NativeVoid(), name = 'o', is_pointer=True),
                                        Variable(dtype=NativeInteger(), name = 'idx')],
                           results   = [Variable(dtype=NativeInteger(), name = 'd')])


# Return the data of ndarray : function definition in pyccel/stdlib/cwrapper.c
array_get_data  = FunctionDef(name   = 'nd_data',
                           body      = [],
                           arguments = [Variable(dtype=NativeVoid(), name = 'o', is_pointer=True)],
                           results   = [Variable(dtype=NativeVoid(), name = 'v', is_pointer=True, rank = 1)])



numpy_flag_own_data     = Variable(dtype=NativeInteger(),  name = 'NPY_ARRAY_OWNDATA')
numpy_flag_c_contig     = Variable(dtype=NativeInteger(),  name = 'NPY_ARRAY_C_CONTIGUOUS')
numpy_flag_f_contig     = Variable(dtype=NativeInteger(),  name = 'NPY_ARRAY_F_CONTIGUOUS')
no_type_check           = Variable(dtype=NativeInteger(),  name = 'NO_TYPE_CHECK')
no_order_check          = Variable(dtype=NativeInteger(),  name = 'NO_ORDER_CHECK')

# https://numpy.org/doc/stable/reference/c-api/dtype.html
numpy_bool_type         = Variable(dtype=NativeInteger(),  name = 'NPY_BOOL', precision = 4)
numpy_byte_type         = Variable(dtype=NativeInteger(),  name = 'NPY_BYTE', precision = 4)
numpy_ubyte_type        = Variable(dtype=NativeInteger(),  name = 'NPY_UBYTE', precision = 4)
numpy_short_type        = Variable(dtype=NativeInteger(),  name = 'NPY_SHORT', precision = 4)
numpy_ushort_type       = Variable(dtype=NativeInteger(),  name = 'NPY_USHORT', precision = 4)
numpy_int_type          = Variable(dtype=NativeInteger(),  name = 'NPY_INT32', precision = 4)
numpy_uint_type         = Variable(dtype=NativeInteger(),  name = 'NPY_UINT', precision = 4)
numpy_long_type         = Variable(dtype=NativeInteger(),  name = 'NPY_LONG', precision = 4)
numpy_ulong_type        = Variable(dtype=NativeInteger(),  name = 'NPY_ULONG', precision = 4)
numpy_longlong_type     = Variable(dtype=NativeInteger(),  name = 'NPY_INT64', precision = 4)
numpy_ulonglong_type    = Variable(dtype=NativeInteger(),  name = 'NPY_ULONGLONG', precision = 4)
numpy_float_type        = Variable(dtype=NativeInteger(),  name = 'NPY_FLOAT', precision = 4)
numpy_double_type       = Variable(dtype=NativeInteger(),  name = 'NPY_DOUBLE', precision = 4)
numpy_longdouble_type   = Variable(dtype=NativeInteger(),  name = 'NPY_LONGDOUBLE', precision = 4)
numpy_cfloat_type       = Variable(dtype=NativeInteger(),  name = 'NPY_CFLOAT', precision = 4)
numpy_cdouble_type      = Variable(dtype=NativeInteger(),  name = 'NPY_CDOUBLE', precision = 4)
numpy_clongdouble_type  = Variable(dtype=NativeInteger(),  name = 'NPY_CLONGDOUBLE', precision = 4)

numpy_num_to_type = {0 : numpy_bool_type,
                     1 : numpy_byte_type,
                     2 : numpy_ubyte_type,
                     3 : numpy_short_type,
                     4 : numpy_ushort_type,
                     5 : numpy_int_type,
                     6 : numpy_uint_type,
                     7 : numpy_long_type,
                     8 : numpy_ulong_type,
                     9 : numpy_longlong_type,
                    10 : numpy_ulonglong_type,
                    11 : numpy_float_type,
                    12 : numpy_double_type,
                    13 : numpy_longdouble_type,
                    14 : numpy_cfloat_type,
                    15 : numpy_cdouble_type,
                    16 : numpy_clongdouble_type}

# This dictionary is required as the precision does not line up with the expected type on windows
numpy_int_type_precision_map = {
        1 : np.dtype(np.int8).num,
        2 : np.dtype(np.int16).num,
        4 : np.dtype(np.int32).num,
        8 : np.dtype(np.int64).num}

numpy_dtype_registry = {('bool',4)     : numpy_bool_type,
                        ('int',1)      : numpy_num_to_type[numpy_int_type_precision_map[1]],
                        ('int',2)      : numpy_num_to_type[numpy_int_type_precision_map[2]],
                        ('int',4)      : numpy_num_to_type[numpy_int_type_precision_map[4]],
                        ('int',8)      : numpy_num_to_type[numpy_int_type_precision_map[8]],
                        ('int',16)     : numpy_longlong_type,
                        ('real',4)     : numpy_float_type,
                        ('real',8)     : numpy_double_type,
                        ('real',16)    : numpy_longdouble_type,
                        ('complex',4)  : numpy_cfloat_type,
                        ('complex',8)  : numpy_cdouble_type,
                        ('complex',16) : numpy_clongdouble_type}

# Needed to check for numpy arguments type
Numpy_Bool_ref       = Variable(dtype=NativeVoid(),  name = 'Bool')
Numpy_Int8_ref       = Variable(dtype=NativeVoid(),  name = 'Int8')
Numpy_Int16_ref      = Variable(dtype=NativeVoid(),  name = 'Int16')
Numpy_Int32_ref      = Variable(dtype=NativeVoid(),  name = 'Int32')
Numpy_Int64_ref      = Variable(dtype=NativeVoid(),  name = 'Int64')
Numpy_Float_ref      = Variable(dtype=NativeVoid(),  name = 'Float32')
Numpy_Double_ref     = Variable(dtype=NativeVoid(),  name = 'Float64')
Numpy_Complex64_ref  = Variable(dtype=NativeVoid(),  name = 'Complex64')
Numpy_Complex128_ref = Variable(dtype=NativeVoid(),  name = 'Complex128')

numpy_type_check_registry = {
    (NativeInteger(), 4)       : Numpy_Int32_ref,
    (NativeInteger(), 8)       : Numpy_Int64_ref,
    (NativeInteger(), 2)       : Numpy_Int16_ref,
    (NativeInteger(), 1)       : Numpy_Int8_ref,
    (NativeReal(), 8)          : Numpy_Double_ref,
    (NativeReal(), 4)          : Numpy_Float_ref,
    (NativeComplex(), 4)       : Numpy_Complex64_ref,
    (NativeComplex(), 8)       : Numpy_Complex128_ref,
    (NativeBool(), 4)          : Numpy_Bool_ref
}

# helpers
def find_in_numpy_dtype_registry(var):
    """ Find the numpy dtype key for a given variable
    """
    dtype = str(var.dtype)
    prec  = var.precision
    try :
        return numpy_dtype_registry[(dtype, prec)]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO,
                symbol = "{}[kind = {}]".format(dtype, prec),
                severity='fatal')


def array_checker(py_variable, c_variable, type_check_needed, language):
    """
    Create FunctionCall responsible of checking numpy array and collecting its value
    Parameters:
    ----------
    c_variable  : Variable
        The variable needed for the generation of the type check
    py_variable : Variable
        The python argument of the check function
    type_check_needed : boolean
        True if data type check is needed, used to avoid multiple type check
        in interface
    language    : string
        Needed to collect the flag for order check
    Returns
    -------
    FunctionCall : Check type FunctionCall
    """
    rank     = c_variable.rank
    type_ref = no_type_check
    flag     = no_order_check

    # extract numpy type ref
    if type_check_needed:
        type_ref = find_in_numpy_dtype_registry(c_variable)
        
    # order flag
    if rank > 1 and language == 'fortran':
        if c_variable.order == 'F':
            flag = numpy_flag_f_contig
        else:
            flag = numpy_flag_c_contig

    check = PyccelNot(FunctionCall(pyarray_check, [py_variable, type_ref, LiteralInteger(rank), flag]))

    return check

def array_type_check(py_variable, c_variable):
    """
    """
    # extract numpy type ref
    type_ref = find_in_numpy_dtype_registry(c_variable)
    
    return PyccelEq(FunctionCall(numpy_get_type, [py_variable]), type_ref)


def NumpyType_Check(py_variable, c_variable):
    """
    Create FunctionCall responsible of checking numpy argument data type
    Parameters:
    ----------
    c_variable : Variable
        The variable needed for the generation of the type check
    py_variable : Variable
        The python argument of the check function
    Returns
    -------
    FunctionCall : Check type FunctionCall
    """
    try :
        check_numpy_ref = numpy_type_check_registry[(c_variable.dtype, c_variable.precision)]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_variable.dtype,severity='fatal')

    check_numpy_func = FunctionDef(name = 'PyArray_IsScalar',
                              body      = [],
                              arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True),
                                           check_numpy_ref],
                              results   = [Variable(dtype=NativeBool(), name = 'r')])

    return FunctionCall(check_numpy_func, [py_variable, check_numpy_ref])
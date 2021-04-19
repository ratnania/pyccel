from collections import OrderedDict
from pyccel.codegen.printing.ccode import CCodePrinter

import functools
import operator

from pyccel.ast.builtins  import PythonRange, PythonFloat, PythonComplex

from pyccel.ast.core      import Declare
from pyccel.ast.core      import FuncAddressDeclare, FunctionCall
from pyccel.ast.core      import Deallocate
from pyccel.ast.core      import FunctionAddress
from pyccel.ast.core      import Assign, datatype, Import, AugAssign
from pyccel.ast.core      import SeparatorComment
from pyccel.ast.core      import create_incremented_string

from pyccel.ast.operators import PyccelAdd, PyccelMul, PyccelMinus, PyccelLt, PyccelGt
from pyccel.ast.operators import PyccelAssociativeParenthesis
from pyccel.ast.operators import PyccelUnarySub, IfTernaryOperator

from pyccel.ast.datatypes import default_precision, str_dtype
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeComplex, NativeReal, NativeTuple

from pyccel.ast.internals import Slice

from pyccel.ast.literals  import LiteralTrue, LiteralImaginaryUnit, LiteralFloat
from pyccel.ast.literals  import LiteralString, LiteralInteger, Literal
from pyccel.ast.literals  import Nil

from pyccel.ast.numpyext import NumpyFull, NumpyArray, NumpyArange
from pyccel.ast.numpyext import NumpyReal, NumpyImag, NumpyFloat

from pyccel.ast.cudext import CudaMalloc, CudaArray
from pyccel.ast.numbaext import NumbaToDevice

from pyccel.ast.cupyext import CupyArray, CupyFull, CupyArange


from pyccel.ast.variable import ValuedVariable
from pyccel.ast.variable import PyccelArraySize, Variable, VariableAddress
from pyccel.ast.variable import DottedName


from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors   import Errors
from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, INCOMPATIBLE_TYPEVAR_TO_FUNC,
                                    PYCCEL_RESTRICTION_IS_ISNOT )

from .fcode import python_builtin_datatypes

cuda_Internal_Var = {
    'CudaThreadIdx' : 'threadIdx',
    'CudaBlockDim'  : 'blockDim',
    'CudaBlockIdx'  : 'blockIdx',
    'CudaGridDim'   : 'gridDim'
}

errors = Errors()

class CuCodePrinter(CCodePrinter):
    """A printer to convert python expressions to strings of c code"""
    printmethod = "_cucode"
    language = "cuda"
    
    def __init__(self, parser, **settings):

        if parser.filename:
            errors.set_target(parser.filename, 'file')

        prefix_module = settings.pop('prefix_module', None)
        CCodePrinter.__init__(self, parser, **settings)
        self._dereference = set(settings.get('dereference', []))
        self.prefix_module = prefix_module
        self._additional_imports = set(['stdlib'])
        self._parser = parser
        self._additional_code = ''
        self._additional_declare = []
        self._additional_args = []
        self._temporary_args = []
        self._additional_imports.add('cuda_ndarrays')

    def _print_KernelCall(self, expr):
        func_name = self._print(expr.func.func_name)
        dims = ",".join([self._print(i) for i in expr.dims])
        args = ",".join([self._print(i) for i in expr.func.args])
        kcall = "{func_name}<<<{dims}>>>({args});".format(func_name=func_name, dims=dims, args=args)
        return kcall

    def _print_FunctionDef(self, expr):

        if len(expr.results) > 1:
            self._additional_args.append(expr.results)
        body  = self._print(expr.body)
        decs  = [Declare(i.dtype, i) if isinstance(i, Variable) else FuncAddressDeclare(i) for i in expr.local_vars]
        if len(expr.results) <= 1 :
            decs += [Declare(i.dtype, i) if isinstance(i, Variable) else FuncAddressDeclare(i) for i in expr.results]
        decs += [Declare(i.dtype, i) for i in self._additional_declare]
        decs  = '\n'.join(self._print(i) for i in decs)
        self._additional_declare.clear()

        sep = self._print(SeparatorComment(40))
        if self._additional_args :
            self._additional_args.pop()
        imports = ''.join(self._print(i) for i in expr.imports)
        doc_string = self._print(expr.doc_string) if expr.doc_string else ''

        parts = [sep,
                 doc_string,
                '{signature}\n{{'.format(signature=self.function_signature(expr)),
                 imports,
                 decs,
                 body,
                 '}',
                 sep]

        return '\n'.join(p for p in parts if p)

    def _print_IndexedElement(self, expr):
        base = expr.base
        inds = list(expr.indices)
        base_shape = base.shape
        allow_negative_indexes = base.allows_negative_indexes
        for i, ind in enumerate(inds):
            if isinstance(ind, PyccelUnarySub) and isinstance(ind.args[0], LiteralInteger):
                inds[i] = PyccelMinus(base_shape[i], ind.args[0])
            else:
                #indices of indexedElement of len==1 shouldn't be a tuple
                if isinstance(ind, tuple) and len(ind) == 1:
                    inds[i].args = ind[0]
                if allow_negative_indexes and \
                        not isinstance(ind, LiteralInteger) and not isinstance(ind, Slice):
                    inds[i] = IfTernaryOperator(PyccelLt(ind, LiteralInteger(0)),
                        PyccelAdd(base_shape[i], ind), ind)
        #set dtype to the C struct types
        dtype = self._print(expr.dtype)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.precision)
        base_name = self._print(base.name)
        if base.is_ndarray:
            if expr.rank > 0:
                #managing the Slice input
                for i , ind in enumerate(inds):
                    if isinstance(ind, Slice):
                        inds[i] = self._new_slice_with_processed_arguments(ind, PyccelArraySize(base, i),
                            allow_negative_indexes)
                    else:
                        inds[i] = Slice(ind, PyccelAdd(ind, LiteralInteger(1)), LiteralInteger(1))
                inds = [self._print(i) for i in inds]
                return "array_slicing(%s, %s, %s)" % (base_name, expr.rank, ", ".join(inds))
            inds = [self._print(i) for i in inds]
        else:
            raise NotImplementedError(expr)
        indices = []
        for i, ind in enumerate(inds):
            indices.append('{}.strides[{}] * {}'.format(base_name, i, ind))
        return "%s.%s[%s]" % (base_name, dtype, "+ ".join(indices))

    def function_signature(self, expr):
        # print(expr)
        args = list(expr.arguments)
        if len(expr.results) == 1:
            ret_type = self.get_declare_type(expr.results[0])
        elif len(expr.results) > 1:
            ret_type = self._print(datatype('int')) + ' '
            args += [a.clone(name = a.name, is_pointer =True) for a in expr.results]
        else:
            ret_type = self._print(datatype('void')) + ' '
        name = expr.name
        if not args:
            arg_code = 'void'
        else:
            arg_code = ', '.join('{}'.format(self.function_signature(i))
                        if isinstance(i, FunctionAddress) else '{0}{1}'.format(self.get_declare_type(i), i)
                        for i in args)

        decorator = '__global__' if 'cuda' or 'cuda.jit' in expr.decorators else ''

        if isinstance(expr, FunctionAddress):
            return '{0}\n{1}(*{2})({3})'.format(decorator, ret_type, name, arg_code)
        else:
            return '{0}\n{1}{2}({3})'.format(decorator, ret_type, name, arg_code)

    def _print_Assign(self, expr):
        # print(expr, expr.rhs, type(expr.rhs))
        if isinstance(expr.rhs, FunctionCall) and isinstance(expr.rhs.dtype, NativeTuple):
            self._temporary_args = [VariableAddress(a) for a in expr.lhs]
            return '{};'.format(self._print(expr.rhs))
        ######
        if isinstance(expr.rhs, (CupyArray, CudaArray, NumbaToDevice)):
            return self.copy_CudaArray_Data(expr)
        if isinstance(expr.rhs, (CupyFull)):
            return self.Cuda_arrayFill(expr)
        # if isinstance(expr.rhs, CupyArange):
        #     return self.Cuda_ArangeFill(expr)
        # if isinstance(expr.rhs, (CudaMalloc)):
        #     return self.CudaMalloc(expr.lhs, expr.rhs)
        ######
       ## print(expr, expr.lhs, expr.rhs, type(expr.rhs))
        if isinstance(expr.rhs, (NumpyArray)):
            return self.copy_NumpyArray_Data(expr)
        if isinstance(expr.rhs, (NumpyFull)):
            return self.arrayFill(expr)
        if isinstance(expr.rhs, NumpyArange):
            return self.fill_NumpyArange(expr.rhs, expr.lhs)
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return '{} = {};'.format(lhs, rhs)

    def copy_NumpyArray_Data(self, expr):
        """ print the assignment of a NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains the declaration of a dummy data_buffer
                       and a call to an operator which copies it to an NdArray struct
                if the ndarray is a stack_array the str will contain the initialization
        """
        rhs = expr.rhs
        lhs = expr.lhs
        if rhs.rank == 0:
            raise NotImplementedError(str(expr))
        dummy_array_name, _ = create_incremented_string(self._parser.used_names, prefix = 'array_dummy')
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
        dtype = self.find_in_ndarray_type_registry(self._print(rhs.dtype), rhs.precision)
        arg = rhs.arg
        if rhs.rank > 1:
            # flattening the args to use them in C initialization.
            arg = self._flatten_list(arg)

        if isinstance(arg, Variable):
            arg = self._print(arg)
            if expr.lhs.is_stack_array:
                cpy_data = self._init_stack_array(expr, rhs.arg)
            elif rhs.arg.is_ondevice:
                cpy_data = "cudaMemcpy({0}.{2}, {1}.{2}, {0}.buffer_size, cudaMemcpyDeviceToHost);".format(lhs, arg, dtype)
            else:
                cpy_data = "memcpy({0}.{2}, {1}.{2}, {0}.buffer_size);".format(lhs, arg, dtype)
            return '%s\n' % (cpy_data)
        else :
            arg = ', '.join(self._print(i) for i in arg)
            dummy_array = "%s %s[] = {%s};\n" % (declare_dtype, dummy_array_name, arg)
            if expr.lhs.is_stack_array:
                cpy_data = self._init_stack_array(expr, dummy_array_name)
            else:
                cpy_data = "memcpy({0}.{2}, {1}, {0}.buffer_size);".format(self._print(lhs), dummy_array_name, dtype)
            return  '%s%s\n' % (dummy_array, cpy_data)

    def copy_CudaArray_Data(self, expr):
        """ print the assignment of a NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains the declaration of a dummy data_buffer
                       and a call to an operator which copies it to an NdArray struct
                if the ndarray is a stack_array the str will contain the initialization
        """
        rhs = expr.rhs
        lhs = expr.lhs
        if rhs.rank == 0:
            raise NotImplementedError(str(expr))
        dummy_array_name, _ = create_incremented_string(self._parser.used_names, prefix = 'array_dummy')
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
        dtype = self.find_in_ndarray_type_registry(self._print(rhs.dtype), rhs.precision)
        arg = rhs.arg
        if rhs.rank > 1:
            #flattening the args to use them in C initialization.
            arg = self._flatten_list(arg)
        if isinstance(arg, Variable):
            arg = self._print(arg)
            cpy_data = "cudaMemcpy({0}.{2}, {1}.{2}, {0}.buffer_size, cudaMemcpyHostToDevice);".format(lhs, arg, dtype)
            return '%s\n' % (cpy_data)
        else :
            arg = ', '.join(self._print(i) for i in arg)
            dummy_array = "%s %s[] = {%s};\n" % (declare_dtype, dummy_array_name, arg)
            cpy_data = "cudaMemcpy({0}.{2}, {1}, {0}.buffer_size, cudaMemcpyHostToDevice);".format(self._print(lhs), dummy_array_name, dtype)
            return  '%s%s\n' % (dummy_array, cpy_data)

    def _print_Allocate(self, expr):
        free_code = ''
        #free the array if its already allocated and checking if its not null if the status is unknown
        if  (expr.status == 'unknown'):
            free_code = 'if (%s.shape != NULL)\n' % self._print(expr.variable.name)
            free_code += "{{\n{};\n}}\n".format(self._print(Deallocate(expr.variable)))
        elif  (expr.status == 'allocated'):
            free_code += self._print(Deallocate(expr.variable))
        self._additional_imports.add('ndarrays')
        self._additional_imports.add('cuda_ndarrays')
        shape = ", ".join(self._print(i) for i in expr.shape)
        dtype = self._print(expr.variable.dtype)
        dummy_shape_name, _ = create_incremented_string(self._parser.used_names, prefix = 'shape_dummy')
        shape_dtype = self.find_in_dtype_registry('int', 8)
        dummy_shape = "%s %s[] = {%s};\n" % (shape_dtype, dummy_shape_name, shape)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.variable.precision)
        if not expr.variable.is_ondevice:
            alloc_code = "{} = array_create({}, {}, {});".format(expr.variable, len(expr.shape), dummy_shape_name, dtype)
        else:
            alloc_code = "{} = cuda_array_create({}, {}, {});".format(expr.variable, len(expr.shape), dummy_shape_name, dtype)

        return '{}\n{}{}'.format(free_code, dummy_shape,alloc_code)

    def _print_Deallocate(self, expr):
        cuda = ''
        #used cudamallocManaged for now so we don't need to deal with deallocation
        if expr.variable.is_ondevice:
            return ''
        if expr.variable.is_pointer:
            return '{}free_pointer({});'.format(cuda, self._print(expr.variable))
        return '{}free_array({});'.format(cuda, self._print(expr.variable))

    def _print_CudaDeviceSynchronize(self, expr):
        return 'cudaDeviceSynchronize();'

    def _print_CudaInternalVar(self, expr):
        var_name = type(expr).__name__
        var_name = cuda_Internal_Var[var_name]
        dim_c = ('x', 'y', 'z')[expr.dim]
        return '{}.{}'.format(var_name, dim_c)

    def Cuda_arrayFill(self, expr):
        """ print the assignment of a NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains a call to the C function array_fill,
                if the ndarray is a stack_array the str will contain the initialization
        """
        rhs = expr.rhs
        lhs = expr.lhs
        code_init = ''
        if lhs.is_stack_array:
            declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
            length = '*'.join(self._print(i) for i in lhs.shape)
            buffer_array = "({declare_dtype}[{length}]){{}}".format(declare_dtype = declare_dtype, length=length)
            code_init += self._init_stack_array(expr, buffer_array)
        if rhs.fill_value is not None:
            if isinstance(rhs.fill_value, Literal):
                dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
                code_init += 'cuda_array_fill<<<1,1>>>(({0}){1}, {2});\n'.format(dtype, self._print(rhs.fill_value), self._print(lhs))
            else:
                code_init += 'cuda_array_fill<<<1,1>>>({0}, {1});\n'.format(self._print(rhs.fill_value), self._print(lhs))
        return '{}'.format(code_init)

    def _print_ModuleHeader(self, expr):
        name = expr.module.name
        # TODO: Add classes and interfaces
        funcs = '\n\n'.join('{};'.format(self.function_signature(f)) for f in expr.module.funcs)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *map(Import, self._additional_imports)]
        imports = '\n'.join(self._print(i) for i in imports)

        return ('#ifndef {name}_H\n'
                '#define {name}_H\n\n'
                '{imports}\n\n'
                '#ifndef __cplusplus\nextern "C" {{\n#endif\n'
                #'{classes}\n\n'
                '{funcs}\n\n'
                '#ifndef __cplusplus\n}}\n#endif\n'
                #'{interfaces}\n\n'
                '#endif // {name}_H\n').format(
                        name    = name.upper(),
                        imports = imports,
                        funcs   = funcs)

def cucode(expr, parser, assign_to=None, **settings):
    """Converts an expr to a string of c code

    expr : Expr
        A pyccel expression to be converted.
    parser : Parser
        The parser used to collect the expression
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    """
    return CuCodePrinter(parser, **settings).doprint(expr, assign_to)

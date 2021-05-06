#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import inspect
from itertools import chain
from collections import namedtuple

from numpy import pi

import pyccel.decorators as pyccel_decorators
from pyccel.symbolic import lambdify
from pyccel.errors.errors import Errors

from .core          import (AsName, Import, FunctionDef, FunctionCall,
                            Allocate, Duplicate, Assign, For, CodeBlock,
                            Concatenate)

from .builtins      import (builtin_functions_dict, PythonMap,
                            PythonRange, PythonList, PythonTuple)
from .internals     import PyccelInternalFunction, Slice
from .itertoolsext  import Product
from .mathext       import math_functions, math_constants
from .literals      import LiteralString, LiteralInteger, Literal, Nil

from .numpyext      import (numpy_functions, numpy_linalg_functions,
                            numpy_random_functions, numpy_constants, NumpyLinspace)
from .operators     import PyccelAdd, PyccelMul, PyccelIs
from .variable      import (Constant, Variable, ValuedVariable,
                            IndexedElement, InhomogeneousTupleVariable, VariableAddress)

errors = Errors()

__all__ = (
    'builtin_function',
    'builtin_import',
    'builtin_import_registery',
    'split_positional_keyword_arguments',
)

scipy_constants = {
    'pi': Constant('real', 'pi', value=pi),
                  }

#==============================================================================
def builtin_function(expr, args=None):
    """Returns a builtin-function call applied to given arguments."""

    if isinstance(expr, FunctionCall):
        name = str(expr.funcdef)
    elif isinstance(expr, str):
        name = expr
    else:
        raise TypeError('expr must be of type str or FunctionCall')

    dic = builtin_functions_dict

    if name in dic.keys() :
        return dic[name](*args)

    if name == 'map':
        return PythonMap(expr.args[0], *args[1:])

    if name == 'lambdify':
        return lambdify(expr, args)

    return None


# TODO add documentation
builtin_import_registery = {'numpy': {
                                      **numpy_functions,
                                      **numpy_constants,
                                      'linalg':numpy_linalg_functions,
                                      'random':numpy_random_functions
                                      },
                            'numpy.linalg': numpy_linalg_functions,
                            'numpy.random': numpy_random_functions,
                            'scipy.constants': scipy_constants,
                            'itertools': {'product': Product},
                            'math': {**math_functions, ** math_constants},
                            'pyccel.decorators': None}

#==============================================================================
def collect_relevant_imports(func_dictionary, targets):
    if len(targets) == 0:
        return func_dictionary

    imports = []
    for target in targets:
        if isinstance(target, AsName):
            import_name = target.name
            code_name = target.target
        else:
            import_name = target
            code_name = import_name

        if import_name in func_dictionary.keys():
            imports.append((code_name, func_dictionary[import_name]))
    return imports

def builtin_import(expr):
    """Returns a builtin pyccel-extension function/object from an import."""

    if not isinstance(expr, Import):
        raise TypeError('Expecting an Import expression')

    if isinstance(expr.source, AsName):
        source = expr.source.name
    else:
        source = str(expr.source)

    if source == 'pyccel.decorators':
        funcs = [f[0] for f in inspect.getmembers(pyccel_decorators, inspect.isfunction)]
        for target in expr.target:
            search_target = target.name if isinstance(target, AsName) else target
            if search_target not in funcs:
                errors = Errors()
                errors.report("{} does not exist in pyccel.decorators".format(target),
                        symbol = expr, severity='error')

    elif source in builtin_import_registery:
        return collect_relevant_imports(builtin_import_registery[source], expr.target)

    return []

#==============================================================================
def get_function_from_ast(ast, func_name):
    node = None
    for stmt in ast:
        if isinstance(stmt, FunctionDef) and str(stmt.name) == func_name:
            node = stmt
            break

    if node is None:
        print('> could not find {}'.format(func_name))

    return node

#==============================================================================
def split_positional_keyword_arguments(*args):
    """ Create a list of positional arguments and a dictionary of keyword arguments
    """

    # Distinguish between positional and keyword arguments
    val_args = ()
    for i, a in enumerate(args):
        if isinstance(a, ValuedVariable):
            args, val_args = args[:i], args[i:]
            break

    # Convert list of keyword arguments into dictionary
    kwargs = {}
    for v in val_args:
        key   = str(v.name)
        value = v.value
        kwargs[key] = value

    return args, kwargs

#==============================================================================
def compatible_operation(*args, language_has_vectors = True):
    """
    Indicates whether an operation requires an index to be
    correctly understood

    Parameters
    ==========
    args      : list of PyccelAstNode
                The operator arguments
    language_has_vectors : bool
                Indicates if the language has support for vector
                operations of the same shape
    Results
    =======
    compatible : bool
                 A boolean indicating if the operation is compatible
    """
    if language_has_vectors:
        # If the shapes don't match then an index must be required
        shapes = [a.shape[::-1] if a.order == 'F' else a.shape for a in args if a.shape != ()]
        shapes = set(tuple(d if isinstance(d, Literal) else -1 for d in s) for s in shapes)
        order  = set(a.order for a in args if a.order is not None)
        return len(shapes) <= 1 and len(order) <= 1
    else:
        return all(a.shape==() for a in args)

#==============================================================================
def insert_index(expr, pos, index_var):
    """
    Function to insert an index into an expression at a given position

    Parameters
    ==========
    expr        : Ast Node
                The expression to be modified
    pos         : int
                The index at which the expression is modified
                (If negative then there is no index to insert)
    index_var   : Variable
                The variable which will be used for indexing

    Returns
    =======
    expr        : Ast Node
                Either a modified version of expr or expr itself

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.operators import PyccelAdd
    >>> from pyccel.ast.utilities import insert_index
    >>> a = Variable('int', 'a', shape=(4,), rank=1)
    >>> b = Variable('int', 'b', shape=(4,), rank=1)
    >>> c = Variable('int', 'c', shape=(4,), rank=1)
    >>> i = Variable('int', 'i', shape=())
    >>> d = PyccelAdd(a,b)
    >>> expr = Assign(c,d)
    >>> insert_index(expr, 0, i, language_has_vectors = False)
    IndexedElement(c, i) := IndexedElement(a, i) + IndexedElement(b, i)
    >>> insert_index(expr, 0, i, language_has_vectors = True)
    c := a + b
    """
    if isinstance(expr, (Variable, VariableAddress)):
        if expr.rank==0 or -pos>expr.rank:
            return expr
        if expr.shape[pos]==1:
            # If there is no dimension in this axis, reduce the rank
            index_var = LiteralInteger(0)

        # Add index at the required position
        indexes = [Slice(None,None)]*(expr.rank+pos) + [index_var]+[Slice(None,None)]*(-1-pos)
        return IndexedElement(expr, *indexes)

    elif isinstance(expr, IndexedElement):
        base = expr.base
        indices = list(expr.indices)
        while -pos<=expr.base.rank and not isinstance(indices[pos], Slice):
            pos -= 1
        if -pos>expr.base.rank:
            return expr

        # Add index at the required position
        if expr.base.shape[pos]==1:
            # If there is no dimension in this axis, reduce the rank
            assert(indices[pos].start is None)
            index_var = LiteralInteger(0)

        else:
            # Calculate new index to preserve slice behaviour
            if indices[pos].step is not None:
                index_var = PyccelMul(index_var, indices[pos].step, simplify=True)
            if indices[pos].start is not None:
                index_var = PyccelAdd(index_var, indices[pos].start, simplify=True)

        indices[pos] = index_var
        return IndexedElement(base, *indices)

    else:
        raise NotImplementedError("Expansion not implemented for type : {}".format(type(expr)))

#==============================================================================

LoopCollection = namedtuple('LoopCollection', ['body', 'length', 'modified_vars'])

#==============================================================================
def collect_loops(block, indices, new_index_name, tmp_vars, language_has_vectors = False, result = None):
    """
    Run through a code block and split it into lists of tuples of lists where
    each inner list represents a code block and the tuples contain the lists
    and the size of the code block.
    So the following:
    a = a+b
    for a: int[:,:] and b: int[:]
    Would be returned as:
    [
      ([
        ([a[i,j]=a[i,j]+b[j]],a.shape[1])
       ]
       , a.shape[0]
      )
    ]

    Parameters
    ==========
    block                 : list of Ast Nodes
                            The expressions to be modified
    indices               : list
                            An empty list to be filled with the temporary variables created
    new_index_name        : function
                            A function which provides a new variable name from a base name,
                            avoiding name collisions
    tmp_vars              : list
                            A list to which any temporary variables created can be appended
    language_has_vectors  : bool
                            Indicates if the language has support for vector
                            operations of the same shape
    Results
    =======
    block : list of tuples of lists
            The modified expression
    """
    if result is None:
        result = []
    current_level = 0
    array_creator_types = (Allocate, PythonList, PythonTuple, Concatenate, Duplicate)
    is_function_call = lambda f: ((isinstance(f, FunctionCall) and not f.funcdef.is_elemental)
                                or (isinstance(f, PyccelInternalFunction) and not f.is_elemental))
    for line in block:

        if (isinstance(line, Assign) and
                not isinstance(line.rhs, (array_creator_types, Nil)) and # not creating array
                not line.rhs.get_attribute_nodes(array_creator_types, excluded_nodes = (ValuedVariable)) and # not creating array
                not is_function_call(line.rhs)): # not a basic function call

            # Collect lhs variable
            # This is needed to know what has already been modified in the loop
            if isinstance(line.lhs, Variable):
                lhs_vars = [line.lhs]
            elif isinstance(line.lhs, IndexedElement):
                lhs_vars = [line.lhs.base]
            else:
                lhs_vars = set(line.lhs.get_attribute_nodes((Variable, IndexedElement)))
                lhs_vars = [v.base if isinstance(v, IndexedElement) else v for v in lhs_vars]

            # Get all objects which affect where indices are inserted
            notable_nodes = line.get_attribute_nodes((Variable,
                                                       IndexedElement,
                                                       VariableAddress,
                                                       FunctionCall,
                                                       PyccelInternalFunction,
                                                       PyccelIs))

            # Find all elemental function calls. Normally function call arguments are not indexed
            # However elemental functions are an exception
            elemental_func_calls  = [f for f in notable_nodes if (isinstance(f, FunctionCall) \
                                                                and f.funcdef.is_elemental)]
            elemental_func_calls += [f for f in notable_nodes if (isinstance(f, PyccelInternalFunction) \
                                                                and f.is_elemental)]

            # Collect all objects into which indices may be inserted
            variables       = [v for v in notable_nodes if isinstance(v, (Variable,
                                                                          IndexedElement,
                                                                          VariableAddress))]
            variables      += [v for f in elemental_func_calls \
                                 for v in f.get_attribute_nodes((Variable, IndexedElement, VariableAddress),
                                                            excluded_nodes = (FunctionDef))]

            is_checks = [n for n in notable_nodes if isinstance(n, PyccelIs)]

            variables = list(set(variables))

            # Check if the expression is already satisfactory
            if compatible_operation(*variables, *is_checks, language_has_vectors = language_has_vectors):
                result.append(line)
                current_level = 0
                continue

            # Find function calls in this line
            funcs           = [f for f in notable_nodes if (isinstance(f, FunctionCall) \
                                                            and not f.funcdef.is_elemental)]
            internal_funcs  = [f for f in notable_nodes if (isinstance(f, PyccelInternalFunction) \
                                                            and not f.is_elemental)]

            # Collect all variables for which values other than the value indexed in the loop are important
            # E.g. x = np.sum(a) has a dependence on a
            dependencies = set(v for f in chain(funcs, internal_funcs) \
                                 for v in f.get_attribute_nodes((Variable, IndexedElement, VariableAddress),
                                     excluded_nodes = (FunctionDef)))

            # Replace function calls with temporary variables
            # This ensures that the function is only called once and stops problems
            # for expressions such as:
            # c += b*np.sum(c)
            func_vars1 = [Variable(f.dtype, new_index_name('tmp')) for f in internal_funcs]
            _          = [v.copy_attributes(f) for v,f in zip(func_vars1, internal_funcs)]
            assigns    = [Assign(v, f) for v,f in zip(func_vars1, internal_funcs)]


            if any(len(f.funcdef.results)!=1 for f in funcs):
                errors.report("Loop unravelling cannot handle function calls \
                        which return tuples or None",
                        symbol=line, severity='fatal')

            func_vars2 = [f.funcdef.results[0].clone(new_index_name('tmp')) for f in funcs]
            assigns   += [Assign(v, f) for v,f in zip(func_vars2, funcs)]

            if assigns:
                # For now we do not handle memory allocation in loop unravelling
                if any(v.rank > 0 for v in func_vars1) or any(v.rank > 0 for v in func_vars1):
                    errors.report("Loop unravelling cannot handle extraction of function calls \
                            which return arrays as this requires allocation. Please place the function \
                            call on its own line",
                            symbol=line, severity='fatal')
                line.substitute(internal_funcs, func_vars1, excluded_nodes=(FunctionCall))
                line.substitute(funcs, func_vars2)
                tmp_vars.extend(func_vars1)
                tmp_vars.extend(func_vars2)
                result.extend(assigns)
                current_level = 0

            rank = line.lhs.rank
            shape = line.lhs.shape
            new_vars = variables
            # Loop over indexes, inserting until the expression can be evaluated
            # in the desired language
            new_level = 0
            for index in range(-rank,0):
                new_level += 1
                # If an index exists at the same depth, reuse it if not create one
                if rank+index >= len(indices):
                    indices.append(Variable('int',new_index_name('i')))
                index_var = indices[rank+index]
                new_vars = [insert_index(v, index, index_var) for v in new_vars]
                if compatible_operation(*new_vars, language_has_vectors = language_has_vectors):
                    break

            # TODO [NH]: get all indices when adding axis argument to linspace function
            if isinstance(line.rhs, NumpyLinspace):
                line.rhs.ind = indices[0]

            # Replace variable expressions with Indexed versions
            line.substitute(variables, new_vars, excluded_nodes = (FunctionCall, PyccelInternalFunction))
            _ = [f.substitute(variables, new_vars, excluded_nodes = (FunctionDef)) for f in elemental_func_calls]

            # Recurse through result tree to save line with lines which need
            # the same set of for loops
            save_spot = result
            j = 0
            for _ in range(min(new_level,current_level)):
                # Select the existing loop if the shape matches the shape of the expression
                # and the loop is not used to modify one of the variable dependencies
                if save_spot[-1].length == shape[j] and not any(u in save_spot[-1].modified_vars for u in dependencies):
                    save_spot[-1].modified_vars.update(lhs_vars)
                    save_spot = save_spot[-1].body
                    j+=1
                else:
                    break

            for k in range(j,new_level):
                # Create new loops until we have the neccesary depth
                save_spot.append(LoopCollection([], shape[k], set(lhs_vars)))
                save_spot = save_spot[-1].body

            # Save results
            save_spot.append(line)
            current_level = new_level

        elif isinstance(line, Assign) and isinstance(line.rhs, Concatenate):
            lhs = line.lhs
            rhs = line.rhs
            arg1, arg2 = rhs.args
            assign1 = Assign(lhs[Slice(LiteralInteger(0), arg1.shape[0])], arg1)
            assign2 = Assign(lhs[Slice(arg1.shape[0], PyccelAdd(arg1.shape[0], arg2.shape[0], simplify=True))], arg2)
            collect_loops([assign1, assign2], indices, new_index_name, tmp_vars, language_has_vectors, result = result)

        elif isinstance(line, Assign) and isinstance(line.rhs, Duplicate):
            lhs = line.lhs
            rhs = line.rhs

            if not isinstance(rhs.length, LiteralInteger):
                if len(indices) == 0:
                    indices.append(Variable('int',new_index_name('i')))
                idx = indices[0]

                assign = Assign(lhs[Slice(PyccelMul(rhs.val.shape[0], idx, simplify=True),
                                          PyccelMul(rhs.val.shape[0],
                                                    PyccelAdd(idx, LiteralInteger(1), simplify=True),
                                                    simplify=True))],
                                rhs.val)

                tmp_indices = indices[1:]

                block = collect_loops([assign], tmp_indices, new_index_name, tmp_vars, language_has_vectors)
                if len(tmp_indices)>len(indices)-1:
                    indices.extend(tmp_indices[len(indices)-1:])

                result.append(LoopCollection([block[-1]],  rhs.val.shape[0], set([lhs])))

            else:
                assigns = [Assign(lhs[Slice(PyccelMul(rhs.val.shape[0], LiteralInteger(idx), simplify=True),
                                          PyccelMul(rhs.val.shape[0],
                                              PyccelAdd(LiteralInteger(idx), LiteralInteger(1), simplify=True),
                                              simplify=True))],
                                rhs.val) for idx in range(rhs.length)]
                collect_loops(assigns, indices, new_index_name, tmp_vars, language_has_vectors, result = result)

        else:
            # Save line in top level (no for loop)
            result.append(line)
            current_level = 0

    return result

#==============================================================================

def insert_fors(blocks, indices, level = 0):
    """
    Run through the output of collect_loops and create For loops of the
    requested sizes

    Parameters
    ==========
    block   : list of LoopCollection
            The result of a call to collect_loops
    indices : list
            The index variables
    level   : int
            The index of the index variable used in the outermost loop
    Results
    =======
    block : list of PyccelAstNodes
            The modified expression
    """
    if all(not isinstance(b, LoopCollection) for b in blocks.body):
        body = blocks.body
    else:
        body = [insert_fors(b, indices, level+1) if isinstance(b, LoopCollection) else [b] \
                for b in blocks.body]
        body = [bi for b in body for bi in b]
    if blocks.length == 1:
        return body
    else:
        body = CodeBlock(body, unravelled = True)
        return [For(indices[level], PythonRange(0,blocks.length), body)]

#==============================================================================
def expand_tuple_assignments(block):
    """
    Simplify expressions in a CodeBlock by unravelling tuple assignments into multiple lines

    Parameters
    ==========
    block      : CodeBlock
                The expression to be modified

    Results
    =======
    list : The contents of a replacement CodeBlock

    Examples
    --------
    >>> from pyccel.ast.builtins  import PythonTuple
    >>> from pyccel.ast.core      import Assign, CodeBlock
    >>> from pyccel.ast.literals  import LiteralInteger
    >>> from pyccel.ast.utilities import expand_to_loops
    >>> from pyccel.ast.variable  import Variable
    >>> a = Variable('int', 'a', shape=(,), rank=0)
    >>> b = Variable('int', 'b', shape=(,), rank=0)
    >>> c = Variable('int', 'c', shape=(,), rank=0)
    >>> expr = [Assign(PythonTuple(a,b,c),PythonTuple(LiteralInteger(0),LiteralInteger(1),LiteralInteger(2))]
    >>> expand_tuple_assignments(CodeBlock(expr))
    [Assign(a, LiteralInteger(0)), Assign(b, LiteralInteger(1)), Assign(c, LiteralInteger(2))]
    """
    assigns = [a for a in block.get_attribute_nodes(Assign) \
                if isinstance(a.lhs, InhomogeneousTupleVariable) \
                and isinstance(a.rhs, (PythonTuple, InhomogeneousTupleVariable))]
    if len(assigns) == 0:
        return
    else:
        new_assigns = [[Assign(l,r) for l,r in zip(a.lhs, a.rhs)] for a in assigns]
        block.substitute(assigns, new_assigns)
        expand_tuple_assignments(block)

#==============================================================================
def expand_to_loops(block, new_index_name, language_has_vectors = False):
    """
    Re-write a list of expressions to include explicit loops where necessary

    Parameters
    ==========
    block          : CodeBlock
                     The expressions to be modified
    new_index_name : function
                     A function which provides a new variable name from a base name,
                     avoiding name collisions
    language_has_vectors : bool
                     Indicates if the language has support for vector
                     operations of the same shape

    Returns
    =======
    expr        : list of Ast Nodes
                The expressions with For loops inserted where necessary

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.operators import PyccelAdd
    >>> from pyccel.ast.utilities import expand_to_loops
    >>> a = Variable('int', 'a', shape=(4,), rank=1)
    >>> b = Variable('int', 'b', shape=(4,), rank=1)
    >>> c = Variable('int', 'c', shape=(4,), rank=1)
    >>> i = Variable('int', 'i', shape=())
    >>> d = PyccelAdd(a,b)
    >>> expr = [Assign(c,d)]
    >>> expand_to_loops(expr, language_has_vectors = False)
    [For(i_0, PythonRange(0, LiteralInteger(4), LiteralInteger(1)), CodeBlock([IndexedElement(c, i_0) := PyccelAdd(IndexedElement(a, i_0), IndexedElement(b, i_0))]), [])]
    """
    expand_tuple_assignments(block)
    indices = []
    tmp_vars = []
    res = collect_loops(block.body, indices, new_index_name, tmp_vars, language_has_vectors)

    body = [insert_fors(b, indices) if isinstance(b, tuple) else [b] for b in res]
    body = [bi for b in body for bi in b]

    return body, indices+tmp_vars

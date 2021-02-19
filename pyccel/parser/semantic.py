# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

# pylint: disable=R0201, missing-function-docstring

from collections import OrderedDict
from itertools import chain

from sympy.utilities.iterables import iterable as sympy_iterable

from sympy import Sum as Summation
from sympy import Symbol as sp_Symbol
from sympy import Integer as sp_Integer
from sympy import ceiling
from sympy import oo  as INF
from sympy.core import cache

#==============================================================================

from pyccel.ast.basic import Basic, PyccelAstNode

from pyccel.ast.core import If, IfSection
from pyccel.ast.core import Allocate, Deallocate
from pyccel.ast.core import Assign, AliasAssign, SymbolicAssign
from pyccel.ast.core import AugAssign, CodeBlock
from pyccel.ast.core import Return, Argument
from pyccel.ast.core import ConstructorCall
from pyccel.ast.core import ValuedFunctionAddress
from pyccel.ast.core import FunctionDef, Interface, FunctionAddress, FunctionCall
from pyccel.ast.core import DottedFunctionCall
from pyccel.ast.core import ClassDef
from pyccel.ast.core import For, FunctionalFor, ForIterator
from pyccel.ast.core import While
from pyccel.ast.core import SymbolicPrint
from pyccel.ast.core import Del
from pyccel.ast.core import EmptyNode
from pyccel.ast.variable import Constant
from pyccel.ast.variable import Variable
from pyccel.ast.variable import TupleVariable
from pyccel.ast.variable import IndexedElement
from pyccel.ast.variable import DottedName, DottedVariable
from pyccel.ast.variable import ValuedVariable
from pyccel.ast.core import ValuedArgument
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.core import With
from pyccel.ast.builtins import PythonList
from pyccel.ast.core import Dlist
from pyccel.ast.core import StarredArguments
from pyccel.ast.core import get_assigned_symbols
from pyccel.ast.operators import PyccelIs, PyccelIsNot, IfTernaryOperator
from pyccel.ast.itertoolsext import Product

from pyccel.ast.functionalexpr import FunctionalSum, FunctionalMax, FunctionalMin

from pyccel.ast.datatypes import NativeRange, str_dtype
from pyccel.ast.datatypes import NativeSymbol
from pyccel.ast.datatypes import DataTypeFactory
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeReal, NativeString, NativeGeneric, NativeComplex

from pyccel.ast.literals import LiteralTrue, LiteralFalse
from pyccel.ast.literals import LiteralInteger, LiteralFloat
from pyccel.ast.literals import Nil

from pyccel.ast.headers import FunctionHeader, ClassHeader, MethodHeader
from pyccel.ast.headers import MacroFunction, MacroVariable

from pyccel.ast.utilities import builtin_function as pyccel_builtin_function
from pyccel.ast.utilities import builtin_import as pyccel_builtin_import
from pyccel.ast.utilities import builtin_import_registery as pyccel_builtin_import_registery
from pyccel.ast.utilities import split_positional_keyword_arguments

from pyccel.ast.builtins import PythonPrint
from pyccel.ast.builtins import PythonInt, PythonBool, PythonFloat, PythonComplex
from pyccel.ast.builtins import python_builtin_datatype
from pyccel.ast.builtins import (PythonRange, PythonZip, PythonEnumerate,
                                 PythonMap, PythonTuple, Lambda)

from pyccel.ast.numpyext import NumpyZeros
from pyccel.ast.numpyext import NumpyInt, NumpyInt32, NumpyInt64
from pyccel.ast.numpyext import NumpyFloat, NumpyFloat32, NumpyFloat64
from pyccel.ast.numpyext import NumpyComplex, NumpyComplex64, NumpyComplex128
from pyccel.ast.numpyext import NumpyArrayClass, NumpyNewArray

from pyccel.ast.internals import Slice, PyccelSymbol

from pyccel.ast.sympy_helper import sympy_to_pyccel, pyccel_to_sympy

from pyccel.errors.errors import Errors
from pyccel.errors.errors import PyccelSemanticError

# TODO - remove import * and only import what we need
#      - use OrderedDict whenever it is possible
from pyccel.errors.messages import *

from pyccel.parser.base      import BasicParser, Scope
from pyccel.parser.base      import get_filename_from_import
from pyccel.parser.syntactic import SyntaxParser

import pyccel.decorators as def_decorators
#==============================================================================

errors = Errors()

#==============================================================================

def _get_name(var):
    """."""

    if isinstance(var, (PyccelSymbol, DottedName)):
        return str(var)
    if isinstance(var, (IndexedElement)):
        return str(var.base)
    if isinstance(var, FunctionCall):
        return var.funcdef
    if isinstance(var, AsName):
        return var.target
    msg = 'Name of Object : {} cannot be determined'.format(type(var).__name__)
    errors.report(PYCCEL_RESTRICTION_TODO+'\n'+msg, symbol=var,
                severity='fatal')

#==============================================================================

class SemanticParser(BasicParser):

    """ Class for a Semantic Parser.
    It takes a syntactic parser as input for the moment"""

    def __init__(self, inputs, **kwargs):

        # a Parser can have parents, who are importing it.
        # imports are then its sons.
        self._parents = kwargs.pop('parents', [])
        self._d_parsers = kwargs.pop('d_parsers', OrderedDict())

        # ...
        if not isinstance(inputs, SyntaxParser):
            raise TypeError('> Expecting a syntactic parser as input')

        parser = inputs
        # ...

        # ...
        BasicParser.__init__(self, **kwargs)
        # ...

        # ...
        self._fst = parser._fst
        self._ast = parser._ast

        self._filename  = parser._filename
        self._metavars  = parser._metavars
        self._namespace = parser._namespace
        self._namespace.imports['imports'] = OrderedDict()
        self._used_names = parser.used_names
        self._dummy_counter = parser._dummy_counter

        # used to store the local variables of a code block needed for garbage collecting
        self._allocs = []

        # we use it to detect the current method or function

        #
        self._code = parser._code
        # ...

        # ... TOD add settings
        settings = {}
        self.annotate()
        # ...

    @property
    def parents(self):
        """Returns the parents parser."""
        return self._parents

    @property
    def d_parsers(self):
        """Returns the d_parsers parser."""

        return self._d_parsers

    def annotate(self, **settings):
        """."""

        if self.semantic_done:
            print ('> semantic analysis already done')
            return self.ast

        # TODO - add settings to Errors
        #      - filename

        errors = Errors()
        if self.filename:
            errors.set_target(self.filename, 'file')
        errors.set_parser_stage('semantic')

        # then we treat the current file

        ast = self.ast

        self._allocs.append([])
        # we add the try/except to allow the parser to find all possible errors
        PyccelAstNode.stage = 'semantic'
        ast = self._visit(ast, **settings)

        self._ast = ast

        # in the case of a header file, we need to convert all headers to
        # FunctionDef etc ...

        if self.is_header_file:
            target = []

            for parent in self.parents:
                for (key, item) in parent.imports.items():
                    if get_filename_from_import(key) == self.filename:
                        target += item

            target = set(target)
            target_headers = target.intersection(self.namespace.headers.keys())
            for name in list(target_headers):
                v = self.namespace.headers[name][0]
                if isinstance(v, FunctionHeader) and not isinstance(v,
                        MethodHeader):
                    F = self.get_function(name)
                    if F is None:
                        interfaces = v.create_definition()
                        for F in interfaces:
                            self.insert_function(F)
                    else:
                        errors.report(IMPORTING_EXISTING_IDENTIFIED,
                                symbol=name, blocker=True,
                                severity='fatal')

        self._semantic_done = True

        # Calling the Garbage collecting,
        # it will add the necessary Deallocate nodes
        # to the ast
        self._ast = ast = self.garbage_collector(ast)

        return ast

    def garbage_collector(self, expr):
        """
        Search in a CodeBlock if no trailing Return Node is present add the needed frees.

        Return the same CodeBlock if a trailing Return is found otherwise Return a new CodeBlock with additional Deallocate Nodes.
        """
        code = expr
        if len(expr.body)>0 and not isinstance(expr.body[-1], Return):
            code = expr.body + tuple(Deallocate(i) for i in self._allocs[-1])
            code = CodeBlock(code)
        self._allocs.pop()
        return code

    def get_variable_from_scope(self, name):
        """
        Search for a Variable object with the given name inside the local Python scope.
        If not found, return None.
        """
        # Walk up nested loops (if any)
        container = self.namespace
        while container.is_loop:
            container = container.parent_scope

        var = self._get_variable_from_scope(name, container)

        return var

    def _get_variable_from_scope(self, name, container):
        """
        Search for a Variable object with the given name in the given Python scope.
        This is a recursive function because it searches inside nested loops, where
        OpenMP variables could be defined.
        """
        if name in container.variables:
            return container.variables[name]

        if name in container.imports['variables']:
            return container.imports['variables'][name]

        # Search downwards, walking down the tree of nested loop Scopes
        for container in container.loops:
            var = self._get_variable_from_scope(name, container)
            if var:
                return var

        return None

    def check_for_variable(self, name):
        """
        Search for a Variable object with the given name in the current namespace,
        defined by the local and global Python scopes. Return None if not found.
        """

        if self.current_class:
            for i in self._current_class.attributes:
                if i.name == name:
                    var = i
                    return var

        # Walk up nested loops (if any)
        container = self.namespace
        while container.is_loop:
            container = container.parent_scope

        # Walk up the tree of Scope objects, until the root if needed
        while container:
            var = self._get_variable_from_scope(name, container)
            if var is not None:
                return var
            container = container.parent_scope

        return None

    def get_variable(self, name):
        """ Like 'check_for_variable', but raise Pyccel error if Variable is not found.
        """
        var = self.check_for_variable(name)
        if var is None:
            errors.report(UNDEFINED_VARIABLE, symbol=name,
            bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
            severity='fatal', blocker=True)
        else:
            return var

    def get_variables(self, container):
        # this only works if called on a function scope
        # TODO needs more tests when we have nested functions
        variables = []
        variables.extend(container.variables.values())
        for container in container.loops:
            variables.extend(self.get_variables(container))
        return variables


    def get_parent_functions(self):
        container = self.namespace
        funcs = container.functions.copy()
        container = container.parent_scope
        while container:
            for i in container.functions:
                if not i in funcs:
                    funcs[i] = container.functions[i]
            container = container.parent_scope
        return funcs


    def get_class(self, name):
        """."""

        container = self.namespace

        while container:
            if name in container.classes:
                return container.classes[name]
            elif name in container.imports['classes']:
                return container.imports['classes'][name]

            container = container.parent_scope
        return None

    def insert_variable(self, var, name=None):
        """."""

        # TODO add some checks before
        if not isinstance(var, Variable):
            raise TypeError('variable must be of type Variable')

        if name is None:
            name = var.name

        self.namespace.variables[name] = var


    def insert_class(self, cls, parent=False):
        """."""

        if isinstance(cls, ClassDef):
            name = cls.name
            container = self.namespace
            if parent:
                container = container.parent_scope
            container.classes[name] = cls
        else:
            raise TypeError('Expected A class definition ')

    def insert_template(self, expr):
        """append the scope's templates with the given template"""
        self.namespace.templates[expr.name] = expr

    def insert_header(self, expr):
        """."""
        if isinstance(expr, (FunctionHeader, MethodHeader)):
            if expr.name in self.namespace.headers:
                self.namespace.headers[expr.name].append(expr)
            else:
                self.namespace.headers[expr.name] = [expr]
        elif isinstance(expr, ClassHeader):
            self.namespace.headers[expr.name] = expr

            #  create a new Datatype for the current class

            iterable = 'iterable' in expr.options
            with_construct = 'with' in expr.options
            dtype = DataTypeFactory(expr.name, '_name',
                                    is_iterable=iterable,
                                    is_with_construct=with_construct)
            self.set_class_construct(expr.name, dtype)
        else:
            msg = 'header of type{0} is not supported'
            msg = msg.format(str(type(expr)))
            raise TypeError(msg)

    def get_function(self, name):
        """."""

        # TODO shall we keep the elif in _imports?

        func = None

        container = self.namespace
        while container:
            if name in container.functions:
                func = container.functions[name]
                break

            if name in container.imports['functions']:
                func =  container.imports['functions'][name]
                break
            container = container.parent_scope

        return func


    def get_import(self, name):
        """
        Search for an import with the given name in the current namespace.
        Return None if not found.
        """

        imp = None

        container = self.namespace
        while container:

            if name in container.imports['imports']:
                imp =  container.imports['imports'][name]
                break
            container = container.parent_scope


        return imp


    def get_symbolic_function(self, name):
        """."""

        # TODO shall we keep the elif in _imports?
        container = self.namespace
        while container:
            if name in container.symbolic_functions:
                return container.symbolic_functions[name]

            if name in container.imports['symbolic_functions']:
                return container.imports['symbolic_functions'][name]
            container = container.parent_scope

        return None

    def get_python_function(self, name):
        """."""

        # TODO shall we keep the elif in _imports?
        container = self.namespace
        while container:
            if name in container.python_functions:
                return container.python_functions[name]

            if name in container.imports['python_functions']:
                return container.imports['python_functions'][name]

            container = container.parent_scope

        return None

    def get_macro(self, name):
        """."""

        # TODO shall we keep the elif in _imports?

        container = self.namespace
        while container:
            if name in container.macros:
                return container.macros[name]
            container = container.parent_scope

        return None

    def insert_import(self, name, target):
        """
            Create and insert a new import in namespace if it's not defined
            otherwise append target to existing import.
        """
        imp = self.get_import(name)

        if imp is not None:
            imp.define_target(target)
        else:
            container = self.namespace.imports
            container['imports'][name] = Import(name, target, True)

    def insert_macro(self, macro):
        """."""

        container = self.namespace.macros

        if isinstance(macro, (MacroFunction, MacroVariable)):
            name = macro.name
            if isinstance(macro.name, DottedName):
                name = name.name[-1]
            container[name] = macro
        else:
            raise TypeError('Expected a macro')

    def remove_variable(self, name):
        """."""

        container = self.namespace
        while container:
            if name in container.variables:
                container.pop(name)
                break
            container = container.parent_scope

    def get_header(self, name):
        """."""
        container = self.namespace
        headers = []
        while container:
            if name in container.headers:
                if isinstance(container.headers[name], list):
                    headers += container.headers[name]
                else:
                    headers.append(container.headers[name])
            container = container.parent_scope
        return headers

    def get_templates(self):
        """Returns templates of the current scope and all its parents scopes"""
        container = self.namespace
        templates = {}
        while container:
            templates.update({tmplt:container.templates[tmplt] for tmplt in container.templates\
                if tmplt not in templates})
            container = container.parent_scope
        return templates

    def find_class_construct(self, name):
        """Returns the class datatype for name if it exists.
        Returns None otherwise
        """
        container = self.namespace
        while container:
            if name in container.cls_constructs:
                return container.cls_constructs[name]
            container = container.parent_scope
        return None

    def get_class_construct(self, name):
        """Returns the class datatype for name if it exists.
        Raises an error otherwise
        """
        result = self.find_class_construct(name)

        if result:
            return result
        else:
            msg = 'class construct {} not found'.format(name)
            return errors.report(msg,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='fatal', blocker=self.blocking)


    def set_class_construct(self, name, value):
        """Sets the class datatype for name."""

        self.namespace.cls_constructs[name] = value

    def create_new_function_scope(self, name, decorators):
        """
        Create a new Scope object for a Python function with the given name,
        and attach any decorators' information to the scope. The new scope is
        a child of the current one, and can be accessed from the dictionary of
        its children using the function name as key.

        Before returning control to the caller, the current scope (stored in
        self._namespace) is changed to the one just created, and the function's
        name is stored in self._current_function.

        Parameters
        ----------
        name : str
            Function's name, used as a key to retrieve the new scope.

        decorators : dict
            Decorators attached to FunctionDef object at syntactic stage.

        """
        child = self.namespace.new_child_scope(name, decorators=decorators)

        self._namespace = child
        if self._current_function:
            name = DottedName(self._current_function, name)
        self._current_function = name

    def exit_function_scope(self):

        self._namespace = self._namespace.parent_scope
        if isinstance(self._current_function, DottedName):

            name = self._current_function.name[:-1]
            if len(name)>1:
                name = DottedName(*name)
            else:
                name = name[0]
        else:
            name = None
        self._current_function = name

    def create_new_loop_scope(self):
        new_scope = Scope(decorators=self._namespace.decorators)
        new_scope._is_loop = True
        new_scope.parent_scope = self._namespace
        self._namespace._loops.append(new_scope)
        self._namespace = new_scope

    def exit_loop_scope(self):
        self._namespace = self._namespace.parent_scope

#==============================================================================

    def _infere_type(self, expr, **settings):
        """
        type inference for expressions
        """

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors

        errors = Errors()

        verbose = settings.pop('verbose', False)
        if verbose:
            print ('*** type inference for : ', type(expr))

        d_var = {}
        # TODO improve => put settings as attribut of Parser

        if expr in (PythonInt, PythonFloat, PythonComplex, PythonBool, NumpyInt,
                      NumpyInt32, NumpyInt64, NumpyComplex, NumpyComplex64,
					  NumpyComplex128, NumpyFloat, NumpyFloat64, NumpyFloat32):

            d_var['datatype'   ] = '*'
            d_var['rank'       ] = 0
            d_var['precision'  ] = 0
            return d_var

        elif isinstance(expr, Variable):

            d_var['datatype'      ] = expr.dtype
            d_var['allocatable'   ] = expr.allocatable
            d_var['shape'         ] = expr.shape
            d_var['rank'          ] = expr.rank
            d_var['cls_base'      ] = expr.cls_base
            d_var['is_pointer'    ] = expr.is_pointer
            d_var['is_target'     ] = expr.is_target
            d_var['order'         ] = expr.order
            d_var['precision'     ] = expr.precision
            return d_var

        elif isinstance(expr, PythonTuple):
            d_var['datatype'      ] = expr.dtype
            d_var['precision']      = expr.precision
            d_var['is_stack_array'] = expr.is_homogeneous
            d_var['shape'         ] = expr.shape
            d_var['rank'          ] = expr.rank
            d_var['is_pointer']     = False

            return d_var

        elif isinstance(expr, Dlist):
            d = self._infere_type(expr.val, **settings)

            # TODO must check that it is consistent with pyccel's rules
            # TODO improve
            d_var['datatype'   ] = d['datatype']
            d_var['rank'       ] = expr.rank
            d_var['shape'      ] = expr.shape
            d_var['allocatable'] = False
            d_var['is_pointer' ] = True
            return d_var

        elif isinstance(expr, NumpyNewArray):
            d_var['datatype'   ] = expr.dtype
            d_var['allocatable'] = expr.rank>0
            d_var['shape'      ] = expr.shape
            d_var['rank'       ] = expr.rank
            d_var['order'      ] = expr.order
            d_var['precision'  ] = expr.precision
            d_var['cls_base'   ] = NumpyArrayClass
            return d_var

        elif isinstance(expr, PyccelAstNode):

            d_var['datatype'   ] = expr.dtype
            d_var['allocatable'] = expr.rank>0
            d_var['shape'      ] = expr.shape
            d_var['rank'       ] = expr.rank
            d_var['order'      ] = expr.order
            d_var['precision'  ] = expr.precision
            return d_var

        elif isinstance(expr, IfTernaryOperator):
            return self._infere_type(expr.args[0][1].body[0])

        elif isinstance(expr, PythonRange):

            d_var['datatype'   ] = NativeRange()
            d_var['allocatable'] = False
            d_var['shape'      ] = ()
            d_var['rank'       ] = 0
            d_var['cls_base'   ] = expr  # TODO: shall we keep it?
            return d_var

        elif isinstance(expr, Lambda):

            d_var['datatype'   ] = NativeSymbol()
            d_var['allocatable'] = False
            d_var['is_pointer' ] = False
            d_var['rank'       ] = 0
            return d_var

        elif isinstance(expr, ConstructorCall):
            cls_name = expr.func.cls_name
            cls = self.get_class(cls_name)

            dtype = self.get_class_construct(cls_name)()

            d_var['datatype'   ] = dtype
            d_var['allocatable'] = False
            d_var['shape'      ] = ()
            d_var['rank'       ] = 0
            d_var['is_target'  ] = False

            # set target  to True if we want the class objects to be pointers

            d_var['cls_base'      ] = cls
            return d_var

        else:
            msg = 'Type of Object : {} cannot be infered'.format(type(expr).__name__)
            errors.report(PYCCEL_RESTRICTION_TODO+'\n'+msg, symbol=expr,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='fatal', blocker=self.blocking)


#==============================================================================
#==============================================================================
#==============================================================================


    def _visit(self, expr, **settings):
        """Annotates the AST.

        The annotation is done by finding the appropriate function _visit_X
        for the object expr. X is the type of the object expr. If this function
        does not exist then the method resolution order is used to search for
        other compatible _visit_X functions. If none are found then an error is
        raised
        """

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors
        current_fst = self._current_fst_node

        if hasattr(expr,'fst') and expr.fst is not None:
            self._current_fst_node = expr.fst

        classes = type(expr).__mro__
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                obj = getattr(self, annotation_method)(expr, **settings)
                if isinstance(obj, Basic) and self._current_fst_node:
                    obj.set_fst(self._current_fst_node)
                self._current_fst_node = current_fst
                return obj

        # Unknown object, we raise an error.
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=type(expr),
            bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
            severity='fatal', blocker=self.blocking)

    def _visit_tuple(self, expr, **settings):
        return tuple(self._visit(i, **settings) for i in expr)

    def _visit_PythonTuple(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        return PythonTuple(*ls)

    def _visit_PythonList(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        expr = PythonList(*ls)

        if not expr.is_homogeneous:
            errors.report(PYCCEL_RESTRICTION_INHOMOG_LIST, symbol=expr,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='fatal')
        return expr

    def _visit_ValuedArgument(self, expr, **settings):
        value = self._visit(expr.value, **settings)
        d_var      = self._infere_type(value, **settings)
        dtype      = d_var.pop('datatype')
        return ValuedVariable(dtype, expr.name,
                               value=value, **d_var)

    def _visit_CodeBlock(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr.body]
        ls = [line for l in ls for line in (l.body if isinstance(l, CodeBlock) else [l])]
        return CodeBlock(ls)

    def _visit_Nil(self, expr, **settings):
        return expr
    def _visit_EmptyNode(self, expr, **settings):
        return expr
    def _visit_Break(self, expr, **settings):
        return expr
    def _visit_Continue(self, expr, **settings):
        return expr
    def _visit_Comment(self, expr, **settings):
        return expr
    def _visit_CommentBlock(self, expr, **settings):
        return expr
    def _visit_AnnotatedComment(self, expr, **settings):
        return expr
    def _visit_Literal(self, expr, **settings):
        return expr
    def _visit_Integer(self, expr, **settings):
        """Visit sympy.Integer"""
        return LiteralInteger(expr.p)
    def _visit_Float(self, expr, **settings):
        """Visit sympy.Integer"""
        return LiteralFloat(expr)
    def _visit_PythonComplex(self, expr, **settings):
        return expr
    def _visit_Pass(self, expr, **settings):
        return expr

    def _visit_NumberSymbol(self, expr, **settings):
        return expr.n()

    def _visit_Number(self, expr, **settings):
        return expr.n()

    def _visit_Variable(self, expr, **settings):
        name = expr.name
        return self.get_variable(name)

    def _visit_str(self, expr, **settings):
        return repr(expr)

    def _visit_Slice(self, expr, **settings):
        start = self._visit(expr.start) if expr.start is not None else None
        stop = self._visit(expr.stop) if expr.stop is not None else None
        step = self._visit(expr.step) if expr.step is not None else None

        return Slice(start, stop, step)

    def _extract_indexed_from_var(self, var, args, name):

        # case of Pyccel ast Variable
        # if not possible we use symbolic objects

        if not isinstance(var, Variable):
            assert(hasattr(var,'__getitem__'))
            if len(args)==1:
                return var[args[0]]
            else:
                return self._visit(var[args[0]][args[1:]])

        args = tuple(args)

        if isinstance(var, TupleVariable) and not var.is_homogeneous:

            arg = args[0]

            if isinstance(arg, Slice):
                if ((arg.start is not None and not isinstance(arg.start, LiteralInteger)) or
                        (arg.stop is not None and not isinstance(arg.stop, LiteralInteger))):
                    errors.report(INDEXED_TUPLE, symbol=var,
                        bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                        severity='fatal', blocker=self.blocking)

                idx = slice(arg.start, arg.stop)
                selected_vars = var.get_var(idx)
                if len(selected_vars)==1:
                    if len(args) == 1:
                        return selected_vars[0]
                    else:
                        var = selected_vars[0]
                        return self._extract_indexed_from_var(var, args[1:], name)
                elif len(selected_vars)<1:
                    return None
                elif len(args)==1:
                    return PythonTuple(*selected_vars)
                else:
                    return PythonTuple(*[self._extract_indexed_from_var(var, args[1:], name) for var in selected_vars])

            elif isinstance(arg, LiteralInteger):

                if len(args)==1:
                    return var[arg]

                var = var[arg]
                return self._extract_indexed_from_var(var, args[1:], name)

            else:
                errors.report(INDEXED_TUPLE, symbol=var,
                    bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                    severity='fatal', blocker=self.blocking)

        if isinstance(var, PythonTuple) and not var.is_homogeneous:
            errors.report(LIST_OF_TUPLES, symbol=var,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='error', blocker=self.blocking)

        for arg in var[args].indices:
            if not isinstance(arg, Slice) and not \
                (hasattr(arg, 'dtype') and isinstance(arg.dtype, NativeInteger)):
                errors.report(INVALID_INDICES, symbol=var[args],
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='error', blocker=self.blocking)
        return var[args]

    def _visit_IndexedElement(self, expr, **settings):
        name = str(expr.base)
        var = self._visit(expr.base)

         # TODO check consistency of indices with shape/rank

        args = list(expr.indices)

        new_args = [self._visit(arg, **settings) for arg in args]

        if (len(new_args)==1 and isinstance(new_args[0],(TupleVariable, PythonTuple))):
            len_args = len(new_args[0])
            args = [new_args[0][i] for i in range(len_args)]
        elif any(isinstance(arg,(TupleVariable, PythonTuple)) for arg in new_args):
            n_exprs = None
            for a in new_args:
                if hasattr(a,'__len__'):
                    if n_exprs:
                        assert(n_exprs)==len(a)
                    else:
                        n_exprs = len(a)
            new_expr_args = []
            for i in range(n_exprs):
                ls = []
                for j,a in enumerate(new_args):
                    if hasattr(a,'__getitem__'):
                        ls.append(args[j][i])
                    else:
                        ls.append(args[j])
                new_expr_args.append(ls)

            return tuple(var[a] for a in new_expr_args)
        else:
            args = new_args
            len_args = len(args)

        return self._extract_indexed_from_var(var, args, name)

    def _visit_PyccelSymbol(self, expr, **settings):
        name = expr

        var = self.check_for_variable(name)

        if var is None:
            var = self.get_function(name)
        if var is None:
            var = self.get_symbolic_function(name)
        if var is None:
            var = python_builtin_datatype(name)

        if var is None:

            errors.report(UNDEFINED_VARIABLE, symbol=name,
            bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
            severity='fatal', blocker=True)
        return var


    def _visit_DottedName(self, expr, **settings):

        var = self.check_for_variable(_get_name(expr))
        if var:
            return var

        lhs = expr.name[0] if len(expr.name) == 2 \
                else DottedName(*expr.name[:-1])
        rhs = expr.name[-1]

        visited_lhs = self._visit(lhs)
        first = visited_lhs
        if isinstance(visited_lhs, FunctionCall):
            results = visited_lhs.funcdef.results
            if len(results) != 1:
                errors.report("Cannot get attribute of function call with multiple returns",
                        symbol=expr,
                        bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                        severity='fatal', blocker=True)
            first = results[0]
        rhs_name = _get_name(rhs)
        attr_name = []

        # Handle case of imported module
        if isinstance(first, dict):

            if rhs_name in first:
                imp = self.get_import(_get_name(lhs))

                new_name = rhs_name
                # If pyccelized file
                if imp is not None:
                    new_name = imp.find_module_target(rhs_name)
                    if new_name is None:
                        new_name = self.get_new_name(rhs_name)

                        # Save the import target that has been used
                        if new_name == rhs_name:
                            imp.define_target(PyccelSymbol(rhs_name))
                        else:
                            imp.define_target(AsName(PyccelSymbol(rhs_name), PyccelSymbol(new_name)))

                if isinstance(rhs, FunctionCall):
                    # If object is a function
                    args  = self._handle_function_args(rhs.args, **settings)
                    func  = first[rhs_name]
                    if new_name != rhs_name:
                        if hasattr(func, 'clone'):
                            func  = func.clone(new_name)
                    return self._handle_function(func, args, **settings)
                elif isinstance(rhs, Constant):
                    var = first[rhs_name]
                    if new_name != rhs_name:
                        var.name = new_name
                    return var
                else:
                    # If object is something else (eg. dict)
                    var = first[rhs_name]
                    return var
            else:
                errors.report(UNDEFINED_IMPORT_OBJECT.format(rhs_name, str(lhs)),
                        symbol=expr,
                        bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                        severity='fatal', blocker=True)

        if not hasattr(first, 'cls_base') or first.cls_base is None:
            errors.report('Attribute {} not found'.format(rhs_name),
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='fatal', blocker=True)

        if first.cls_base:
            attr_name = [i.name for i in first.cls_base.attributes]

        # look for a class method
        if isinstance(rhs, FunctionCall):
            methods = list(first.cls_base.methods) + list(first.cls_base.interfaces)
            for method in methods:
                if isinstance(method, Interface):
                    errors.report('Generic methods are not supported yet',
                        symbol=method.name,
                        bounding_box=(self._current_fst_node.lineno,
                            self._current_fst_node.col_offset),
                        severity='fatal')
            macro = self.get_macro(rhs_name)
            if macro is not None:
                master = macro.master
                name = macro.name
                args = rhs.args
                args = [lhs] + list(args)
                args = [self._visit(i, **settings) for i in args]
                args = macro.apply(args)
                return FunctionCall(master, args, self._current_function)

            args = [self._visit(arg, **settings) for arg in
                    rhs.args]
            for i in methods:
                if str(i.name) == rhs_name:
                    if 'numpy_wrapper' in i.decorators.keys():
                        func = i.decorators['numpy_wrapper']
                        return func(visited_lhs, *args)
                    else:
                        return DottedFunctionCall(i, args, prefix = visited_lhs,
                                    current_function = self._current_function)

        # look for a class attribute / property
        elif isinstance(rhs, PyccelSymbol) and first.cls_base:
            methods = list(first.cls_base.methods) + list(first.cls_base.interfaces)
            for method in methods:
                if isinstance(method, Interface):
                    errors.report('Generic methods are not supported yet',
                        symbol=method.name,
                        bounding_box=(self._current_fst_node.lineno,
                            self._current_fst_node.col_offset),
                        severity='fatal')
            # standard class attribute
            if rhs in attr_name:
                self._current_class = first.cls_base
                second = self._visit(rhs, **settings)
                self._current_class = None
                return second.clone(second.name, new_class = DottedVariable, lhs = visited_lhs)

            # class property?
            else:
                for i in methods:
                    if i.name == rhs and \
                            'property' in i.decorators.keys():
                        if 'numpy_wrapper' in i.decorators.keys():
                            func = i.decorators['numpy_wrapper']
                            self.insert_import('numpy', rhs)
                            return func(visited_lhs)
                        else:
                            return DottedFunctionCall(i, [], prefix = visited_lhs,
                                    current_function = self._current_function)

        # look for a macro
        else:

            macro = self.get_macro(rhs_name)

            # Macro
            if isinstance(macro, MacroVariable):
                return macro.master
            elif isinstance(macro, MacroFunction):
                args = macro.apply([visited_lhs])
                return FunctionCall(macro.master, args, self._current_function)

        # did something go wrong?
        return errors.report('Attribute {} not found'.format(rhs_name),
            bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
            severity='fatal', blocker=True)

    def _visit_PyccelOperator(self, expr, **settings):
        args     = [self._visit(a, **settings) for a in expr.args]
        try:
            expr_new = type(expr)(*args)
        except PyccelSemanticError as err:
            msg = str(err)
            errors.report(msg, symbol=expr,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='fatal', blocker=True)
        #if stmts:
        #    expr_new = CodeBlock(stmts + [expr_new])
        return expr_new

    def _visit_PyccelAdd(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        if isinstance(args[0], (TupleVariable, PythonTuple, PythonList)):
            get_vars = lambda a: a.get_vars() if isinstance(a, TupleVariable) else a.args
            tuple_args = [ai for a in args for ai in get_vars(a)]
            expr_new = PythonTuple(*tuple_args)
        else:
            _ = [a.invalidate_node() for a in args]
            expr_new = self._visit_PyccelOperator(expr, **settings)
        return expr_new

    def _visit_PyccelMul(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        if isinstance(args[0], (TupleVariable, PythonTuple, PythonList)):
            expr_new = self._visit(Dlist(args[0], args[1]))
        elif isinstance(args[1], (TupleVariable, PythonTuple, PythonList)):
            expr_new = self._visit(Dlist(args[1], args[0]))
        else:
            _ = [a.invalidate_node() for a in args]
            expr_new = self._visit_PyccelOperator(expr, **settings)
        return expr_new

    def _visit_Lambda(self, expr, **settings):


        expr_names = set(map(str, expr.expr.get_attribute_nodes((PyccelSymbol, Argument))))
        var_names = map(str, expr.variables)
        missing_vars = expr_names.difference(var_names)
        if len(missing_vars) > 0:
            errors.report(UNDEFINED_LAMBDA_VARIABLE, symbol = missing_vars,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='fatal', blocker=True)
        funcs = expr.expr.get_attribute_nodes(FunctionCall)
        for func in funcs:
            name = _get_name(func)
            f = self.get_symbolic_function(name)
            if f is None:
                errors.report(UNDEFINED_LAMBDA_FUNCTION, symbol=name,
                    bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                    severity='fatal', blocker=True)
            else:

                f = f(*func.args)
                expr_new = expr.expr.subs(func, f)
                expr = Lambda(tuple(expr.variables), expr_new)
        return expr

    def _handle_function_args(self, arguments, **settings):
        args  = []
        for arg in arguments:
            a = self._visit(arg, **settings)
            if isinstance(a, StarredArguments):
                args.extend(a.args_var)
            else:
                args.append(a)
        return args

    def _handle_function(self, func, args, **settings):
        if not isinstance(func, (FunctionDef, Interface)):
            args, kwargs = split_positional_keyword_arguments(*args)
            for a in args:
                if getattr(a,'dtype',None) == 'tuple':
                    self._infere_type(a, **settings)
            for a in kwargs.values():
                if getattr(a,'dtype',None) == 'tuple':
                    self._infere_type(a, **settings)
            expr = func(*args, **kwargs)

            return expr
        else:
            expr = FunctionCall(func, args, self._current_function)
            return expr

    def _visit_FunctionCall(self, expr, **settings):
        name     = expr.funcdef

        # Check for specialised method
        annotation_method = '_visit_' + name
        if hasattr(self, annotation_method):
            return getattr(self, annotation_method)(expr, **settings)

        func     = self.get_function(name)

        args = self._handle_function_args(expr.args, **settings)

        if name == 'lambdify':
            args = self.get_symbolic_function(str(expr.args[0]))
        F = pyccel_builtin_function(expr, args)

        if F is not None:
            return F

        elif self.find_class_construct(name):

            # TODO improve the test
            # we must not invoke the namespace like this

            cls = self.get_class(name)
            d_methods = cls.methods_as_dict
            method = d_methods.pop('__init__', None)

            if method is None:

                # TODO improve case of class with the no __init__

                errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='error', blocker=True)
            args = expr.args

            # TODO check compatibility
            # TODO treat parametrized arguments.

            expr = ConstructorCall(method, args, cls_variable=None)
            #if len(stmts) > 0:
            #    stmts.append(expr)
            #    return CodeBlock(stmts)
            return expr
        else:

            # first we check if it is a macro, in this case, we will create
            # an appropriate FunctionCall

            macro = self.get_macro(name)
            if macro is not None:
                func = macro.master
                name = _get_name(func.name)
                args = macro.apply(args)
            else:
                func = self.get_function(name)
            if func is None:
                return errors.report(UNDEFINED_FUNCTION, symbol=name,
                        bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                        severity='fatal', blocker=self.blocking)
            else:
                return self._handle_function(func, args, **settings)

    def _visit_Expr(self, expr, **settings):
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
            bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
            severity='fatal', blocker=self.blocking)

    def _create_variable(self, name, dtype, rhs, d_lhs):
        """
        Create a new variable. In most cases this is just a call to
        Variable.__init__
        but in the case of a tuple variable it is a recursive call to
        create all elements in the tuple.
        This is done separately to _assign_lhs_variable to ensure that
        elements of a tuple do not exist in the scope

        Parameters
        ----------
        name : str
            The name of the new variable

        dtype : DataType
            The data type of the new variable

        rhs : Variable
            The value assigned to the lhs. This is required to call
            self._infere_type recursively for tuples

        d_lhs : dict
            Dictionary of properties for the new Variable
        """

        if isinstance(rhs, (TupleVariable, PythonTuple, PythonList)):
            elem_vars = []
            for i,r in enumerate(rhs):
                elem_name = self.get_new_name( name + '_' + str(i) )
                elem_d_lhs = self._infere_type( r )

                self._ensure_target( r, elem_d_lhs )

                elem_dtype = elem_d_lhs.pop('datatype')

                var = self._create_variable(elem_name, elem_dtype, r, elem_d_lhs)
                elem_vars.append(var)

            d_lhs['is_pointer'] = any(v.is_pointer for v in elem_vars)
            lhs = TupleVariable(elem_vars, dtype, name, **d_lhs)

        else:
            lhs = Variable(dtype, name, **d_lhs)

        return lhs

    def _ensure_target(self, rhs, d_lhs):
        if isinstance(rhs, Variable) and rhs.allocatable:
            d_lhs['allocatable'] = False
            d_lhs['is_pointer' ] = True

            # TODO uncomment this line, to make rhs target for
            #      lists/tuples.
            rhs.is_target = True
        if isinstance(rhs, IndexedElement) and rhs.rank > 0 and (rhs.base.allocatable or rhs.base.is_pointer):
            d_lhs['allocatable'] = False
            d_lhs['is_pointer' ] = True

            # TODO uncomment this line, to make rhs target for
            #      lists/tuples.
            rhs.base.is_target = not rhs.base.is_pointer

    def _assign_lhs_variable(self, lhs, d_var, rhs, new_expressions, is_augassign, **settings):
        """
        Create a lhs based on the information in d_var
        If the lhs already exists then check that it has the expected properties.

        Parameters
        ----------
        lhs : PyccelSymbol (or DottedName of PyccelSymbols)
            The representation of the lhs provided by the SyntacticParser

        d_var : dict
            Dictionary of expected lhs properties

        rhs : Variable / expression
            The representation of the rhs provided by the SemanticParser.
            This is necessary in order to set the rhs 'is_target' property
            if necessary

        new_expression : list
            A list which allows collection of any additional expressions
            resulting from this operation (e.g. Allocation)

        is_augassign : bool
            Indicates whether this is an assign ( = ) or an augassign ( += / -= / etc )
            This is necessary as the restrictions on the dtype are less strict in this
            case

        settings : dictionary
            Provided to all _visit_ClassName functions
        """

        if isinstance(lhs, PyccelSymbol):

            name = lhs
            dtype = d_var.pop('datatype')

            d_lhs = d_var.copy()
            # ISSUES #177: lhs must be a pointer when rhs is allocatable array
            self._ensure_target(rhs, d_lhs)

            var = self.get_variable_from_scope(name)

            # Variable not yet declared (hence array not yet allocated)
            if var is None:

                # Update variable's dictionary with information from function decorators
                decorators = self._namespace.decorators
                if decorators:
                    if 'stack_array' in decorators:
                        if name in decorators['stack_array']:
                            d_lhs.update(is_stack_array=True,
                                    allocatable=False, is_pointer=False)
                    if 'allow_negative_index' in decorators:
                        if lhs in decorators['allow_negative_index']:
                            d_lhs.update(allows_negative_indexes=True)

                # Create new variable
                lhs = self._create_variable(name, dtype, rhs, d_lhs)

                # Add variable to scope
                self.insert_variable(lhs, name=lhs.name)

                # ...
                # Add memory allocation if needed
                if lhs.allocatable:
                    if self._namespace.is_loop:
                        # Array defined in a loop may need reallocation at every cycle
                        errors.report(ARRAY_DEFINITION_IN_LOOP, symbol=name,
                            severity='warning', blocker=False,
                            bounding_box=(self._current_fst_node.lineno,
                                self._current_fst_node.col_offset))
                        status='unknown'
                    else:
                        # Array defined outside of a loop will be allocated only once
                        status='unallocated'

                    # Create Allocate node
                    new_expressions.append(Allocate(lhs, shape=lhs.alloc_shape, order=lhs.order, status=status))
                # ...

                # ...
                # Add memory deallocation for array variables
                if lhs.is_ndarray and not lhs.is_stack_array:
                    # Create Deallocate node
                    self._allocs[-1].append(lhs)
                # ...

                # We cannot allow the definition of a stack array in a loop
                if lhs.is_stack_array and self._namespace.is_loop:
                    errors.report(STACK_ARRAY_DEFINITION_IN_LOOP, symbol=name,
                        severity='error', blocker=False,
                        bounding_box=(self._current_fst_node.lineno,
                            self._current_fst_node.col_offset))

                # Not yet supported for arrays: x=y+z, x=b[:]
                # Because we cannot infer shape of right-hand side yet
                know_lhs_shape = all(sh is not None for sh in lhs.alloc_shape) \
                    or (lhs.rank == 0)

                if not know_lhs_shape:
                    msg = "Cannot infer shape of right-hand side for expression {} = {}".format(lhs, rhs)
                    errors.report(PYCCEL_RESTRICTION_TODO+'\n'+msg,
                        bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                        severity='fatal', blocker=self.blocking)

            # Variable already exists
            else:

                # TODO improve check type compatibility
                if not hasattr(var, 'dtype'):
                    errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT,
                            symbol = '|{name}| <module> -> {rhs}'.format(name=name, rhs=rhs),
                            bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                            severity='fatal', blocker=False)

                elif not is_augassign and var.is_ndarray and isinstance(rhs, (Variable, IndexedElement)) and var.allocatable:
                    errors.report(ASSIGN_ARRAYS_ONE_ANOTHER,
                        bounding_box=(self._current_fst_node.lineno,
                            self._current_fst_node.col_offset),
                                severity='error', symbol=lhs)

                elif not is_augassign and var.is_ndarray and var.is_target:
                    errors.report(ARRAY_ALREADY_IN_USE,
                        bounding_box=(self._current_fst_node.lineno,
                            self._current_fst_node.col_offset),
                                severity='error', symbol=var.name)

                elif var.is_ndarray and var.is_pointer and isinstance(rhs, NumpyNewArray):
                    errors.report(INVALID_POINTER_REASSIGN,
                        bounding_box=(self._current_fst_node.lineno,
                            self._current_fst_node.col_offset),
                                severity='error', symbol=var.name)

                elif var.is_ndarray and var.is_pointer:
                    # we allow pointers to be reassigned multiple times
                    # pointers reassigning need to call free_pointer func
                    # to remove memory leaks
                    new_expressions.append(Deallocate(var))

                elif not is_augassign and str(dtype) != str(getattr(var, 'dtype', 'None')):
                    txt = '|{name}| {old} <-> {new}'
                    txt = txt.format(name=name, old=var.dtype, new=dtype)

                    errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT,
                    symbol=txt,bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                    severity='error', blocker=False)

                elif not is_augassign:

                    rank  = getattr(var, 'rank' , 'None')
                    order = getattr(var, 'order', 'None')
                    shape = getattr(var, 'shape', 'None')

                    if (d_var['rank'] != rank) or (rank > 1 and d_var['order'] != order):

                        txt = '|{name}| {dtype}{old} <-> {dtype}{new}'
                        format_shape = lambda s: "" if len(s)==0 else s
                        txt = txt.format(name=name, dtype=dtype, old=format_shape(var.shape),
                            new=format_shape(d_var['shape']))
                        errors.report(INCOMPATIBLE_REDEFINITION, symbol=txt,
                            bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                            severity='error', blocker=False)

                    elif d_var['shape'] != shape:

                        if var.is_stack_array:
                            errors.report(INCOMPATIBLE_REDEFINITION_STACK_ARRAY, symbol=name,
                                severity='error', blocker=False,
                                bounding_box=(self._current_fst_node.lineno,
                                    self._current_fst_node.col_offset))

                        else:
                            previous_allocations = var.get_direct_user_nodes(lambda p: isinstance(p, Allocate))
                            if not previous_allocations:
                                errors.report("PYCCEL INTERNAL ERROR : Variable exists already, but it has never been allocated",
                                        symbol=var, severity='fatal')
                            if previous_allocations[-1].get_user_nodes((If, For, While)):
                                status='unknown'
                            elif previous_allocations[-1].get_user_nodes(IfSection):
                                status = previous_allocations[-1].status
                            else:
                                status='allocated'
                            new_expressions.append(Allocate(var,
                                shape=d_var['shape'], order=d_var['order'],
                                status=status))

                            if status != 'unallocated':
                                errors.report(ARRAY_REALLOCATION, symbol=name,
                                    severity='warning', blocker=False,
                                    bounding_box=(self._current_fst_node.lineno,
                                        self._current_fst_node.col_offset))

                # in the case of elemental, lhs is not of the same dtype as
                # var.
                # TODO d_lhs must be consistent with var!
                # the following is a small fix, since lhs must be already
                # declared
                lhs = var

        elif isinstance(lhs, DottedName):

            dtype = d_var.pop('datatype')
            name = lhs.name[:-1]
            if self._current_function == '__init__':

                cls      = self.get_variable('self')
                cls_name = str(cls.cls_base.name)
                cls      = self.get_class(cls_name)

                attributes = cls.attributes
                parent     = cls.superclass
                attributes = list(attributes)
                n_name     = str(lhs.name[-1])

                # update the self variable with the new attributes

                dt       = self.get_class_construct(cls_name)()
                cls_base = self.get_class(cls_name)
                var      = Variable(dt, 'self', cls_base=cls_base)
                d_lhs    = d_var.copy()
                self.insert_variable(var, 'self')


                # ISSUES #177: lhs must be a pointer when rhs is allocatable array
                self._ensure_target(rhs, d_lhs)

                member = self._create_variable(n_name, dtype, rhs, d_lhs)
                lhs    = member.clone(member.name, new_class = DottedVariable, lhs = var)

                # update the attributes of the class and push it to the namespace
                attributes += [member]
                new_cls = ClassDef(cls_name, attributes, [], superclass=parent)
                self.insert_class(new_cls, parent=True)
            else:
                lhs = self._visit(lhs, **settings)
        else:
            raise NotImplementedError("_assign_lhs_variable does not handle {}".format(str(type(lhs))))

        return lhs


    def _visit_Assign(self, expr, **settings):
        # TODO unset position at the end of this part
        new_expressions = []
        fst = expr.fst
        assert(fst)

        rhs = expr.rhs
        lhs = expr.lhs

        if isinstance(rhs, FunctionCall):
            name = rhs.funcdef
            macro = self.get_macro(name)
            if macro is None:
                rhs = self._visit(rhs, **settings)
            else:

                # TODO check types from FunctionDef

                master = macro.master
                name = _get_name(master.name)

                # all terms in lhs must be already declared and available
                # the namespace
                # TODO improve

                if not sympy_iterable(lhs):
                    lhs = [lhs]

                results = []
                for a in lhs:
                    _name = _get_name(a)
                    var = self.get_variable(_name)
                    results.append(var)

                # ...

                args = [self._visit(i, **settings) for i in
                            rhs.args]
                args = macro.apply(args, results=results)
                if isinstance(master, FunctionDef):
                    return FunctionCall(master, args, self._current_function)
                else:
                    # TODO treate interface case
                    errors.report(PYCCEL_RESTRICTION_TODO,
                                  bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                                  severity='fatal')

        elif isinstance(rhs, DottedVariable):
            var = rhs.rhs
            name = _get_name(var)
            macro = self.get_macro(name)
            if macro is None:
                rhs = self._visit(rhs, **settings)
            else:
                master = macro.master
                if isinstance(macro, MacroVariable):
                    rhs = master
                else:
                    # If macro is function, create left-hand side variable
                    if isinstance(master, FunctionDef) and master.results:
                        d_var = self._infere_type(master.results[0], **settings)
                        dtype = d_var.pop('datatype')
                        lhs = Variable(dtype, lhs.name, **d_var)
                        var = self.get_variable_from_scope(lhs.name)
                        if var is None:
                            self.insert_variable(lhs)

                    name = macro.name
                    if not sympy_iterable(lhs):
                        lhs = [lhs]
                    results = []
                    for a in lhs:
                        _name = _get_name(a)
                        var = self.get_variable(_name)
                        results.append(var)

                    args = rhs.rhs.args
                    args = [rhs.lhs] + list(args)
                    args = [self._visit(i, **settings) for i in args]

                    args = macro.apply(args, results=results)

                    # Distinguish between function
                    if master.results:
                        return Assign(lhs[0], FunctionCall(master, args, self._current_function))
                    else:
                        return FunctionCall(master, args, self._current_function)

        else:
            rhs = self._visit(rhs, **settings)

        if isinstance(rhs, FunctionDef):

            # case of lambdify

            rhs = rhs.rename(expr.lhs.name)
            for i in rhs.body:
                i.set_fst(fst)
            rhs = self._visit_FunctionDef(rhs, **settings)
            return rhs

        elif isinstance(rhs, CodeBlock):
            if len(rhs.body)>1 and isinstance(rhs.body[1], FunctionalFor):
                return rhs

            # case of complex stmt
            # that needs to be splitted
            # into a list of stmts
            stmts = rhs.body
            stmt  = stmts[-1]
            lhs   = expr.lhs
            if isinstance(lhs, PyccelSymbol):
                name = lhs
                if self.check_for_variable(name) is None:
                    d_var = self._infere_type(stmt, **settings)
                    dtype = d_var.pop('datatype')
                    lhs = Variable(dtype, name , **d_var)
                    self.insert_variable(lhs)

            if isinstance(expr, Assign):
                stmt = Assign(lhs, stmt)
            elif isinstance(expr, AugAssign):
                stmt = AugAssign(lhs, expr.op, stmt)
            stmt.set_fst(fst)
            stmts[-1] = stmt
            return CodeBlock(stmts)

        elif isinstance(rhs, FunctionCall):

            func = rhs.funcdef
            if isinstance(func, FunctionDef):
                results = func.results
                if results:
                    d_var = [self._infere_type(i, **settings)
                                 for i in results]
                else:
                    raise NotImplementedError("Cannot assign result of a function without a return")

                # case of elemental function
                # if the input and args of func do not have the same shape,
                # then the lhs must be already declared
                if func.is_elemental:
                    # we first compare the funcdef args with the func call
                    # args
#                   d_var = None
                    func_args = func.arguments
                    call_args = rhs.args
                    f_ranks = [x.rank for x in func_args]
                    c_ranks = [x.rank for x in call_args]
                    same_ranks = [x==y for (x,y) in zip(f_ranks, c_ranks)]
                    if not all(same_ranks):
                        assert(len(c_ranks) == 1)
                        for d in d_var:
                            d['shape'      ] = call_args[0].shape
                            d['rank'       ] = call_args[0].rank
                            d['allocatable'] = call_args[0].allocatable
                            d['order'      ] = call_args[0].order

            elif isinstance(func, Interface):
                d_var = [self._infere_type(i, **settings) for i in
                         func.functions[0].results]

                # TODO imporve this will not work for
                # the case of different results types
                d_var[0]['datatype'] = rhs.dtype

            else:
                d_var = self._infere_type(rhs, **settings)

        elif isinstance(rhs, PythonMap):

            name = str(rhs.args[0])
            func = self.get_function(name)

            if func is None:
                errors.report(UNDEFINED_FUNCTION, symbol=name,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='error',blocker=self.blocking)

            dvar  = self._infere_type(rhs.args[1], **settings)
            d_var = [self._infere_type(result, **settings) for result in func.results]
            for d_var_i in d_var:
                d_var_i['shape'] = dvar['shape']
                d_var_i['rank' ]  = dvar['rank']

        else:
            d_var  = self._infere_type(rhs, **settings)
            d_list = d_var if isinstance(d_var, list) else [d_var]

            for d in d_list:
                name = d['datatype'].__class__.__name__

                if name.startswith('Pyccel'):
                    name = name[6:]
                    d['cls_base'] = self.get_class(name)
                    #TODO: Avoid writing the default variables here
                    d['is_pointer'] = d_var.get('is_target',False) or d_var.get('is_pointer',False)

                    # TODO if we want to use pointers then we set target to true
                    # in the ConsturcterCall

                if isinstance(rhs, Variable) and rhs.is_target:
                    # case of rhs is a target variable the lhs must be a pointer
                    d['is_target' ] = False
                    d['is_pointer'] = True

        lhs = expr.lhs
        if isinstance(lhs, (PyccelSymbol, DottedName)):
            if isinstance(d_var, list):
                if len(d_var) == 1:
                    d_var = d_var[0]
                else:
                    errors.report(WRONG_NUMBER_OUTPUT_ARGS, symbol=expr,
                        bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                        severity='error', blocker=self.blocking)
                    return None

            lhs = self._assign_lhs_variable(lhs, d_var, rhs, new_expressions, isinstance(expr, AugAssign), **settings)
        elif isinstance(lhs, PythonTuple):
            n = len(lhs)
            if isinstance(rhs, PythonTuple):
                new_lhs = []
                for i,(l,r) in enumerate(zip(lhs,rhs)):
                    d = self._infere_type(r, **settings)
                    new_lhs.append( self._assign_lhs_variable(l, d, r, new_expressions, isinstance(expr, AugAssign), **settings) )
                lhs = PythonTuple(*new_lhs)

            elif isinstance(rhs, TupleVariable):
                new_lhs = []

                if rhs.is_homogeneous:
                    d_var = self._infere_type(rhs[0])
                    new_rhs = []
                    for i,l in enumerate(lhs):
                        new_lhs.append( self._assign_lhs_variable(l, d_var.copy(),
                            rhs[i], new_expressions, isinstance(expr, AugAssign), **settings) )
                        new_rhs.append(rhs[i])
                    rhs = PythonTuple(*new_rhs)
                    d_var = [d_var]
                else:
                    d_var = [self._infere_type(v) for v in rhs]
                    for i,(l,r) in enumerate(zip(lhs,rhs)):
                        new_lhs.append( self._assign_lhs_variable(l, d_var[i].copy(), r, new_expressions, isinstance(expr, AugAssign), **settings) )

                lhs = PythonTuple(*new_lhs)


            elif isinstance(d_var, list) and len(d_var)== n:
                new_lhs = []
                if hasattr(rhs,'__getitem__'):
                    for i,l in enumerate(lhs):
                        new_lhs.append( self._assign_lhs_variable(l, d_var[i].copy(), rhs[i], new_expressions, isinstance(expr, AugAssign), **settings) )
                else:
                    for i,l in enumerate(lhs):
                        new_lhs.append( self._assign_lhs_variable(l, d_var[i].copy(), rhs, new_expressions, isinstance(expr, AugAssign), **settings) )
                lhs = PythonTuple(*new_lhs)

            elif d_var['shape'][0]==n:
                new_lhs = []
                new_rhs = []

                for l, r in zip(lhs, rhs):
                    new_lhs.append( self._assign_lhs_variable(l, self._infere_type(r), r, new_expressions, isinstance(expr, AugAssign), **settings) )
                    new_rhs.append(r)

                lhs = PythonTuple(*new_lhs)
                rhs = new_rhs
            else:
                errors.report(WRONG_NUMBER_OUTPUT_ARGS, symbol=expr,
                    bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                    severity='error', blocker=self.blocking)
                return None
        else:
            lhs = self._visit(lhs, **settings)

        if isinstance(rhs, (PythonMap, PythonZip)):
            func  = _get_name(rhs.args[0])
            alloc = Assign(lhs, NumpyZeros(lhs.shape, lhs.dtype))
            alloc.set_fst(fst)
            index_name = self.get_new_name(expr)
            index = Variable('int',index_name)
            range_ = FunctionCall('range', (FunctionCall('len', lhs,),))
            name  = _get_name(lhs)
            var   = IndexedElement(name, index)
            args  = rhs.args[1:]
            args  = [_get_name(arg) for arg in args]
            args  = [IndexedElement(arg, index) for arg in args]
            func  = FunctionCall(func, args)
            body  = [Assign(var, func)]
            body[0].set_fst(fst)
            body  = For(index, range_, body)
            body  = self._visit_For(body, **settings)
            body  = [alloc , body]
            return CodeBlock(body)

        elif not isinstance(lhs, (list, tuple)):
            lhs = [lhs]
            if isinstance(d_var,dict):
                d_var = [d_var]

        if len(lhs) == 1:
            lhs = lhs[0]

        if isinstance(lhs, Variable):
            is_pointer = lhs.is_pointer
        elif isinstance(lhs, IndexedElement):
            is_pointer = False
        elif isinstance(lhs, (PythonTuple, PythonList)):
            is_pointer = any(l.is_pointer for l in lhs)

        # TODO: does is_pointer refer to any/all or last variable in list (currently last)
        is_pointer = is_pointer and isinstance(rhs, (Variable, Dlist))
        is_pointer = is_pointer or isinstance(lhs, Variable) and lhs.is_pointer

        # ISSUES #177: lhs must be a pointer when rhs is allocatable array
        if not ((isinstance(lhs, PythonTuple) or (isinstance(lhs, TupleVariable) and not lhs.is_homogeneous)) \
                and isinstance(rhs,(PythonTuple, TupleVariable, list))):
            lhs = [lhs]
            rhs = [rhs]

        for l, r in zip(lhs,rhs):
            is_pointer_i = l.is_pointer if isinstance(l, Variable) else is_pointer

            new_expr = Assign(l, r)

            if is_pointer_i:
                new_expr = AliasAssign(l, r)

            elif isinstance(expr, AugAssign):
                new_expr = AugAssign(l, expr.op, r)


            elif new_expr.is_symbolic_alias:
                new_expr = SymbolicAssign(l, r)

                # in a symbolic assign, the rhs can be a lambda expression
                # it is then treated as a def node

                F = self.get_symbolic_function(l)
                if F is None:
                    self.insert_symbolic_function(new_expr)
                else:
                    errors.report(PYCCEL_RESTRICTION_TODO,
                                  bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                                  severity='fatal')
            new_expressions.append(new_expr)
        if (len(new_expressions)==1):
            new_expressions = new_expressions[0]

            return new_expressions
        else:
            result = CodeBlock(new_expressions)
            return result

    def _visit_For(self, expr, **settings):

        self.create_new_loop_scope()

        # treatment of the index/indices
        iterable = self._visit(expr.iterable, **settings)
        body     = list(expr.body.body)
        iterator = expr.target

        PyccelAstNode.stage = 'syntactic'

        if isinstance(iterable, Variable):
            indx   = self.get_new_variable()
            assign = Assign(iterator, IndexedElement(iterable, indx))
            assign.set_fst(expr.fst)
            iterator = indx
            body     = [assign] + body

        elif isinstance(iterable, PythonMap):
            indx   = self.get_new_variable()
            func   = iterable.args[0]
            args   = [IndexedElement(arg, indx) for arg in iterable.args[1:]]
            assign = Assign(iterator, FunctionCall(func, args))
            assign.set_fst(expr.fst)
            iterator = indx
            body     = [assign] + body

        elif isinstance(iterable, PythonZip):
            args = iterable.args
            indx = self.get_new_variable()
            for i, arg in enumerate(args):
                assign = Assign(iterator[i], IndexedElement(arg, indx))
                assign.set_fst(expr.fst)
                body = [assign] + body
            iterator = indx

        elif isinstance(iterable, PythonEnumerate):
            indx   = iterator.args[0]
            var    = iterator.args[1]
            assign = Assign(var, IndexedElement(iterable.element, indx))
            assign.set_fst(expr.fst)
            iterator = indx
            body     = [assign] + body

        elif isinstance(iterable, Product):
            args     = iterable.elements
            iterator = list(iterator)
            for i,arg in enumerate(args):
                if not isinstance(arg, PythonRange):
                    indx   = self.get_new_variable()
                    assign = Assign(iterator[i], IndexedElement(arg, indx))

                    assign.set_fst(expr.fst)
                    body        = [assign] + body
                    iterator[i] = indx

        if isinstance(iterator, PyccelSymbol):
            name   = iterator
            var    = self.check_for_variable(name)
            target = var
            if var is None:
                target = Variable('int', name, rank=0)
                self.insert_variable(target)

        elif isinstance(iterator, list):
            target = []
            for name in iterator:
                var  = Variable('int', name, rank=0)
                self.insert_variable(var)
                target.append(var)
        else:

            # TODO ERROR not tested yet

            errors.report(INVALID_FOR_ITERABLE, symbol=expr.target,
                   bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                   severity='error', blocker=self.blocking)
        PyccelAstNode.stage = 'semantic'

        body = [self._visit(i, **settings) for i in body]

        local_vars = list(self.namespace.variables.values())
        self.exit_loop_scope()

        if isinstance(iterable, Variable):
            return ForIterator(target, iterable, body)

        return For(target, iterable, body, local_vars=local_vars)


    def _visit_GeneratorComprehension(self, expr, **settings):
        msg = "Generator expressions as args are not currently correctly implemented\n"
        msg += "See issue #272 at https://github.com/pyccel/pyccel/issues"
        errors.report(msg, symbol = expr,
                  bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                  severity='fatal')

        result   = expr.expr
        lhs_name = _get_name(expr.lhs)
        lhs  = self.check_for_variable(lhs_name)

        if lhs is None:
            tmp_lhs  = Variable('int', lhs_name)
            self.insert_variable(tmp_lhs)
        else:
            tmp_lhs = None

        loops  = [self._visit(i, **settings) for i in expr.loops]
        result = self._visit(result, **settings)
        if isinstance(result, CodeBlock):
            result = result.body[-1]


        d_var = self._infere_type(result, **settings)
        dtype = d_var.pop('datatype')

        if tmp_lhs is not None:
            self.remove_variable(tmp_lhs)
            lhs = Variable(dtype, lhs_name, **d_var)
            self.insert_variable(lhs)


        if isinstance(expr, FunctionalSum):
            val = LiteralInteger(0)
            if str_dtype(dtype) in ['real', 'complex']:
                val = LiteralFloat(0.0)
        elif isinstance(expr, FunctionalMin):
            val = INF
        elif isinstance(expr, FunctionalMax):
            val = -INF

        stmt = Assign(expr.lhs, val)
        stmt.set_fst(expr.fst)
        loops.insert(0, stmt)

        if isinstance(expr, FunctionalSum):
            expr_new = FunctionalSum(loops, lhs=lhs)
        elif isinstance(expr, FunctionalMin):
            expr_new = FunctionalMin(loops, lhs=lhs)
        elif isinstance(expr, FunctionalMax):
            expr_new = FunctionalMax(loops, lhs=lhs)
        expr_new.set_fst(expr.fst)
        return expr_new

    def _visit_FunctionalFor(self, expr, **settings):

        target  = expr.expr
        index   = expr.index
        indices = expr.indices
        dims    = []
        body    = expr.loops[1]

        sp_indices  = [sp_Symbol(i) for i in indices]
        idx_subs = dict()

        # The symbols created to represent unknown valued objects are temporary
        tmp_used_names = self.used_names.copy()
        while isinstance(body, For):

            stop  = None
            start = LiteralInteger(0)
            step  = LiteralInteger(1)
            var   = body.target
            a     = self._visit(body.iterable, **settings)
            if isinstance(a, PythonRange):
                var   = Variable('int', var)
                stop  = a.stop
                start = a.start
                step  = a.step
            elif isinstance(a, (PythonZip, PythonEnumerate)):
                dvar  = self._infere_type(a.element, **settings)
                dtype = dvar.pop('datatype')
                if dvar['rank'] > 0:
                    dvar['rank' ] -= 1
                    dvar['shape'] = (dvar['shape'])[1:]
                if dvar['rank'] == 0:
                    dvar['allocatable'] = dvar['is_pointer'] = False
                var  = Variable(dtype, var, **dvar)
                stop = a.element.shape[0]
            elif isinstance(a, Variable):
                dvar  = self._infere_type(a, **settings)
                dtype = dvar.pop('datatype')
                if dvar['rank'] > 0:
                    dvar['rank'] -= 1
                    dvar['shape'] = (dvar['shape'])[1:]
                if dvar['rank'] == 0:
                    dvar['allocatable'] = dvar['is_pointer'] = False

                var  = Variable(dtype, var, **dvar)
                stop = a.shape[0]
            else:
                errors.report(PYCCEL_RESTRICTION_TODO,
                              bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                              severity='fatal')
            self.insert_variable(var)
            step.invalidate_node()
            step  = pyccel_to_sympy(step , idx_subs, tmp_used_names)
            start.invalidate_node()
            start = pyccel_to_sympy(start, idx_subs, tmp_used_names)
            stop.invalidate_node()
            stop  = pyccel_to_sympy(stop , idx_subs, tmp_used_names)
            size = (stop - start) / step
            if (step != 1):
                size = ceiling(size)

            body = body.body.body[0]
            dims.append((size, step, start, stop))

        # we now calculate the size of the array which will be allocated

        for idx in indices:
            var = self.get_variable(idx)
            idx_subs[idx] = var


        dim = sp_Integer(1)

        for i in reversed(range(len(dims))):
            size  = dims[i][0]
            step  = dims[i][1]
            start = dims[i][2]
            stop  = dims[i][3]

            # For complicated cases we must ensure that the upper bound is never smaller than the
            # lower bound as this leads to too little memory being allocated
            min_size = size
            # Collect all uses of other indices
            start_idx = [-1] + [sp_indices.index(a) for a in start.atoms(sp_Symbol) if a in sp_indices]
            stop_idx  = [-1] + [sp_indices.index(a) for a in  stop.atoms(sp_Symbol) if a in sp_indices]
            start_idx.sort()
            stop_idx.sort()

            # Find the minimum size
            while max(len(start_idx),len(stop_idx))>1:
                # Use the maximum value of the start
                if start_idx[-1] > stop_idx[-1]:
                    s = start_idx.pop()
                    min_size = min_size.subs(sp_indices[s], dims[s][3])
                # and the minimum value of the stop
                else:
                    s = stop_idx.pop()
                    min_size = min_size.subs(sp_indices[s], dims[s][2])

            # While the min_size is not a known integer, assume that the bounds are positive
            j = 0
            while not isinstance(min_size, sp_Integer) and j<=i:
                min_size = min_size.subs(dims[j][3]-dims[j][2], 1).simplify()
                j+=1
            # If the min_size is negative then the size will be wrong and an error is raised
            if isinstance(min_size, sp_Integer) and min_size < 0:
                errors.report(PYCCEL_RESTRICTION_LIST_COMPREHENSION_LIMITS.format(indices[i]),
                          bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                          severity='error')

            # sympy is necessary to carry out the summation
            dim   = dim.subs(sp_indices[i], start+step*sp_indices[i])
            dim   = Summation(dim, (sp_indices[i], 0, size-1))
            dim   = dim.doit()

        try:
            dim = sympy_to_pyccel(dim, idx_subs)
        except TypeError:
            errors.report(PYCCEL_RESTRICTION_LIST_COMPREHENSION_SIZE + '\n Deduced size : {}'.format(dim),
                          bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                          severity='fatal')

        # TODO find a faster way to calculate dim
        # when step>1 and not isinstance(dim, Sum)
        # maybe use the c++ library of sympy

        # we annotate the target to infere the type of the list created

        target = self._visit(target, **settings)
        d_var = self._infere_type(target, **settings)

        dtype = d_var['datatype']

        if dtype is NativeGeneric():
            errors.report(LIST_OF_TUPLES,
                          bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                          severity='fatal')

        d_var['rank'] += 1
        d_var['allocatable'] = True
        shape = list(d_var['shape'])
        shape.insert(0, dim)
        d_var['shape'] = shape
        d_var['is_stack_array'] = False # PythonTuples can be stack arrays

        # ...
        # TODO [YG, 30.10.2020]:
        #  - Check if we should allow the possibility that is_stack_array=True
        # ...
        lhs_symbol = expr.lhs.base
        ne = []
        lhs = self._assign_lhs_variable(lhs_symbol, d_var, rhs=expr, new_expressions=ne, is_augassign=False, **settings)
        lhs_alloc = ne[0]

        if isinstance(target, PythonTuple) and not target.is_homogeneous:
            errors.report(LIST_OF_TUPLES, symbol=expr,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                severity='error', blocker=self.blocking)

        target.invalidate_node()

        loops = [self._visit(i, **settings) for i in expr.loops]
        index = self._visit(index, **settings)

        return CodeBlock([lhs_alloc, FunctionalFor(loops, lhs=lhs, indices=indices, index=index)])

    def _visit_While(self, expr, **settings):

        self.create_new_loop_scope()
        test = self._visit(expr.test, **settings)
        body = self._visit(expr.body, **settings)
        local_vars = list(self.namespace.variables.values())
        self.exit_loop_scope()

        return While(test, body, local_vars)

    def _visit_IfSection(self, expr, **settings):
        cond = self._visit(expr.condition)
        body = self._visit(expr.body)
        return IfSection(cond, body)

    def _visit_If(self, expr, **settings):
        args = [self._visit(i, **settings) for i in expr.blocks]
        return If(*args)

    def _visit_IfTernaryOperator(self, expr, **settings):
        cond        = self._visit(expr.cond, **settings)
        value_true  = self._visit(expr.value_true, **settings)
        value_false = self._visit(expr.value_false, **settings)
        return IfTernaryOperator(cond, value_true, value_false)

    def _visit_VariableHeader(self, expr, **settings):

        # TODO improve
        #      move it to the ast like create_definition for FunctionHeader?

        name  = expr.name
        d_var = expr.dtypes.copy()
        dtype = d_var.pop('datatype')
        d_var.pop('is_func')

        var = Variable(dtype, name, **d_var)
        self.insert_variable(var)
        return expr

    def _visit_FunctionHeader(self, expr, **settings):
        # TODO should we return it and keep it in the AST?
        self.insert_header(expr)
        return expr

    def _visit_Template(self, expr, **settings):
        self.insert_template(expr)
        return expr

    def _visit_ClassHeader(self, expr, **settings):
        # TODO should we return it and keep it in the AST?
        self.insert_header(expr)
        return expr

    def _visit_Return(self, expr, **settings):

        results     = expr.expr
        f_name      = self._current_function
        if isinstance(f_name, DottedName):
            f_name = f_name.name[-1]

        return_vars = self.get_function(f_name).results
        assigns     = []
        for v,r in zip(return_vars, results):
            if not (isinstance(r, PyccelSymbol) and r == v):
                a = Assign(v, r)
                a.set_fst(expr.fst)
                assigns.append(a)

        assigns = [self._visit_Assign(e) for e in assigns]
        results = [self._visit(i, **settings) for i in return_vars]

        #add the Deallocate node before the Return node
        code = assigns + [Deallocate(i) for i in self._allocs[-1]]
        if code:
            expr  = Return(results, CodeBlock(code))
        else:
            expr  = Return(results)
        return expr

    def _visit_FunctionDef(self, expr, **settings):

        name            = expr.name
        name            = name.replace("'", '')
        cls_name        = expr.cls_name
        decorators      = expr.decorators
        funcs           = []
        sub_funcs       = []
        func_interfaces = []
        is_pure         = expr.is_pure
        is_elemental    = expr.is_elemental
        is_private      = expr.is_private
        doc_string      = self._visit(expr.doc_string) if expr.doc_string else expr.doc_string
        headers = []

        not_used = [d for d in decorators if d not in def_decorators.__all__]
        if len(not_used) >= 1:
            errors.report(UNUSED_DECORATORS, symbol=', '.join(not_used), severity='warning')

        args_number = len(expr.arguments)
        templates = self.get_templates()
        if decorators['template']:
            # Load templates dict from decorators dict
            templates.update(decorators['template']['template_dict'])

        tmp_headers = expr.headers
        if cls_name:
            tmp_headers += self.get_header(cls_name + '.' + name)
            args_number -= 1
        else:
            tmp_headers += self.get_header(name)
        for header in tmp_headers:
            if all(header.dtypes != hd.dtypes for hd in headers):
                headers.append(header)
            else:
                errors.report(DUPLICATED_SIGNATURE, symbol=header,
                        severity='warning')
        for hd in headers:
            if (args_number != len(hd.dtypes)):
                msg = 'The number of arguments in the function {} ({}) does not match the number\
                        of types in decorator/header ({}).'.format(name ,args_number, len(hd.dtypes))
                if (args_number < len(hd.dtypes)):
                    errors.report(msg, symbol=expr.arguments, severity='warning')
                else:
                    errors.report(msg, symbol=expr.arguments, severity='fatal')

        interfaces = []
        if len(headers) == 0:
            # check if a header is imported from a header file
            # TODO improve in the case of multiple headers ( interface )
            func       = self.get_function(name)
            if func and func.is_header:
                interfaces = [func]

        if expr.arguments and not headers and not interfaces:

            # TODO ERROR wrong position

            errors.report(FUNCTION_TYPE_EXPECTED, symbol=name,
                   bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                   severity='error', blocker=self.blocking)

        # We construct a FunctionDef from each function header
        for hd in headers:
            interfaces += hd.create_definition(templates)

        if not interfaces:
            # this for the case of a function without arguments => no headers
            interfaces = [FunctionDef(name, [], [], [])]

#        TODO move this to codegen
#        vec_func = None
#        if 'vectorize' in decorators:
#            #TODO move to another place
#            vec_name  = 'vec_' + name
#            arg       = decorators['vectorize'][0]
#            arg       = str(arg.name)
#            args      = [str(i.name) for i in expr.arguments]
#            index_arg = args.index(arg)
#            arg       = Symbol(arg)
#            vec_arg   = arg
#            index     = self.get_new_variable()
#            range_    = Function('range')(Function('len')(arg))
#            args      = symbols(args)
#            args[index_arg] = vec_arg[index]
#            body_vec        = Assign(args[index_arg], Function(name)(*args))
#            body_vec.set_fst(expr.fst)
#            body_vec   = [For(index, range_, [body_vec])]
#            header_vec = header.vectorize(index_arg)
#            vec_func   = expr.vectorize(body_vec, header_vec)

        interface_name = name

        for i, m in enumerate(interfaces):
            args           = []
            results        = []
            local_vars     = []
            global_vars    = []
            imports        = []
            arg            = None
            arguments      = expr.arguments
            header_results = m.results

            if len(interfaces) > 1:
                name = interface_name + '_' + str(i).zfill(2)
            self.create_new_function_scope(name, decorators)

            if cls_name and str(arguments[0].name) == 'self':
                arg       = arguments[0]
                arguments = arguments[1:]
                dt        = self.get_class_construct(cls_name)()
                cls_base  = self.get_class(cls_name)
                var       = Variable(dt, 'self', cls_base=cls_base)
                self.insert_variable(var, 'self')

            if arguments:
                for (a, ah) in zip(arguments, m.arguments):
                    additional_args = []
                    if isinstance(ah, FunctionAddress):
                        d_var = {}
                        d_var['is_argument'] = True
                        d_var['is_pointer'] = True
                        d_var['is_kwonly'] = a.is_kwonly
                        if isinstance(a, ValuedArgument):

                            # optional argument only if the value is None
                            if isinstance(a.value, Nil):
                                d_var['is_optional'] = True

                            a_new = ValuedFunctionAddress(a.name, ah.arguments, ah.results, [],
                                        value=a.value, **d_var)
                        else:
                            a_new = FunctionAddress(a.name, ah.arguments, ah.results, [], **d_var)
                    else:
                        d_var = self._infere_type(ah, **settings)
                        d_var['shape'] = ah.alloc_shape
                        d_var['is_argument'] = True
                        d_var['is_kwonly'] = a.is_kwonly
                        d_var['is_const'] = ah.is_const
                        dtype = d_var.pop('datatype')
                        if d_var['rank']>0 and not d_var['cls_base']:
                            d_var['cls_base'] = NumpyArrayClass

                        if 'allow_negative_index' in self._namespace.decorators:
                            if a.name in decorators['allow_negative_index']:
                                d_var.update(allows_negative_indexes=True)
                        # this is needed for the static case
                        if isinstance(a, ValuedArgument):

                            # optional argument only if the value is None
                            if isinstance(a.value, Nil):
                                d_var['is_optional'] = True

                            a_new = ValuedVariable(dtype, a.name,
                                        value=a.value, **d_var)
                        else:
                            a_new = Variable(dtype, a.name, **d_var)

                    if additional_args:
                        args += additional_args

                    args.append(a_new)
                    if isinstance(a_new, FunctionAddress):
                        self.insert_function(a_new)
                    else:
                        self.insert_variable(a_new, name=a_new.name)
            results = expr.results
            if header_results:
                new_results = []

                for a, ah in zip(results, header_results):
                    d_var = self._infere_type(ah, **settings)
                    dtype = d_var.pop('datatype')
                    a_new = Variable(dtype, a, **d_var)
                    self.insert_variable(a_new, name=a_new.name)
                    new_results.append(a_new)

                results = new_results

            # insert the FunctionDef into the scope
            # to handle the case of a recursive function
            # TODO improve in the case of an interface
            recursive_func_obj = FunctionDef(name, args, results, [])
            self.insert_function(recursive_func_obj)

            # Create a new list that store local variables for each FunctionDef to handle nested functions
            self._allocs.append([])

            # we annotate the body
            body = self._visit(expr.body)

            # Calling the Garbage collecting,
            # it will add the necessary Deallocate nodes
            # to the body of the function
            body = self.garbage_collector(body)

            results = [self._visit(a) for a in results]

            if arg and cls_name:
                dt       = self.get_class_construct(cls_name)()
                cls_base = self.get_class(cls_name)
                var      = Variable(dt, 'self', cls_base=cls_base)
                args     = [var] + args

            # Determine local and global variables
            local_vars  = [v for v in self.get_variables(self.namespace)              if v not in args + results]
            global_vars = [v for v in self.get_variables(self.namespace.parent_scope) if v not in args + results + local_vars]

            # get the imports
            imports   = self.namespace.imports['imports'].values()
            imports   = list(set(imports))

            # remove the FunctionDef from the function scope
            # TODO improve func_ is None in the case of an interface
            func_     = self.namespace.functions.pop(name, None)
            is_recursive = False
            # check if the function is recursive if it was called on the same scope
            if func_ and func_.is_recursive:
                is_recursive = True

            sub_funcs = [i for i in self.namespace.functions.values() if not i.is_header and not isinstance(i, FunctionAddress)]

            func_args = [i for i in self.namespace.functions.values() if isinstance(i, FunctionAddress)]
            if func_args:
                func_interfaces.append(Interface('', func_args, is_argument = True))

            self.exit_function_scope()

            # ... computing inout arguments
            args_inout = [False] * len(args)

            results_names = [i.name for i in results]

            all_assigned = get_assigned_symbols(body)
            assigned     = [a.name for a in all_assigned if a.rank > 0]
            all_assigned = [i.name for i in all_assigned]

            apps = body.get_attribute_nodes(FunctionCall)
            apps = [i for i in apps if (i.__class__.__name__
                    in self.get_parent_functions())]

            d_apps = OrderedDict((a, []) for a in args)
            for f in apps:
                a_args = set(f.args) & set(args)
                for a in a_args:
                    d_apps[a].append(f)

            for i, a in enumerate(args):
                if a.name in chain(results_names, assigned, ['self']):
                    args_inout[i] = True

                if d_apps[a] and not( args_inout[i] ):
                    intent = False
                    n_fa = len(d_apps[a])
                    i_fa = 0
                    while not(intent) and i_fa < n_fa:
                        fa = d_apps[a][i_fa]
                        f_name = fa.__class__.__name__
                        func = self.get_function(f_name)

                        j = list(fa.args).index(a)
                        intent = func.arguments_inout[j]
                        if intent:
                            args_inout[i] = True

                        i_fa += 1
                if isinstance(a, Variable):
                    if a.is_const and (args_inout[i] or (a.name in all_assigned)):
                        msg = "Cannot modify 'const' argument ({})".format(a)
                        errors.report(msg, bounding_box=(self._current_fst_node.lineno,
                            self._current_fst_node.col_offset),
                            severity='fatal', blocker=self.blocking)
            # ...

            # Raise an error if one of the return arguments is either:
            #   a) a pointer
            #   b) array which is not among arguments, hence intent(out)
            for r in results:
                if r.is_pointer:
                    errors.report(UNSUPPORTED_ARRAY_RETURN_VALUE,
                    symbol=r,bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                    severity='fatal')
                elif (r not in args) and r.rank > 0:
                    errors.report(UNSUPPORTED_ARRAY_RETURN_VALUE,
                    symbol=r,bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                    severity='fatal')

            func = FunctionDef(name,
                    args,
                    results,
                    body,
                    local_vars=local_vars,
                    global_vars=global_vars,
                    cls_name=cls_name,
                    is_pure=is_pure,
                    is_elemental=is_elemental,
                    is_private=is_private,
                    imports=imports,
                    decorators=decorators,
                    is_recursive=is_recursive,
                    arguments_inout=args_inout,
                    functions = sub_funcs,
                    interfaces = func_interfaces,
                    doc_string = doc_string)
            if recursive_func_obj not in body.get_attribute_nodes(FunctionDef):
                recursive_func_obj.invalidate_node()

            if cls_name:
                cls = self.get_class(cls_name)
                methods = list(cls.methods) + [func]

                # update the class methods

                self.insert_class(ClassDef(cls_name, cls.attributes,
                        methods, superclass=cls.superclass))

            funcs += [func]

            #clear the sympy cache
            #TODO clear all variable except the global ones
            cache.clear_cache()
        if len(funcs) == 1:
            funcs = funcs[0]
            self.insert_function(funcs)

        else:
            for f in funcs:
                self.insert_function(f)

            funcs = Interface(interface_name, funcs)
            self.insert_function(funcs)
#        TODO move this to codegen
#        if vec_func:
#           self._visit_FunctionDef(vec_func, **settings)
#           vec_func = self.namespace.functions.pop(vec_name)
#           if isinstance(funcs, Interface):
#               funcs = list(funcs.funcs)+[vec_func]
#           else:
#               self.namespace.sons_scopes['sc_'+ name] = self.namespace.sons_scopes[name]
#               funcs = funcs.rename('sc_'+ name)
#               funcs = [funcs, vec_func]
#           funcs = Interface(name, funcs)
#           self.insert_function(funcs)
        return EmptyNode()

    def _visit_PythonPrint(self, expr, **settings):
        args = [self._visit(i, **settings) for i in expr.expr]
        if len(args) == 0:
            return PythonPrint(args)

        is_symbolic = lambda var: isinstance(var, Variable) \
            and isinstance(var.dtype, NativeSymbol)

        # TODO fix: not yet working because of mpi examples
#        if not test:
#            # TODO: Add description to parser/messages.py
#            errors.report('Either all arguments must be symbolic or none of them can be',
#                   bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
#                   severity='fatal', blocker=self.blocking)

        if is_symbolic(args[0]):
            _args = []
            for a in args:
                f = self.get_symbolic_function(a.name)
                if f is None:
                    _args.append(a)
                else:

                    # TODO improve: how can we print SymbolicAssign as  lhs = rhs

                    _args.append(f)
            return SymbolicPrint(_args)
        else:
            return PythonPrint(args)

    def _visit_ClassDef(self, expr, **settings):

        # TODO - improve the use and def of interfaces
        #      - wouldn't be better if it is done inside ClassDef?

        name = expr.name
        name = name.replace("'", '')
        methods = list(expr.methods)
        parent = expr.superclass
        interfaces = []

        # remove quotes for str representation
        cls = ClassDef(name, [], [], superclass=parent)
        self.insert_class(cls)
        const = None

        for (i, method) in enumerate(methods):
            m_name = method.name.replace("'", '')

            if m_name == '__init__':
                self._visit_FunctionDef(method, **settings)
                methods.pop(i)
                const = self.namespace.functions.pop(m_name)
                break



        if not const:
            errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                   bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                   severity='error', blocker=True)

        ms = []
        for i in methods:
            self._visit_FunctionDef(i, **settings)
            m_name = i.name.replace("'", '')
            m = self.namespace.functions.pop(m_name)
            ms.append(m)

        methods = [const] + ms
        header = self.get_header(name)

        if not header:
            errors.report(PYCCEL_MISSING_HEADER, symbol=name,
                   bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                   severity='fatal', blocker=self.blocking)

        attributes = self.get_class(name).attributes

        for i in methods:
            if isinstance(i, Interface):
                methods.remove(i)
                interfaces += [i]

        cls = ClassDef(name, attributes, methods,
              interfaces=interfaces, superclass=parent)
        self.insert_class(cls)

        return EmptyNode()

    def _visit_Del(self, expr, **settings):

        ls = [self._visit(i, **settings) for i in expr.variables]
        return Del(ls)

    def _visit_PyccelIs(self, expr, **settings):
        # Handles PyccelIs and PyccelIsNot
        IsClass = type(expr)

        # TODO ERROR wrong position ??

        var1 = self._visit(expr.lhs)
        var2 = self._visit(expr.rhs)

        if (var1 is var2) or (isinstance(var2, Nil) and isinstance(var1, Nil)):
            if IsClass == PyccelIsNot:
                return LiteralFalse()
            elif IsClass == PyccelIs:
                return LiteralTrue()

        if isinstance(var1, Nil):
            var1, var2 = var2, var1

        if isinstance(var2, Nil):
            if not var1.is_optional:
                errors.report(PYCCEL_RESTRICTION_OPTIONAL_NONE,
                        bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                        severity='error', blocker=self.blocking)
            return IsClass(var1, expr.rhs)

        if (var1.dtype != var2.dtype):
            if IsClass == PyccelIs:
                return LiteralFalse()
            elif IsClass == PyccelIsNot:
                return LiteralTrue()

        if (isinstance(var1.dtype, NativeBool) and
            isinstance(var2.dtype, NativeBool)):
            return IsClass(var1, var2)

        lst = [NativeString(), NativeComplex(), NativeReal(), NativeInteger()]
        if (var1.dtype in lst):
            errors.report(PYCCEL_RESTRICTION_PRIMITIVE_IMMUTABLE, symbol=expr,
            bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
            severity='error', blocker=self.blocking)
            return IsClass(var1, var2)

        errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
            bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
            severity='error', blocker=self.blocking)
        return IsClass(var1, var2)

    def _visit_Import(self, expr, **settings):

        # TODO - must have a dict where to store things that have been
        #        imported
        #      - should not use namespace

        container = self.namespace.imports

        if isinstance(expr.source, AsName):
            source        = expr.source.name
            source_target = expr.source.target
        else:
            source        = str(expr.source)
            source_target = source

        if source in pyccel_builtin_import_registery:
            imports = pyccel_builtin_import(expr)

            def _insert_obj(location, target, obj):
                F = self.check_for_variable(target)

                if obj is F:
                    errors.report(FOUND_DUPLICATED_IMPORT,
                                symbol=target, severity='warning')
                elif F is None or isinstance(F, dict):
                    container[location][target] = obj
                else:
                    errors.report(IMPORTING_EXISTING_IDENTIFIED,
                                  bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                                  severity='fatal')

            if expr.target:
                for (name, atom) in imports:
                    if not name is None:
                        if isinstance(atom, Constant):
                            _insert_obj('variables', name, atom)
                        else:
                            _insert_obj('functions', name, atom)
            else:
                _insert_obj('variables', source_target, imports)
            _insert_obj('imports', source_target, Import(source, expr.target, True))

        else:

            # in some cases (blas, lapack, openmp and openacc level-0)
            # the import should not appear in the final file
            # all metavars here, will have a prefix and suffix = __

            __ignore_at_import__ = False
            __module_name__      = None
            __import_all__       = False
            __print__            = False

            # we need to use str here since source has been defined
            # using repr.
            # TODO shall we improve it?

            p       = self.d_parsers[source_target]
            if expr.target:
                targets = [i.target if isinstance(i,AsName) else i for i in expr.target]
                names = [i.name if isinstance(i,AsName) else i for i in expr.target]
                for entry in ['variables', 'classes', 'functions']:
                    d_son = getattr(p.namespace, entry)
                    for t,n in zip(targets,names):
                        if n in d_son:
                            e = d_son[n]
                            if t == n:
                                container[entry][t] = e
                            else:
                                container[entry][t] = e.clone(t)
            else:
                imported_dict = []
                for entry in ['variables', 'classes', 'functions']:
                    d_son = getattr(p.namespace, entry)
                    imported_dict.extend(d_son.items())
                container['variables'][source_target] = dict(imported_dict)

            self.namespace.cls_constructs.update(p.namespace.cls_constructs)
            self.namespace.macros.update(p.namespace.macros)

            # ... meta variables

            if 'ignore_at_import' in list(p.metavars.keys()):
                __ignore_at_import__ = p.metavars['ignore_at_import']

            if 'import_all' in list(p.metavars.keys()):
                __import_all__ = p.metavars['import_all']

            if 'module_name' in list(p.metavars.keys()):
                __module_name__ = p.metavars['module_name']

            if 'print' in list(p.metavars.keys()):
                __print__ = True

            if len(expr.target) == 0 and isinstance(expr.source,AsName):
                expr = Import(expr.source.name)

            if source_target in container['imports']:
                targets = container['imports'][source_target].target + expr.target
            else:
                targets = expr.target

            expr = Import(expr.source, targets)

            if __import_all__:
                expr = Import(__module_name__)
                container['imports'][source_target] = expr

            elif __module_name__:
                expr = Import(__module_name__, expr.target)
                container['imports'][source_target] = expr

            # ...
            elif __print__ in p.metavars.keys():
                source = str(expr.source).split('.')[-1]
                source = 'mod_' + source
                expr   = Import(source, expr.target)
                container['imports'][source_target] = expr
            elif not __ignore_at_import__:

                container['imports'][source_target] = expr

        return EmptyNode()



    def _visit_With(self, expr, **settings):

        domaine = self._visit(expr.test, **settings)
        parent  = domaine.cls_base
        if not parent.is_with_construct:
            errors.report(UNDEFINED_WITH_ACCESS,
                   bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                   severity='fatal', blocker=self.blocking)

        body = self._visit(expr.body, **settings)
        return With(domaine, body).block



    def _visit_MacroFunction(self, expr, **settings):
        # we change here the master name to its FunctionDef

        f_name = expr.master
        header = self.get_header(f_name)
        if not header:
            func = self.get_function(f_name)
            if func is None:
                errors.report(MACRO_MISSING_HEADER_OR_FUNC,
                symbol=f_name,severity='error', blocker=self.blocking,
                bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset))
        else:
            interfaces = []
            for hd in header:
                interfaces += hd.create_definition()

            # TODO -> Said: must handle interface

            func = interfaces[0]

        name = expr.name
        args = [self._visit(a, **settings) if isinstance(a, ValuedArgument)
                else a for a in expr.arguments]
        master_args = [self._visit(a, **settings) if isinstance(a, ValuedArgument)
                else a for a in expr.master_arguments]
        results = expr.results
        macro   = MacroFunction(name, args, func, master_args,
                                  results=results)
        self.insert_macro(macro)

        return macro

    def _visit_MacroShape(self, expr, **settings):
        return expr

    def _visit_MacroVariable(self, expr, **settings):

        master = expr.master
        if isinstance(master, DottedName):
            errors.report(PYCCEL_RESTRICTION_TODO,
                          bounding_box=(self._current_fst_node.lineno, self._current_fst_node.col_offset),
                          severity='fatal')
        header = self.get_header(master)
        if header is None:
            var = self.get_variable(master)
        else:
            var = Variable(header.dtype, header.name)

                # TODO -> Said: must handle interface

        expr = MacroVariable(expr.name, var)
        self.insert_macro(expr)
        return expr

    def _visit_Dlist(self, expr, **settings):
        # Arguments have been treated in PyccelMul

        val = expr.val
        length = expr.length
        if isinstance(val, (TupleVariable, PythonTuple)) and \
                not isinstance(val, PythonList):
            if isinstance(length, LiteralInteger):
                length = length.p
            if isinstance(val, TupleVariable):
                return PythonTuple(*(val.get_vars()*length))
            else:
                return PythonTuple(*(val.args*length))
        return Dlist(val, length)

    def _visit_StarredArguments(self, expr, **settings):
        var = self._visit(expr.args_var)
        assert(var.rank==1)
        size = var.shape[0]
        return StarredArguments([var[i] for i in range(size)])

#==============================================================================


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        raise ValueError('Expecting an argument for filename')

    parser = SyntaxParser(filename)
#    print(parser.namespace)
    parser = SemanticParser(parser)
#    print(parser.ast)
#    parser.view_namespace('variables')

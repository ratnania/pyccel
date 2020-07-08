# -*- coding: utf-8 -*-
# pylint: disable=R0201

import redbaron
import os
import re

import ast
#==============================================================================
from pyccel.parser.extend_tree import extend_tree
from pyccel.parser.extend_tree import CommentLine
#==============================================================================

from redbaron import RedBaron
from redbaron import AssignmentNode
from redbaron import DefNode
from redbaron import ClassNode
from redbaron import TupleNode, ListNode
from redbaron import ArgumentGeneratorComprehensionNode
from redbaron import DictitemNode
from redbaron import DotNode
from redbaron import CallNode
from redbaron import GetitemNode
from redbaron import CommentNode

from sympy.core.function import Function
from sympy import Symbol
from sympy import IndexedBase
from sympy import Tuple
from sympy import Lambda
from sympy import Dict

#==============================================================================

from pyccel.ast.basic import PyccelAstNode

from pyccel.ast.core import ParserResult
from pyccel.ast.core import String
from pyccel.ast.core import Nil
from pyccel.ast.core import DottedName, DottedVariable
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Return
from pyccel.ast.core import Pass
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import PythonFunction, SympyFunction
from pyccel.ast.core import ClassDef
from pyccel.ast.core import For, FunctionalFor
from pyccel.ast.core import If, IfTernaryOperator
from pyccel.ast.core import While
from pyccel.ast.core import Del
from pyccel.ast.core import Assert
from pyccel.ast.core import PythonTuple
from pyccel.ast.core import Comment, EmptyLine, NewLine
from pyccel.ast.core import Break, Continue
from pyccel.ast.core import Slice
from pyccel.ast.core import Argument, ValuedArgument
from pyccel.ast.core import Is, IsNot
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.core import CommentBlock
from pyccel.ast.core import With
from pyccel.ast.core import List
from pyccel.ast.core import StarredArguments
from pyccel.ast.core import CodeBlock
from pyccel.ast.core import create_variable

from pyccel.ast.core import PyccelPow, PyccelAdd, PyccelMul, PyccelDiv, PyccelMod, PyccelFloorDiv
from pyccel.ast.core import PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe
from pyccel.ast.core import PyccelAnd, PyccelOr,  PyccelNot, PyccelMinus, PyccelAssociativeParenthesis
from pyccel.ast.core import PyccelOperator, PyccelUnary

from pyccel.ast.builtins import Print
from pyccel.ast.headers  import Header, MetaVariable
from pyccel.ast.numbers  import Integer, Float, Complex, BooleanFalse, BooleanTrue
from pyccel.ast.functionalexpr import FunctionalSum, FunctionalMax, FunctionalMin

from pyccel.parser.base import BasicParser
from pyccel.parser.utilities import fst_move_directives, preprocess_imports
from pyccel.parser.utilities import reconstruct_pragma_multilines
from pyccel.parser.utilities import read_file
from pyccel.parser.utilities import get_default_path

from pyccel.parser.syntax.headers import parse as hdr_parse
from pyccel.parser.syntax.openmp  import parse as omp_parse
from pyccel.parser.syntax.openacc import parse as acc_parse

from pyccel.errors.errors import Errors, PyccelSyntaxError

# TODO - remove import * and only import what we need
#      - use OrderedDict whenever it is possible
from pyccel.errors.messages import *

#==============================================================================
errors = Errors()
#==============================================================================

strip_ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]|[\n\t\r]')

redbaron.ipython_behavior = False

# use this to delete ansi_escape characters from a string
# Useful for very coarse version differentiation.

#==============================================================================

def change_priority( expr ):
    """
       RedBaron parses an expression from right to left
       this function makes sure that we evaluate our expression
       from left to right based in the priority of the operator.

       Examples
       --------
       >>> change_priority(PyccelMinus(1,PyccelMinus(1,1)))
       PyccelMinus(PyccelMinus(1,1),1)

       >>> change_priority(PyccelMinus(1,PyccelMul(1,1)))
       PyccelMinus(1,PyccelMul(1,1))

       >>> change_priority(PyccelDiv(1,PyccelMul(1,1)))
       PyccelMul(PyccelDiv(1, 1), 1)

    """
    first  = expr.args[0]
    second = expr.args[1]
    if isinstance(second, PyccelOperator) and second.p<=expr.p:
            a    = first
            b    = second.args[0]
            c    = second.args[1]
            a    = expr.func(a,b)
            a    = change_priority(a)
            return second.func(a,c)
    else:
        return expr

class SyntaxParser(BasicParser):

    """ Class for a Syntax Parser.

        inputs: str
            filename or code to parse as a string
    """

    def __init__(self, inputs, **kwargs):
        BasicParser.__init__(self, **kwargs)

        # check if inputs is a file
        code = inputs
        if os.path.isfile(inputs):

            # we don't use is_valid_filename_py since it uses absolute path
            # file extension

            ext = inputs.split(""".""")[-1]
            if not ext in ['py', 'pyh']:
                errors = Errors()
                errors.report(INVALID_FILE_EXTENSION, symbol=ext,
                              severity='fatal')
                errors.check()

            code = read_file(inputs)
            self._filename = inputs

        self._code = code

        self._scope = []

        tree = extend_tree(code)

        #preprocess_imports(red)

        #red = fst_move_directives(red)
        self._fst = tree
        

        self.parse(verbose=True)

    def parse(self, verbose=False):
        """converts redbaron fst to sympy ast."""

        if self.syntax_done:
            print ('> syntax analysis already done')
            return self.ast

        # TODO - add settings to Errors
        #      - filename

        errors = Errors()
        if self.filename:
            errors.set_target(self.filename, 'file')
        errors.set_parser_stage('syntax')

        # we add the try/except to allow the parser to find all possible errors
        PyccelAstNode.stage = 'syntactic'
        ast = self._visit(self.fst)

        self._ast = ast

        self._visit_done = True

        return ast

    def _treat_iterable(self, stmt):

        """
        since redbaron puts the first comments after a block statement
        inside the block, we need to remove them. this is in particular the
        case when using openmp/openacc pragmas like #$ omp end loop
        """

        ls = [self._visit(i) for i in stmt]

        if isinstance(stmt, list):
            return ls
        elif isinstance(stmt, (tuple, TupleNode)):
            return PythonTuple(*ls)
        else:
            return Tuple(*ls, sympify=False)

    def _visit(self, stmt):
        """Creates AST from FST."""

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors

        cls = type(stmt)
        syntax_method = '_visit_' + cls.__name__
        if hasattr(self, syntax_method):
            self._scope.append(cls)
            result = getattr(self, syntax_method)(stmt)
            self._scope.pop()
            return result

        # Unknown object, we raise an error.
        if hasattr(stmt, 'lineno'):
            bounding_box = (stmt.lineno, stmt.col_offset)
        else:
            bounding_box = None
        errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX, symbol=ast.dump(stmt),
                      bounding_box=bounding_box,
                      severity='fatal')

    def _visit_Module(self, stmt):
        """ Visits the ast and splits the result into elements relevant for the module or the program"""
        prog = []
        mod  = []
        start = []
        current_file = start
        targets = []
        n_empty_lines = 0
        is_prog = False
        for i in stmt.body:
            v = self._visit(i)
            if n_empty_lines > 3:
                current_file = start
            if isinstance(v,(FunctionDef, ClassDef)):
                # Functions and classes are always defined in a module
                n_empty_lines = 0
                mod.append(v)
                targets.append(v.name)
                current_file = mod
            elif isinstance(v,(Header, Comment, CommentBlock)):
                # Headers and Comments are defined in the same block as the following object
                n_empty_lines = 0
                current_file = start
                current_file.append(v)
            elif isinstance(v, (NewLine, EmptyLine)):
                # EmptyLines are defined in the same block as the previous line
                current_file.append(v)
                n_empty_lines += 1
            elif isinstance(v, Import):
                # Imports are defined in both the module and the program
                n_empty_lines = 0
                mod.append(v)
                prog.append(v)
            else:
                # Everything else is defined in a module
                is_prog = True
                n_empty_lines = 0
                prog.append(v)
                current_file = prog

            # If the current file is now a program or a module. Add headers and comments before the line we just read
            if len(start)>0 and current_file is not start:
                current_file[-1:-1] = start
                start = []
        if len(start)>0:
            mod.extend(start)

        # Define the names of the module and program
        # The module name allows it to be correctly referenced from an import command
        current_mod_name = os.path.splitext(os.path.basename(self._filename))[0]
        prog_name = 'prog_' + current_mod_name
        mod_code = CodeBlock(mod) if len(targets)>0 else None
        if is_prog:
            if mod_code:
                expr = Import(targets, source=current_mod_name)
                prog.insert(0,expr)
            prog_code = CodeBlock(prog)
            prog_code.set_fst(stmt)
        else:
            prog_code = None
            # If the file only contains headers
            if mod_code is None:
                mod_code = CodeBlock(mod)
        assert( mod_code is not None or prog_code is not None)
        code = ParserResult(program   = prog_code,
                            module    = mod_code,
                            prog_name = prog_name,
                            mod_name  = current_mod_name)
        code.set_fst(stmt)
        code._fst.lineno=1
        code._fst.col_offset=1
        return code

    def _visit_Expr(self, stmt):
        return self._visit(stmt.value)

    def _visit_LineProxyList(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_CommaProxyList(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_NodeList(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_Tuple(self, stmt):
        return PythonTuple(*self._treat_iterable(stmt.elts))

    def _visit_List(self, stmt):
        return List(*self._treat_iterable(stmt.elts), sympify=False)

    def _visit_tuple(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_list(self, stmt):
        return self._treat_iterable(stmt)


    def _visit_DottedAsNameNode(self, stmt):

        names = []
        for a in stmt.value:
            names.append(strip_ansi_escape.sub('', a.value))

        if len(names) == 1:
            return names[0]
        else:

            return DottedName(*names)

    def _visit_alias(self, stmt):
        if not isinstance(stmt.name, str):
            raise TypeError('Expecting a string')

        old = self._visit(stmt.name)

        if stmt.asname:
            new = self._visit(stmt.asname)
            return AsName(new, old)
        else:
            return old

    def _visit_Dict(self, stmt):

        d = {}
        for key, value in zip(stmt.keys, stmt.values):

            key = self._visit(key)
            value = self._visit(value)

            # sympy does not allow keys to be strings

            if isinstance(key, String):
                errors.report(SYMPY_RESTRICTION_DICT_KEYS,
                              severity='error')

            d[key] = value
        return Dict(d)

    def _visit_NoneType(self, stmt):
        return Nil()

    def _visit_str(self, stmt):

        return stmt

    def _visit_Str(self, stmt):
        val =  stmt.s
        if isinstance(self._scope[-1],(ast.Module, ast.FunctionDef)):
            return CommentBlock(val)
        return String(val)

    def _visit_Num(self, stmt):
        val = stmt.n

        if isinstance(val, int):
            return Integer(val)
        elif isinstance(val, float):
            return Float(val)
        elif isinstance(val, complex):
            return Complex(Float(val.real), Float(val.imag))
        else:
            raise NotImplementedError('Num type {} not recognised'.format(type(val)))

    def _visit_Assign(self, stmt):

        lhs = self._visit(stmt.targets)
        if len(lhs)==1:
            lhs = lhs[0]
        else:
            lhs = PythonTuple(*lhs)

        rhs = self._visit(stmt.value)
        expr = Assign(lhs, rhs)

        # we set the fst to keep track of needed information for errors

        expr.set_fst(stmt)
        return expr

    def _visit_AugAssign(self, stmt):

        lhs = self._visit(stmt.target)
        rhs = self._visit(stmt.value)
        if isinstance(stmt.op, ast.Add):
            expr = AugAssign(lhs, '+', rhs)
        elif isinstance(stmt.op, ast.Sub):
            expr = AugAssign(lhs, '-', rhs)
        elif isinstance(stmt.op, ast.Mult):
            expr = AugAssign(lhs, '*', rhs)
        elif isinstance(stmt.op, ast.Div):
            expr = AugAssign(lhs, '/', rhs)
        elif isinstance(stmt.op, ast.Mod):
            expr = AugAssign(lhs, '%', rhs)
        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol = stmt,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='fatal')

        # we set the fst to keep track of needed information for errors

        expr.set_fst(stmt)
        return expr

    def _visit_arguments(self, stmt):
        arguments = []
        if stmt.vararg or stmt.kwarg:
            errors.report(VARARGS, symbol = stmt,
                    bounding_box=(stmt.lineno, stmt.col_offset),
                    severity='fatal')

        if stmt.args:
            n_expl = len(stmt.args)-len(stmt.defaults)
            arguments += [Argument(a.arg) for a in stmt.args[:n_expl]]
            arguments += [ValuedArgument(Argument(a.arg),self._visit(d)) for a,d in zip(stmt.args[n_expl:],stmt.defaults)]
        elif stmt.kwonlyargs:
            arguments += [ValuedArgument(Argument(a.arg),self._visit(d)) for a,d in zip(stmt.kwonlyargs,stmt.kw_defaults)]

        return arguments

    def _visit_Constant(self, stmt):
        # New in python3.8 this class contains NameConstant, Num, and String types
        if stmt.value is None:
            return Nil()

        elif stmt.value is True:
            return BooleanTrue()

        elif stmt.value is False:
            return BooleanFalse()

        elif isinstance(stmt.value, int):
            return Integer(stmt.value)

        elif isinstance(stmt.value, float):
            return Float(stmt.value)

        elif isinstance(stmt.value, complex):
            return Complex(Float(stmt.value.real), Float(stmt.value.imag))

        elif isinstance(stmt.value, str):
            return self._visit_Str(stmt)

        else:
            raise NotImplementedError('Constant type {} not recognised'.format(type(stmt.value)))

    def _visit_NameConstant(self, stmt):
        if stmt.value is None:
            return Nil()

        elif stmt.value is True:
            return BooleanTrue()

        elif stmt.value is False:
            return BooleanFalse()

        else:
            raise NotImplementedError("Unknown NameConstant : {}".format(stmt.value))


    def _visit_Name(self, stmt):
        return Symbol(stmt.id)

    def _visit_Import(self, stmt):
        errors.report(PYCCEL_UNEXPECTED_IMPORT,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

        ls = [self._visit(n) for n in stmt.names]
        ls = [get_default_path(l) for l in ls]
        expr = [Import(i) for i in ls]
        for e in expr:
            e.set_fst(stmt)
            self.insert_import(e)
        if len(expr)==1:
            return expr[0]
        else:
            expr = CodeBlock(expr)
            expr.set_fst(stmt)
            return expr

    def _visit_ImportFrom(self, stmt):

        source = stmt.module

        if stmt.module.count('.') == 0:
            source = Symbol(stmt.module)
        else:
            source = DottedName(*stmt.module.split('.'))

        source = get_default_path(source)
        targets = []
        for i in stmt.names:
            s = self._visit(i)
            if s == '*':
                errors.report(PYCCEL_RESTRICTION_IMPORT_STAR,
                              bounding_box=(stmt.lineno, stmt.col_offset),
                              severity='error')

            targets.append(s)

        expr = Import(targets, source=source)
        expr.set_fst(stmt)
        self.insert_import(expr)
        return expr

    def _visit_Delete(self, stmt):
        arg = self._visit(stmt.targets)
        return Del(arg)

    def _visit_UnaryOp(self, stmt):

        target = self._visit(stmt.operand)
        if isinstance(stmt.op, ast.Not):
            return PyccelUnary(PyccelNot(target))
        if isinstance(stmt.op, ast.UAdd):
            return PyccelUnary(target)
        if isinstance(stmt.op, ast.USub):
            return PyccelUnary(PyccelMinus(target))
        if isinstance(stmt.op, ast.Invert):

            errors.report(PYCCEL_RESTRICTION_UNARY_OPERATOR,
                          bounding_box=(stmt.lineno, stmt.col_offset),
                          severity='error')
        else:
            errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                          bounding_box=(stmt.lineno, stmt.col_offset),
                          severity='fatal')

    def _visit_BinOp(self, stmt):

        first  = self._visit(stmt.left)
        second = self._visit(stmt.right)

        if isinstance(stmt.op, ast.Add):
            expr = PyccelAdd(first, second)
            return change_priority(expr)

        elif isinstance(stmt.op, ast.Mult):
            expr = PyccelMul(first, second)
            return change_priority(expr)

        elif isinstance(stmt.op, ast.Sub):
            expr = PyccelMinus(first, second)
            return change_priority(expr)

        elif isinstance(stmt.op, ast.Div):
            expr = PyccelDiv(first, second)
            return change_priority(expr)

        elif isinstance(stmt.op, ast.Pow):
            expr = PyccelPow(first, second)
            return change_priority(expr)

        elif isinstance(stmt.op, ast.FloorDiv):
            expr = PyccelFloorDiv(first, second)
            return change_priority(expr)

        elif isinstance(stmt.op, ast.Mod):
            expr = PyccelMod(first, second)
            return change_priority(expr)
        else:
            errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                          bounding_box=(stmt.lineno, stmt.col_offset),
                          severity='fatal')

    def _visit_BoolOp(self, stmt):

        first = self._visit(stmt.values[0])
        second = self._visit(stmt.values[1])

        if isinstance(stmt.op, ast.And):
            if isinstance(second, PyccelOr):
                args  = second.args
                first = PyccelAnd(first, args[0]  )
                return  PyccelOr (first, *args[1:])
            else:
                return PyccelAnd(first, second)

        if isinstance(stmt.op, ast.Or):
            if isinstance(second, PyccelAnd):
                args  = second.args
                first = PyccelOr(first, args[0])
                return  PyccelAnd(first, *args[1:])
            else:
                return PyccelOr(first, second)

        errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                      symbol = ast.dump(stmt.op),
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='fatal')

    def _visit_Compare(self, stmt):
        if len(stmt.ops)>1:
            errors.report(PYCCEL_RESTRICTION_MULTIPLE_COMPARISONS,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='fatal')

        first = self._visit(stmt.left)
        second = self._visit(stmt.comparators[0])
        op = stmt.ops[0]

        if isinstance(op, ast.Eq):
            return PyccelEq(first, second)
        if isinstance(op, ast.NotEq):
            return PyccelNe(first, second)
        if isinstance(op, ast.Lt):
            return PyccelLt(first, second)
        if isinstance(op, ast.Gt):
            return PyccelGt(first, second)
        if isinstance(op, ast.LtE):
            return PyccelLe(first, second)
        if isinstance(op, ast.GtE):
            return PyccelGe(first, second)
        if isinstance(op, ast.Is):
            return Is(first, second)
        if isinstance(op, ast.IsNot):
            return IsNot(first, second)

        errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='fatal')

    def _visit_AssociativeParenthesisNode(self, stmt):
        return PyccelAssociativeParenthesis(self._visit(stmt.value))

    def _visit_DefArgumentNode(self, stmt):
        name = str(self._visit(stmt.target))
        name = strip_ansi_escape.sub('', name)
        arg = Argument(name)
        if stmt.value is None:
            return arg
        else:

            value = self._visit(stmt.value)
            return ValuedArgument(arg, value)

    def _visit_Return(self, stmt):
        results = self._visit(stmt.value)
        if not isinstance(results, (list, PythonTuple, List)):
            results = [results]
        assigns  = []
        new_vars = []
        for result in results:
            if not isinstance(result, Symbol):
                new_vars.append(create_variable(result))
                new_stmt  = Assign(new_vars[-1], result)
                new_stmt.set_fst(stmt)
                assigns.append(new_stmt)
            else:
                new_vars.append(result)

        if assigns:
            expr = Return(new_vars, CodeBlock(assigns))
        else:
            expr = Return(new_vars)
        expr.set_fst(stmt)
        return expr

    def _visit_Pass(self, stmt):
        return Pass()

    def _visit_FunctionDef(self, stmt):

        #  TODO check all inputs and which ones should be treated in stage 1 or 2
        if isinstance(self._scope[-1], ast.ClassDef):
            cls_name = stmt.parent.name
        else:
            cls_name = None

        name = self._visit(stmt.name)
        name = name.replace("'", '')

        arguments    = self._visit(stmt.args)
        results      = []
        local_vars   = []
        global_vars  = []
        header       = None
        hide         = False
        kind         = 'function'
        is_pure      = False
        is_elemental = False
        is_private   = False
        imports      = []

        # TODO improve later
        decorators = {str(d) if isinstance(d,Symbol) else str(type(d)): d \
                            for d in self._visit(stmt.decorator_list)}

        if 'bypass' in decorators:
            return EmptyLine()

        if 'stack_array' in decorators:
            args = list(decorators['stack_array'].args)
            for i in range(len(args)):
                args[i] = str(args[i]).replace("'", '')
            decorators['stack_array'] = tuple(args)
        # extract the types to construct a header
        if 'types' in decorators:
            types = []
            results = []
            container = types
            i = 0
            ls = decorators['types'].args
            while i<len(ls) :
                arg = ls[i]

                if isinstance(arg, Symbol):
                    arg = arg.name
                    container.append(arg)
                elif isinstance(arg, String):
                    arg = str(arg)
                    arg = arg.strip("'").strip('"')
                    container.append(arg)
                elif isinstance(arg, ValuedArgument):
                    arg_name = arg.name
                    arg  = arg.value
                    container = results
                    if not arg_name == 'results':
                        msg = 'Argument "{}" provided to the types decorator is not valid'.format(arg_name)
                        errors.report(msg,
                                      bounding_box=(stmt.lineno, stmt.col_offset),
                                      severity='error')
                    else:
                        ls = arg if isinstance(arg, PythonTuple) else [arg]
                        i = -1
                else:
                    msg = 'Invalid argument of type {} passed to types decorator'.format(type(arg))
                    errors.report(msg,
                                  bounding_box=(stmt.lineno, stmt.col_offset),
                                  severity='error')

                i = i+1

            txt  = '#$ header ' + name
            txt += '(' + ','.join(types) + ')'

            if results:
                txt += ' results(' + ','.join(results) + ')'

            header = hdr_parse(stmts=txt)
            if name in self.namespace.static_functions:
                header = header.to_static()

        body = stmt.body

        if 'sympy' in decorators.keys():
            # TODO maybe we should run pylint here
            stmt.decorators.pop()
            func = SympyFunction(name, arguments, [],
                    [stmt.__str__()])
            func.set_fst(stmt)
            self.insert_function(func)
            return EmptyLine()

        elif 'python' in decorators.keys():

            # TODO maybe we should run pylint here

            stmt.decorators.pop()
            func = PythonFunction(name, arguments, [],
                    [stmt.__str__()])
            func.set_fst(stmt)
            self.insert_function(func)
            return EmptyLine()

        else:
            body = self._visit(body)

        if 'pure' in decorators.keys():
            is_pure = True

        if 'elemental' in decorators.keys():
            is_elemental = True

        if 'private' in decorators.keys():
            is_private = True

        func = FunctionDef(
               name,
               arguments,
               results,
               body,
               local_vars=local_vars,
               global_vars=global_vars,
               cls_name=cls_name,
               hide=hide,
               kind=kind,
               is_pure=is_pure,
               is_elemental=is_elemental,
               is_private=is_private,
               imports=imports,
               decorators=decorators,
               header=header)

        func.set_fst(stmt)
        return func

    def _visit_ClassDef(self, stmt):

        name = stmt.name
        methods = [i for i in stmt.body if isinstance(i, ast.FunctionDef)]
        methods = self._visit(methods)
        attributes = methods[0].arguments
        parent = [self._visit(i) for i in stmt.bases]
        expr = ClassDef(name=name, attributes=attributes,
                        methods=methods, parent=parent)

        # we set the fst to keep track of needed information for errors

        expr.set_fst(stmt)
        return expr

    def _visit_AtomtrailersNode(self, stmt):

        return self._visit(stmt.value)

    def _visit_Subscript(self, stmt):

        ch = stmt
        args = []
        while isinstance(ch, ast.Subscript):
            val = self._visit(ch.slice)
            if isinstance(val, (Tuple, PythonTuple)):
                args += val
            else:
                args.insert(0, val)
            ch = ch.value
        args = tuple(args)
        var = self._visit(ch)
        var = IndexedBase(var)[args]
        return var

    def _visit_ExtSlice(self, stmt):
        return self._visit(tuple(stmt.dims))

    def _visit_Slice(self, stmt):

        upper = self._visit(stmt.upper)
        lower = self._visit(stmt.lower)

        if stmt.step is not None:
            raise NotImplementedError("Steps in slices are not implemented")

        if not isinstance(upper, Nil) and not isinstance(lower, Nil):

            return Slice(lower, upper)
        elif not isinstance(lower, Nil):

            return Slice(lower, None)
        elif not isinstance(upper, Nil):

            return Slice(None, upper)
        else:

            return Slice(None, None)

    def _visit_Index(self, stmt):
        return self._visit(stmt.value)

    def _visit_DotProxyList(self, stmt):

        n = len(stmt) - 1
        ls = []
        while n > -1:
            if isinstance(stmt[n], GetitemNode):
                args = self._visit(stmt[n])
                while isinstance(stmt[n].previous, GetitemNode):
                    n = n - 1
                var = self._visit(stmt[:n])
                var = IndexedBase(var)[args]
                n = 0
            elif isinstance(stmt[n], CallNode):
                var = self._visit(stmt[n])
                n = n - 1
            else:
                while n > 0 and not isinstance(stmt[n].previous,
                        DotNode) and not isinstance(stmt[n], (CallNode)):
                    n = n - 1
                var = self._visit(stmt[n])
            ls.insert(0,var)
            n = n - 1

        if len(ls) == 1:
            expr = ls[0]
        else:
            n = 0
            var = DottedVariable(ls[0], ls[1])
            n = 2
            while n < len(ls):
                var = DottedVariable(var, ls[n])
                n = n + 1

            expr = var
        return expr

    def _visit_Call(self, stmt):

        args = []
        if stmt.args:
            args += self._visit(stmt.args)
        if stmt.keywords:
            args += self._visit(stmt.keywords)
        f_name = self._visit(stmt.func)
        if len(args) == 0:
            args = ( )

        if str(f_name) == "print":
            func = Print(PythonTuple(*args))
        else:
            func = Function(f_name)(*args)

        return func

    def _visit_keyword(self, stmt):

        target = stmt.arg
        val = self._visit(stmt.value)
        return ValuedArgument(target, val)

    def _visit_For(self, stmt):

        iterator = self._visit(stmt.target)
        iterable = self._visit(stmt.iter)
        body = self._visit(stmt.body)
        expr = For(iterator, iterable, body, strict=False)
        expr.set_fst(stmt)
        return expr

    def _visit_ComprehensionLoopNode(self, stmt):

        iterator = self._visit(stmt.iterator)
        iterable = self._visit(stmt.target)
        ifs = stmt.ifs
        expr = For(iterator, iterable, [], strict=False)
        expr.set_fst(stmt)
        return expr

    def _visit_ArgumentGeneratorComprehensionNode(self, stmt):

        result = self._visit(stmt.result)

        generators = self._visit(stmt.generators)
        parent = stmt.parent.parent.parent

        if isinstance(parent, AssignmentNode):
            lhs = self._visit(parent.target)
            name = strip_ansi_escape.sub('', parent.value[0].value)
            cond = False
        else:
            lhs = create_variable(result)
            name = stmt.parent.parent
            name = strip_ansi_escape.sub('', name.value[0].value)
            cond = True
        body = result
        if name == 'sum':
            body = AugAssign(lhs, '+', body)
        else:
            body = Function(name)(lhs, body)
            body = Assign(lhs, body)

        body.set_fst(parent)
        indices = []
        generators = list(generators)
        while len(generators) > 0:
            indices.append(generators[-1].target)
            generators[-1].insert2body(body)
            body = generators.pop()
        indices = indices[::-1]
        body = [body]
        if name == 'sum':
            expr = FunctionalSum(body, result, lhs, indices, None)
        elif name == 'min':
            expr = FunctionalMin(body, result, lhs, indices, None)
        elif name == 'max':
            expr = FunctionalMax(body, result, lhs, indices, None)
        else:
            errors.report(PYCCEL_RESTRICTION_TODO,
                          bounding_box=(stmt.lineno, stmt.col_offset),
                          severity='fatal')

        expr.set_fst(stmt)
        return expr

    def _visit_If(self, stmt):

        test = self._visit(stmt.test)
        body = self._visit(stmt.body)
        orelse = self._visit(stmt.orelse)
        if isinstance(orelse,If):
            orelse = orelse._args
        else:
            orelse = Tuple(BooleanTrue(), orelse, sympify=False)

        return If(Tuple(test, body, sympify=False), orelse)

    def _visit_TernaryOperatorNode(self, stmt):

        test1 = self._visit(stmt.value)
        first = self._visit(stmt.first)
        second = self._visit(stmt.second)
        args = [Tuple(test1, [first], sympify=False),
                Tuple(BooleanTrue(), [second], sympify=False)]
        expr = IfTernaryOperator(*args)
        expr.set_fst(stmt)
        return expr

    def _visit_While(self, stmt):

        test = self._visit(stmt.test)
        body = self._visit(stmt.body)
        return While(test, body)

    def _visit_Assert(self, stmt):
        expr = self._visit(stmt.test)
        return Assert(expr)

    def _visit_EndlNode(self, stmt):

        return NewLine()

    def _visit_CommentLine(self, stmt):

        # if annotated comment

        if stmt.s.startswith('#$'):
            env = stmt.s[2:].lstrip()
            if env.startswith('omp'):
                txt = reconstruct_pragma_multilines(stmt)
                return omp_parse(stmts=txt)
            elif env.startswith('acc'):

                txt = reconstruct_pragma_multilines(stmt)
                return acc_parse(stmts=txt)
            elif env.startswith('header'):

                txt = reconstruct_pragma_multilines(stmt)
                expr = hdr_parse(stmts=txt)
                if isinstance(expr, MetaVariable):

                    # a metavar will not appear in the semantic stage.
                    # but can be used to modify the ast

                    self._metavars[str(expr.name)] = str(expr.value)

                    # return NewLine()

                    expr = EmptyLine()
                else:
                    expr.set_fst(stmt)

                return expr
            else:

                # TODO an info should be reported saying that either we
                # found a multiline pragma or an invalid pragma statement

                return NewLine()
        else:

#                    errors.report(PYCCEL_INVALID_HEADER,
#                                  bounding_box=(stmt.lineno, stmt.col_offset),
#                                  severity='error')

            # TODO improve

            txt = stmt.s[1:].lstrip()
            return Comment(txt)

    def _visit_Break(self, stmt):
        return Break()

    def _visit_ContinueNode(self, stmt):
        return Continue()

    def _visit_StarNode(self, stmt):
        return '*'

    def _visit_LambdaNode(self, stmt):

        expr = self._visit(stmt.value)
        args = []

        for i in stmt.arguments:
            var = self._visit(i.name)
            args += [var]

        return Lambda(tuple(args), expr)

    def _visit_WithNode(self, stmt):
        domain = self._visit(stmt.contexts[0].value)
        body = self._visit(stmt.value)
        settings = None
        return With(domain, body, settings)

    def _visit_ListComprehensionNode(self, stmt):

        import numpy as np
        result = self._visit(stmt.result)
        generators = list(self._visit(stmt.generators))
        lhs = self._visit(stmt.parent.target)
        index = create_variable(lhs)
        if isinstance(result, (PythonTuple, Tuple, list, tuple)):
            rank = len(np.shape(result))
        else:
            rank = 0
        args = [Slice(None, None)] * rank
        args.append(index)
        target = IndexedBase(lhs)[args]
        target = Assign(target, result)
        assign1 = Assign(index, Integer(0))
        assign1.set_fst(stmt)
        target.set_fst(stmt)
        generators[-1].insert2body(target)
        assign2 = Assign(index, PyccelAdd(index, Integer(1)))
        assign2.set_fst(stmt)
        generators[-1].insert2body(assign2)

        indices = [generators[-1].target]
        while len(generators) > 1:
            F = generators.pop()
            generators[-1].insert2body(F)
            indices.append(generators[-1].target)
        indices = indices[::-1]
        return FunctionalFor([assign1, generators[-1]],target.rhs, target.lhs,
                             indices, index)

    def _visit_TryNode(self, stmt):
        # this is a blocking error, since we don't want to convert the try body
        errors.report(PYCCEL_RESTRICTION_TRY_EXCEPT_FINALLY,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_RaiseNode(self, stmt):
        errors.report(PYCCEL_RESTRICTION_RAISE,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_YieldAtomNode(self, stmt):
        errors.report(PYCCEL_RESTRICTION_YIELD,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_YieldNode(self, stmt):
        errors.report(PYCCEL_RESTRICTION_YIELD,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_ListArgumentNode(self, stmt):
        return StarredArguments(self._visit(stmt.value))

#==============================================================================


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        raise ValueError('Expecting an argument for filename')

    parser = SyntaxParser(filename)
    print(parser.ast)


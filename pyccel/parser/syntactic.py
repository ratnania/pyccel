# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

# pylint: disable=R0201, missing-function-docstring

import os
import re

import ast

#==============================================================================

from sympy.core import cache

#==============================================================================

from pyccel.ast.basic import Basic, PyccelAstNode

from pyccel.ast.core import FunctionCall
from pyccel.ast.core import ParserResult
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Return
from pyccel.ast.core import Pass
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import PythonFunction, SympyFunction
from pyccel.ast.core import ClassDef
from pyccel.ast.core import For, FunctionalFor
from pyccel.ast.core import If, IfSection
from pyccel.ast.core import While
from pyccel.ast.core import Del
from pyccel.ast.core import Assert
from pyccel.ast.core import Comment, EmptyNode
from pyccel.ast.core import Break, Continue
from pyccel.ast.core import Argument, ValuedArgument
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.core import CommentBlock
from pyccel.ast.core import With
from pyccel.ast.core import StarredArguments
from pyccel.ast.core import CodeBlock
from pyccel.ast.core import IndexedElement
from pyccel.ast.core import create_variable

from pyccel.ast.bitwise_operators import PyccelRShift, PyccelLShift, PyccelBitXor, PyccelBitOr, PyccelBitAnd, PyccelInvert
from pyccel.ast.operators import PyccelPow, PyccelAdd, PyccelMul, PyccelDiv, PyccelMod, PyccelFloorDiv
from pyccel.ast.operators import PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe
from pyccel.ast.operators import PyccelAnd, PyccelOr,  PyccelNot, PyccelMinus
from pyccel.ast.operators import PyccelUnary, PyccelUnarySub
from pyccel.ast.operators import PyccelIs, PyccelIsNot
from pyccel.ast.operators import IfTernaryOperator

from pyccel.ast.builtins import PythonTuple, PythonList
from pyccel.ast.builtins import PythonPrint, Lambda
from pyccel.ast.headers  import Header, MetaVariable
from pyccel.ast.literals import LiteralInteger, LiteralFloat, LiteralComplex
from pyccel.ast.literals import LiteralFalse, LiteralTrue, LiteralString
from pyccel.ast.literals import Nil
from pyccel.ast.functionalexpr import FunctionalSum, FunctionalMax, FunctionalMin
from pyccel.ast.variable  import DottedName

from pyccel.ast.internals import Slice, PyccelSymbol

from pyccel.parser.extend_tree import extend_tree
from pyccel.parser.base import BasicParser
from pyccel.parser.utilities import read_file
from pyccel.parser.utilities import get_default_path

from pyccel.parser.syntax.headers import parse as hdr_parse
from pyccel.parser.syntax.openmp  import parse as omp_parse
from pyccel.parser.syntax.openacc import parse as acc_parse

from pyccel.errors.errors import Errors

# TODO - remove import * and only import what we need
#      - use OrderedDict whenever it is possible
from pyccel.errors.messages import *

def get_name(a):
    """ get the name of variable or an argument of the AST node."""
    if isinstance(a, ast.Name):
        return a.id
    elif isinstance(a, ast.arg):
        return a.arg
    else:
        raise NotImplementedError()

#==============================================================================
errors = Errors()
#==============================================================================

strip_ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]|[\n\t\r]')

# use this to delete ansi_escape characters from a string
# Useful for very coarse version differentiation.

#==============================================================================

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

            self._filename = inputs
            errors.set_target(self.filename, 'file')

            # we don't use is_valid_filename_py since it uses absolute path
            # file extension

            code = read_file(inputs)

        self._code  = code
        self._scope = []

        self.load()

        tree                = extend_tree(code)
        self._fst           = tree
        self._used_names    = set(get_name(a) for a in ast.walk(self._fst) if isinstance(a, (ast.Name, ast.arg)))
        self._dummy_counter = 1

        self.parse(verbose=True)
        self.dump()

    def parse(self, verbose=False):
        """converts python ast to sympy ast."""

        if self.syntax_done:
            return self.ast

        # TODO - add settings to Errors
        #      - filename
        errors.set_parser_stage('syntax')

        PyccelAstNode.stage = 'syntactic'
        ast       = self._visit(self.fst)
        self._ast = ast

        self._visit_done = True
        self._syntax_done = True

        return ast

    def _treat_iterable(self, stmt):
        return (self._visit(i) for i in stmt)

    def _visit(self, stmt):
        """Creates AST from FST."""

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors

        cls = type(stmt)
        syntax_method = '_visit_' + cls.__name__
        if hasattr(self, syntax_method):
            self._scope.append(stmt)
            result = getattr(self, syntax_method)(stmt)
            if isinstance(result, Basic) and isinstance(stmt, ast.AST):
                result.set_fst(stmt)
            self._scope.pop()
            return result

        # Unknown object, we raise an error.
        errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX, symbol=stmt,
                      severity='fatal')

    def _visit_Module(self, stmt):
        """ Visits the ast and splits the result into elements relevant for the module or the program"""
        prog          = []
        mod           = []
        start         = []
        current_file  = start
        targets       = []
        n_empty_lines = 0
        is_prog       = False
        body          = [self._visit(v) for v in stmt.body]

        # Define the names of the module and program
        # The module name allows it to be correctly referenced from an import command
        current_mod_name = os.path.splitext(os.path.basename(self._filename))[0]
        prog_name = 'prog_' + current_mod_name

        new_body      = []
        for i in body:
            if isinstance(i, CodeBlock):
                new_body += list(i.body)
            else:
                new_body.append(i)

        body = new_body
        for v in body:

            if n_empty_lines > 3:
                current_file = start
            if isinstance(v,(FunctionDef, ClassDef)):
                # Functions and classes are always defined in a module
                n_empty_lines = 0
                mod.append(v)
                targets.append(v.name)
                current_file = mod
                im = Import(source=current_mod_name, target = [v.name])
                prog.append(im)
            elif isinstance(v,(Header, Comment, CommentBlock)):
                # Headers and Comments are defined in the same block as the following object
                n_empty_lines = 0
                current_file = start
                current_file.append(v)
            elif isinstance(v, EmptyNode):
                # EmptyNodes are defined in the same block as the previous line
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

        mod_code = CodeBlock(mod) if len(targets)>0 else None
        if is_prog:
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
        return code

    def _visit_Expr(self, stmt):
        return self._visit(stmt.value)

    def _visit_Tuple(self, stmt):
        return PythonTuple(*self._treat_iterable(stmt.elts))

    def _visit_List(self, stmt):
        return PythonList(*self._treat_iterable(stmt.elts))

    def _visit_tuple(self, stmt):
        return tuple(self._treat_iterable(stmt))

    def _visit_list(self, stmt):
        return list(self._treat_iterable(stmt))

    def _visit_alias(self, stmt):
        if not isinstance(stmt.name, str):
            raise TypeError('Expecting a string')

        old = self._visit(stmt.name)

        if stmt.asname:
            new = self._visit(stmt.asname)
            return AsName(old, new)
        else:
            return old

    def _visit_Dict(self, stmt):
        errors.report(PYCCEL_RESTRICTION_TODO,
                symbol=stmt, severity='fatal')

    def _visit_NoneType(self, stmt):
        return Nil()

    def _visit_str(self, stmt):

        return stmt

    def _visit_Str(self, stmt):
        val =  stmt.s
        if isinstance(self._scope[-2], ast.Expr):
            return CommentBlock(val)
        return LiteralString(val)

    def _visit_Num(self, stmt):
        val = stmt.n

        if isinstance(val, int):
            return LiteralInteger(val)
        elif isinstance(val, float):
            return LiteralFloat(val)
        elif isinstance(val, complex):
            return LiteralComplex(val.real, val.imag)
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
                      severity='fatal')

        # we set the fst to keep track of needed information for errors

        return expr

    def _visit_arguments(self, stmt):

        if stmt.vararg or stmt.kwarg:
            errors.report(VARARGS, symbol = stmt,
                    severity='fatal')

        arguments       = []
        if stmt.args:
            n_expl = len(stmt.args)-len(stmt.defaults)
            positional_args        = [Argument(a.arg, annotation=self._visit(a.annotation)) for a in stmt.args[:n_expl]]
            valued_arguments       = [ValuedArgument(Argument(a.arg, annotation=self._visit(a.annotation)),\
                                      self._visit(d)) for a,d in zip(stmt.args[n_expl:],stmt.defaults)]
            arguments              = positional_args + valued_arguments

        if stmt.kwonlyargs:
            for a,d in zip(stmt.kwonlyargs,stmt.kw_defaults):
                annotation = self._visit(a.annotation)
                if d is not None:
                    arg = Argument(a.arg, annotation=annotation)
                    arg = ValuedArgument(arg, self._visit(d), kwonly=True)
                else:
                    arg = Argument(a.arg, kwonly=True, annotation=annotation)

                arguments.append(arg)

        return arguments

    def _visit_Constant(self, stmt):
        # New in python3.8 this class contains NameConstant, Num, and String types
        if stmt.value is None:
            return Nil()

        elif stmt.value is True:
            return LiteralTrue()

        elif stmt.value is False:
            return LiteralFalse()

        elif isinstance(stmt.value, int):
            return LiteralInteger(stmt.value)

        elif isinstance(stmt.value, float):
            return LiteralFloat(stmt.value)

        elif isinstance(stmt.value, complex):
            return LiteralComplex(stmt.value.real, stmt.value.imag)

        elif isinstance(stmt.value, str):
            return self._visit_Str(stmt)

        else:
            raise NotImplementedError('Constant type {} not recognised'.format(type(stmt.value)))

    def _visit_NameConstant(self, stmt):
        if stmt.value is None:
            return Nil()

        elif stmt.value is True:
            return LiteralTrue()

        elif stmt.value is False:
            return LiteralFalse()

        else:
            raise NotImplementedError("Unknown NameConstant : {}".format(stmt.value))


    def _visit_Name(self, stmt):
        return PyccelSymbol(stmt.id)

    def _treat_import_source(self, source, level):
        source = '.'*level + source
        if source.count('.') == 0:
            source = PyccelSymbol(source)
        else:
            source = DottedName(*source.split('.'))

        return get_default_path(source)

    def _visit_Import(self, stmt):
        expr = []
        for name in stmt.names:
            imp = self._visit(name)
            if isinstance(imp, AsName):
                source = AsName(self._treat_import_source(imp.name, 0), imp.target)
            else:
                source = self._treat_import_source(imp, 0)
            import_line = Import(source)
            import_line.set_fst(stmt)
            self.insert_import(import_line)
            expr.append(import_line)

        if len(expr)==1:
            return expr[0]
        else:
            expr = CodeBlock(expr)
            return expr

    def _visit_ImportFrom(self, stmt):

        source = self._treat_import_source(stmt.module, stmt.level)

        targets = []
        for i in stmt.names:
            s = self._visit(i)
            if s == '*':
                errors.report(PYCCEL_RESTRICTION_IMPORT_STAR,
                              bounding_box=(stmt.lineno, stmt.col_offset),
                              severity='error')

            targets.append(s)

        expr = Import(source, targets)
        self.insert_import(expr)
        return expr

    def _visit_Delete(self, stmt):
        arg = self._visit(stmt.targets)
        return Del(arg)

    def _visit_UnaryOp(self, stmt):

        target = self._visit(stmt.operand)

        if isinstance(stmt.op, ast.Not):
            Func = PyccelNot

        elif isinstance(stmt.op, ast.UAdd):
            Func = PyccelUnary

        elif isinstance(stmt.op, ast.USub):
            Func = PyccelUnarySub

        elif isinstance(stmt.op, ast.Invert):
            Func = PyccelInvert
        else:
            errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                          symbol = stmt,
                          severity='fatal')

        return Func(target)

    def _visit_BinOp(self, stmt):

        first  = self._visit(stmt.left)
        second = self._visit(stmt.right)

        if isinstance(stmt.op, ast.Add):
            return PyccelAdd(first, second)

        elif isinstance(stmt.op, ast.Mult):
            return PyccelMul(first, second)

        elif isinstance(stmt.op, ast.Sub):
            return PyccelMinus(first, second)

        elif isinstance(stmt.op, ast.Div):
            return PyccelDiv(first, second)

        elif isinstance(stmt.op, ast.Pow):
            return PyccelPow(first, second)

        elif isinstance(stmt.op, ast.FloorDiv):
            return PyccelFloorDiv(first, second)

        elif isinstance(stmt.op, ast.Mod):
            return PyccelMod(first, second)

        elif isinstance(stmt.op, ast.RShift):
            return PyccelRShift(first, second)

        elif isinstance(stmt.op, ast.LShift):
            return PyccelLShift(first, second)

        elif isinstance(stmt.op, ast.BitXor):
            return PyccelBitXor(first, second)

        elif isinstance(stmt.op, ast.BitOr):
            return PyccelBitOr(first, second)

        elif isinstance(stmt.op, ast.BitAnd):
            return PyccelBitAnd(first, second)

        else:
            errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                          symbol = stmt,
                          severity='fatal')

    def _visit_BoolOp(self, stmt):

        args = [self._visit(a) for a in stmt.values]

        if isinstance(stmt.op, ast.And):
            return PyccelAnd(*args)

        if isinstance(stmt.op, ast.Or):
            return PyccelOr(*args)

        errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                      symbol = stmt.op,
                      severity='fatal')

    def _visit_Compare(self, stmt):
        if len(stmt.ops)>1:
            errors.report(PYCCEL_RESTRICTION_MULTIPLE_COMPARISONS,
                      symbol = stmt,
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
            return PyccelIs(first, second)
        if isinstance(op, ast.IsNot):
            return PyccelIsNot(first, second)

        errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                      symbol = stmt,
                      severity='fatal')

    def _visit_Return(self, stmt):
        results = self._visit(stmt.value)
        if not isinstance(results, (list, PythonTuple, PythonList)):
            results = [results]
        expr = Return(results)
        return expr

    def _visit_Pass(self, stmt):
        return Pass()

    def _visit_FunctionDef(self, stmt):

        #  TODO check all inputs and which ones should be treated in stage 1 or 2

        name = self._visit(stmt.name)
        name = name.replace("'", '')

        arguments    = self._visit(stmt.args)

        local_vars   = []
        global_vars  = []
        headers      = []
        template    = {}
        is_pure      = False
        is_elemental = False
        is_private   = False
        imports      = []
        doc_string   = None

        def fill_types(ls):
            container = []
            for arg in ls:
                if isinstance(arg, PyccelSymbol):
                    container.append(arg)
                elif isinstance(arg, LiteralString):
                    arg = str(arg)
                    arg = arg.strip("'").strip('"')
                    container.append(arg)
                else:
                    msg = 'Invalid argument of type {} passed to types decorator'.format(type(arg))
                    errors.report(msg,
                                bounding_box = (stmt.lineno, stmt.col_offset),
                                severity='error')
            return container

        decorators = {}

        # add the decorator @types if the arguments are annotated
        annotated_args = []
        for a in arguments:
            if isinstance(a, Argument):
                annotated_args.append(a.annotation)
            elif isinstance(a, ValuedArgument):
                annotated_args.append(a.argument.annotation)

        if all(not isinstance(a, Nil) for a in annotated_args):
            if stmt.returns:
                returns = ValuedArgument(PyccelSymbol('results'),self._visit(stmt.returns))
                annotated_args.append(returns)
            decorators['types'] = [FunctionCall('types', annotated_args)]

        for d in self._visit(stmt.decorator_list):
            tmp_var = d if isinstance(d, PyccelSymbol) else d.funcdef
            if tmp_var in decorators:
                decorators[tmp_var] += [d]
            else:
                decorators[tmp_var] = [d]

        if 'bypass' in decorators:
            return EmptyNode()

        if 'stack_array' in decorators:
            decorators['stack_array'] = tuple(str(b) for a in decorators['stack_array']
                for b in a.args)

        if 'allow_negative_index' in decorators:
            decorators['allow_negative_index'] = tuple(str(b) for a in decorators['allow_negative_index'] for b in a.args)
        template['template_dict'] = {}
        # extract the templates
        if 'template' in decorators:
            for comb_types in decorators['template']:
                cache.clear_cache()
                types = []
                if len(comb_types.args) != 2:
                    msg = 'Number of Arguments provided to the template decorator is not valid'
                    errors.report(msg,
                                    symbol = comb_types,
                                    bounding_box = (stmt.lineno, stmt.col_offset),
                                    severity='error')

                for i in comb_types.args:
                    if isinstance(i, ValuedArgument) and not i.name in ('name',
                            'types'):
                        msg = 'Argument provided to the template decorator is not valid'
                        errors.report(msg,
                                        symbol = comb_types,
                                        bounding_box = (stmt.lineno, stmt.col_offset),
                                        severity='error')
                if all(isinstance(i, ValuedArgument) for i in comb_types.args):
                    tp_name, ls = (comb_types.args[0].value, comb_types.args[1].value) if\
                            comb_types.args[0].name == 'name' else\
                            (comb_types.args[1].value, comb_types.args[0].value)
                else:
                    tp_name = comb_types.args[0]
                    ls = comb_types.args[1]
                    ls = ls.value if isinstance(ls, ValuedArgument) else ls
                try:
                    tp_name = str(tp_name)
                    ls = ls if isinstance(ls, PythonTuple) else list(ls)
                except TypeError:
                    msg = 'Argument provided to the template decorator is not valid'
                    errors.report(msg,
                                    symbol = comb_types,
                                    bounding_box = (stmt.lineno, stmt.col_offset),
                                    severity='fatal')

                types = fill_types(ls)

                txt  = '#$ header template ' + tp_name
                txt += '(' + '|'.join(types) + ')'
                if tp_name in template['template_dict']:
                    msg = 'The template "{}" is duplicated'.format(tp_name)
                    errors.report(msg,
                                bounding_box = (stmt.lineno, stmt.col_offset),
                                severity='warning')
                # Make templates decorator dict accessible from decorators dict
                template['template_dict'][tp_name] = hdr_parse(stmts=txt)
            # Make template decorator list accessible from decorators dict
            template['decorator_list'] = decorators['template']
            decorators['template'] = template

        if not template['template_dict']:
            decorators['template'] = None
        # extract the types to construct a header
        if 'types' in decorators:
            for comb_types in decorators['types']:

                cache.clear_cache()
                results = []
                ls = comb_types.args

                if len(ls) > 0 and isinstance(ls[-1], ValuedArgument):
                    arg_name = ls[-1].name
                    if not arg_name == 'results':
                        msg = 'Argument "{}" provided to the types decorator is not valid'.format(arg_name)
                        errors.report(msg,
                                    symbol = comb_types,
                                    bounding_box = (stmt.lineno, stmt.col_offset),
                                    severity='error')
                    else:
                        container = ls[-1].value
                        container = container if isinstance(container, PythonTuple) else [container]
                        results = fill_types(container)
                    types = fill_types(ls[:-1])
                else:
                    types = fill_types(ls)

                txt  = '#$ header ' + name
                txt += '(' + ','.join(types) + ')'

                if results:
                    txt += ' results(' + ','.join(results) + ')'

                header = hdr_parse(stmts=txt)
                if name in self.namespace.static_functions:
                    header = header.to_static()
                headers += [header]

        body = stmt.body

        if 'sympy' in decorators.keys():
            # TODO maybe we should run pylint here
            stmt.decorators.pop()
            func = SympyFunction(name, arguments, [],
                    [stmt.__str__()])
            func.set_fst(stmt)
            self.insert_function(func)
            return EmptyNode()

        elif 'python' in decorators.keys():

            # TODO maybe we should run pylint here

            stmt.decorators.pop()
            func = PythonFunction(name, arguments, [],
                    [stmt.__str__()])
            func.set_fst(stmt)
            self.insert_function(func)
            return EmptyNode()

        else:
            body = self._visit(body)
        if len(body) > 0 and isinstance(body[0], CommentBlock):
            doc_string = body[0]
            doc_string.header = ''
            body = body[1:]

        if 'pure' in decorators.keys():
            is_pure = True

        if 'elemental' in decorators.keys():
            is_elemental = True
            if len(arguments) > 1:
                errors.report(FORTRAN_ELEMENTAL_SINGLE_ARGUMENT,
                              symbol=decorators['elemental'],
                              bounding_box=(stmt.lineno, stmt.col_offset),
                              severity='error')

        if 'private' in decorators.keys():
            is_private = True

        body = CodeBlock(body)

        returns = [i.expr for i in body.get_attribute_nodes(Return)]
        assert all(len(i) == len(returns[0]) for i in returns)
        results = []
        result_counter = 1

        for r in zip(*returns):
            r0 = r[0]

            pyccel_symbol  = isinstance(r0, PyccelSymbol)
            same_results   = all(r0 == ri for ri in r)
            name_available = all(r0 != a.name for a in arguments)

            if pyccel_symbol and same_results and name_available:
                result_name = r0
            else:
                result_name, result_counter = create_variable(self._used_names, \
                            prefix = 'Out', counter = result_counter)

            results.append(result_name)

        func = FunctionDef(
               name,
               arguments,
               results,
               body,
               local_vars=local_vars,
               global_vars=global_vars,
               is_pure=is_pure,
               is_elemental=is_elemental,
               is_private=is_private,
               imports=imports,
               decorators=decorators,
               headers=headers,
               doc_string=doc_string)

        return func

    def _visit_ClassDef(self, stmt):

        name = stmt.name
        methods = [self._visit(i) for i in stmt.body if isinstance(i, ast.FunctionDef)]
        for i in methods:
            i.cls_name = name
        attributes = methods[0].arguments
        parent = [self._visit(i) for i in stmt.bases]
        expr = ClassDef(name=name, attributes=attributes,
                        methods=methods, superclass=parent)

        # we set the fst to keep track of needed information for errors

        return expr

    def _visit_Subscript(self, stmt):

        ch = stmt
        args = []
        while isinstance(ch, ast.Subscript):
            val = self._visit(ch.slice)
            if isinstance(val, (PythonTuple, list)):
                args += val
            else:
                args.insert(0, val)
            ch = ch.value
        args = tuple(args)
        var = self._visit(ch)
        var = IndexedElement(var, *args)
        return var

    def _visit_ExtSlice(self, stmt):
        return self._visit(stmt.dims)

    def _visit_Slice(self, stmt):

        upper = self._visit(stmt.upper) if stmt.upper is not None else None
        lower = self._visit(stmt.lower) if stmt.lower is not None else None
        step = self._visit(stmt.step) if stmt.step is not None else None

        return Slice(lower, upper, step)

    def _visit_Index(self, stmt):
        return self._visit(stmt.value)

    def _visit_Attribute(self, stmt):
        val  = self._visit(stmt.value)
        attr = PyccelSymbol(stmt.attr)
        return DottedName(val, attr)


    def _visit_Call(self, stmt):

        args = []
        if stmt.args:
            args += self._visit(stmt.args)
        if stmt.keywords:
            args += self._visit(stmt.keywords)

        if len(args) == 0:
            args = ()

        func = self._visit(stmt.func)

        if isinstance(func, PyccelSymbol):
            if func == "print":
                func = PythonPrint(PythonTuple(*args))
            else:
                func = FunctionCall(func, args)
        elif isinstance(func, DottedName):
            func_attr = FunctionCall(func.name[-1], args)
            func = DottedName(*func.name[:-1], func_attr)
        else:
            raise NotImplementedError(' Unknown function type {}'.format(str(type(func))))

        return func

    def _visit_keyword(self, stmt):
        print(['HERE'], stmt)
        target = stmt.arg
        val = self._visit(stmt.value)
        return ValuedArgument(target, val)

    def _visit_For(self, stmt):

        iterator = self._visit(stmt.target)
        iterable = self._visit(stmt.iter)
        body = self._visit(stmt.body)
        expr = For(iterator, iterable, body)
        return expr

    def _visit_comprehension(self, stmt):

        iterator = self._visit(stmt.target)
        iterable = self._visit(stmt.iter)
        expr = For(iterator, iterable, [])
        return expr

    def _visit_ListComp(self, stmt):

        result = self._visit(stmt.elt)
        generators = list(self._visit(stmt.generators))

        if not isinstance(self._scope[-2],ast.Assign):
            errors.report(PYCCEL_RESTRICTION_LIST_COMPREHENSION_ASSIGN,
                          symbol = stmt,
                          severity='error')
            lhs = self.get_new_variable()
        else:
            lhs = self._visit(self._scope[-2].targets)
            if len(lhs)==1:
                lhs = lhs[0]
            else:
                raise NotImplementedError("A list comprehension cannot be unpacked")

        index = self.get_new_variable()

        args = [index]
        target = IndexedElement(lhs, *args)
        target = Assign(target, result)
        assign1 = Assign(index, LiteralInteger(0))
        assign1.set_fst(stmt)
        target.set_fst(stmt)
        generators[-1].insert2body(target)
        assign2 = Assign(index, PyccelAdd(index, LiteralInteger(1)))
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

    def _visit_GeneratorExp(self, stmt):


        result = self._visit(stmt.elt)

        generators = self._visit(stmt.generators)
        parent = self._scope[-3]
        if not isinstance(parent, ast.Call):
            raise NotImplementedError("GeneratorExp is not the argument of a function call")

        name = str(self._visit(parent.func))

        grandparent = self._scope[-4]
        if isinstance(grandparent, ast.Assign):
            if len(grandparent.targets) != 1:
                raise NotImplementedError("Cannot unpack function with generator expression argument")
            lhs = self._visit(grandparent.targets[0])
        else:
            lhs = self.get_new_variable()

        body = result
        if name == 'sum':
            body = AugAssign(lhs, '+', body)
        else:
            body = FunctionCall(name, (lhs, body))
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
                          symbol = name,
                          bounding_box=(stmt.lineno, stmt.col_offset),
                          severity='fatal')

        return expr

    def _visit_If(self, stmt):

        test = self._visit(stmt.test)
        body = self._visit(stmt.body)
        orelse = self._visit(stmt.orelse)
        if len(orelse)==1 and isinstance(orelse[0],If):
            orelse = orelse[0].blocks
            return If(IfSection(test, body), *orelse)
        elif len(orelse)==0:
            return If(IfSection(test, body))
        else:
            orelse = IfSection(LiteralTrue(), orelse)
            return If(IfSection(test, body), orelse)

    def _visit_IfExp(self, stmt):

        test1 = self._visit(stmt.test)
        first = self._visit(stmt.body)
        second = self._visit(stmt.orelse)
        expr = IfTernaryOperator(test1, first, second)
        return expr

    def _visit_While(self, stmt):

        test = self._visit(stmt.test)
        body = self._visit(stmt.body)
        return While(test, body)

    def _visit_Assert(self, stmt):
        expr = self._visit(stmt.test)
        return Assert(expr)

    def _visit_CommentMultiLine(self, stmt):

        exprs = []
        # if annotated comment
        for com in stmt.s.split('\n'):
            if com.startswith('#$'):
                env = com[2:].lstrip()
                if env.startswith('omp'):
                    exprs.append(omp_parse(stmts=com))
                elif env.startswith('acc'):
                    exprs.append(acc_parse(stmts=com))
                elif env.startswith('header'):
                    expr = hdr_parse(stmts=com)
                    if isinstance(expr, MetaVariable):

                        # a metavar will not appear in the semantic stage.
                        # but can be used to modify the ast

                        self._metavars[str(expr.name)] = str(expr.value)
                        expr = EmptyNode()
                    else:
                        expr.set_fst(stmt)

                    exprs.append(expr)
                else:
                    errors.report(PYCCEL_INVALID_HEADER,
                                  symbol = stmt,
                                  severity='error')
            else:

                txt = com[1:].lstrip()
                exprs.append(Comment(txt))

        if len(exprs) == 1:
            return exprs[0]
        else:
            return CodeBlock(exprs)

    def _visit_CommentLine(self, stmt):

        # if annotated comment

        if stmt.s.startswith('#$'):
            env = stmt.s[2:].lstrip()
            if env.startswith('omp'):
                return omp_parse(stmts=stmt.s)
            elif env.startswith('acc'):
                return acc_parse(stmts=stmt.s)
            elif env.startswith('header'):
                expr = hdr_parse(stmts=stmt.s)
                if isinstance(expr, MetaVariable):

                    # a metavar will not appear in the semantic stage.
                    # but can be used to modify the ast

                    self._metavars[str(expr.name)] = str(expr.value)
                    expr = EmptyNode()

                return expr
            else:

                errors.report(PYCCEL_INVALID_HEADER,
                              symbol = stmt,
                              severity='error')

        else:
            txt = stmt.s[1:].lstrip()
            return Comment(txt)

    def _visit_Break(self, stmt):
        return Break()

    def _visit_Continue(self, stmt):
        return Continue()

    def _visit_Lambda(self, stmt):

        expr = self._visit(stmt.body)
        args = self._visit(stmt.args)

        return Lambda(tuple(args), expr)

    def _visit_withitem(self, stmt):
        # stmt.optional_vars
        context = self._visit(stmt.context_expr)
        if stmt.optional_vars:
            return AsName(context, stmt.optional_vars)
        else:
            return context

    def _visit_With(self, stmt):
        domain = self._visit(stmt.items)
        if len(domain) == 1:
            domain = domain[0]
        body = self._visit(stmt.body)
        return With(domain, body)

    def _visit_Try(self, stmt):
        # this is a blocking error, since we don't want to convert the try body
        errors.report(PYCCEL_RESTRICTION_TRY_EXCEPT_FINALLY,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_Raise(self, stmt):
        errors.report(PYCCEL_RESTRICTION_RAISE,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_Yield(self, stmt):
        errors.report(PYCCEL_RESTRICTION_YIELD,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_Starred(self, stmt):
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


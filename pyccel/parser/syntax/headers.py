# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
"""
from os.path import join, dirname

from textx.metamodel import metamodel_from_file

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.headers   import FunctionHeader, ClassHeader, MethodHeader, VariableHeader, Template
from pyccel.ast.headers   import MetaVariable , UnionType, InterfaceHeader
from pyccel.ast.headers   import construct_macro, MacroFunction, MacroVariable
from pyccel.ast.basic     import PyccelAstNode
from pyccel.ast.core      import ValuedArgument
from pyccel.ast.variable  import DottedName
from pyccel.ast.datatypes import dtype_and_precision_registry as dtype_registry, default_precision
from pyccel.ast.literals  import LiteralString
from pyccel.ast.internals import PyccelSymbol
from pyccel.errors.errors import Errors

DEBUG = False
errors = Errors()

class Header(object):
    """Class for Header syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Header.

        """
        self.statements = kwargs.pop('statements', [])

class FuncType(BasicStmt):
    """Base class representing a  FunctionType in the grammar."""
    def __init__(self, **kwargs):
        self.decs = kwargs.pop('decs')
        self.results = kwargs.pop('results')

        super().__init__(**kwargs)

    @property
    def expr(self):
        decs = []
        if self.decs:
            decs = [i.expr for i in self.decs]

        results = []
        if self.results:
            results = [i.expr for i in self.results]

        d_var = {}
        d_var['decs'] = decs
        d_var['results'] = results
        d_var['is_func'] = True

        return d_var

class TemplateStmt(BasicStmt):
    """Base class representing a  template in the grammar."""
    def __init__(self, **kwargs):
        self.dtypes  = kwargs.pop('dtypes')
        self.name   = kwargs.pop('name')
        BasicStmt.__init__(self)

    @property
    def expr(self):
        if any(isinstance(d_type, FuncType) for d_type in self.dtypes):
            msg = 'Functions in a template are not supported yet'
            errors.report(msg,
                        severity='fatal')

        possible_dtypes = {tuple(t.expr.items())  for t in self.dtypes}
        dtypes = tuple(dict(d_type) for d_type in possible_dtypes)
        return Template(self.name, dtypes)

class ListType(BasicStmt):
    """Base class representing a  ListType in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a TypeHeader.

        dtype: list of str
        """
        self.dtype = kwargs.pop('dtype')

        super(ListType, self).__init__(**kwargs)

    @property
    def expr(self):
        dtypes = [str(i.expr['datatype']) for i in self.dtype]
        precisions = [i.expr['precision'] for i in self.dtype]
        if not (all(dtypes[0]==i for i in dtypes)):
            raise TypeError('all element of the TypeList must have the same type')

        d_var = {}
        d_var['datatype'] = str(dtypes[0])
        d_var['rank'] = len(dtypes)
        d_var['is_pointer'] = len(dtypes)>0
        d_var['allocatable'] = False
        d_var['precision'] = max(precisions)
        d_var['order'] = 'C'
        d_var['is_func'] = False
        d_var['is_const'] = False
        if not(d_var['precision']):
            if d_var['datatype'] in ['double','float','complex','int']:
                d_var['precision'] = default_precision[d_var['datatype']]
        return d_var

class Type(BasicStmt):
    """Base class representing a header type in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Type.

        dtype: str
            variable type
        """
        self.dtype   = kwargs.pop('dtype')
        self.prec    = kwargs.pop('prec')
        self.trailer = kwargs.pop('trailer', [])

        super(Type, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype = self.dtype
        precision = self.prec
        if dtype in dtype_registry.keys():
            dtype,precision = dtype_registry[dtype]
        trailer = self.trailer
        order = 'C'

        if trailer:
            if trailer.order:
                order = str(trailer.order)
            trailer = [str(i) for i in trailer.args]
        else:
            trailer = []
        d_var={}
        d_var['datatype']=dtype
        d_var['rank'] = len(trailer)
        d_var['allocatable'] = len(trailer)>0
        d_var['is_pointer'] = False
        d_var['precision']  = precision
        d_var['is_func'] = False
        d_var['is_const'] = False
        if not(precision):
            if dtype in ['double' ,'float','complex', 'int']:
                d_var['precision'] = default_precision[dtype]

        if d_var['rank']>1:
            d_var['order'] = order
        return d_var

class TypeHeader(BasicStmt):
    pass

class StringStmt(BasicStmt):
    def __init__(self, **kwargs):
        self.arg = kwargs.pop('arg')
    @property
    def expr(self):
        return LiteralString(repr(str(self.arg)))

class UnionTypeStmt(BasicStmt):
    def __init__(self, **kwargs):
        """
        Constructor for a TypeHeader.

        dtype: list fo str
        """
        self.dtypes = kwargs.pop('dtypes')
        self.const = kwargs.pop('const')

        super(UnionTypeStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        dtypes = [i.expr for i in self.dtypes]
        if self.const:
            for d_type in dtypes:
                d_type["is_const"] = True
        if len(dtypes)==1:
            return dtypes[0]
        if any(isinstance(d_type, FuncType) for d_type in self.dtypes):
            msg = 'Functions in a uniontype are not supported yet'
            errors.report(msg,
                        severity='fatal')

        possible_dtypes = {tuple(t.items())  for t in dtypes}
        dtypes = [dict(d_type) for d_type in possible_dtypes]
        return UnionType(dtypes)

class HeaderResults(BasicStmt):
    """Base class representing a HeaderResults in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a HeaderResults.

        decs: list of TypeHeader
        """
        self.decs = kwargs.pop('decs')

        super(HeaderResults, self).__init__(**kwargs)

    @property
    def expr(self):
        decs = [i.expr for i in self.decs]
        return decs


class VariableHeaderStmt(BasicStmt):
    """Base class representing a header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a VariableHeader statement.
        In the case of builtin datatypes, we export a Variable

        name: str
            variable name
        dec: list, tuple
            list of argument types
        """
        self.name = kwargs.pop('name')
        self.dec  = kwargs.pop('dec')

        super(VariableHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype = self.dec.expr

        return VariableHeader(self.name, dtype)

class FunctionHeaderStmt(BasicStmt):
    """Base class representing a function header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a FunctionHeader statement

        name: str
            function name
        kind: str
            one among {function, method}
        decs: list, tuple
            list of argument types
        results: list, tuple
            list of output types
        """
        self.name = kwargs.pop('name')
        self.kind = kwargs.pop('kind', None)
        self.static = kwargs.pop('static', None)
        self.decs = kwargs.pop('decs')
        self.results = kwargs.pop('results', None)

        super(FunctionHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        # TODO: do we need dtypes and results to be attributs of the class?
        dtypes = []
        for dec in self.decs:
            if isinstance(dec,UnionTypeStmt):
                dtypes += [dec.expr]

        if self.kind is None:
            kind = 'function'
        else:
            kind = str(self.kind)

        is_static = False
        if self.static == 'static':
            is_static = True

        results = []
        if self.results:
            results = self.results.expr

        if kind == 'method':
            dtype = dtypes[0]
            if isinstance(dtype, UnionType):
                cls_instance = dtype.args[0]['datatype']
            else:
                cls_instance = dtype['datatype']
            dtypes = dtypes[1:] # remove the attribut
            return MethodHeader((cls_instance, self.name), dtypes, [] )
        else:
            return FunctionHeader(self.name,
                                  dtypes,
                                  results=results,
                                  is_static=is_static)

class ClassHeaderStmt(BasicStmt):
    """Base class representing a class header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Header statement

        name: str
            class name
        options: list, tuple
            list of class options
        """
        self.name    = kwargs.pop('name')
        self.options = kwargs.pop('options')

        super(ClassHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        options = [str(i) for i in self.options]
        return ClassHeader(self.name, options)


class MetavarHeaderStmt(BasicStmt):
    """Base class representing a metavar header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a MetavarHeader statement.

        name: str
            metavar name
        value: str
            associated value
        """
        self.name = kwargs.pop('name')
        self.value = kwargs.pop('value')

        super(MetavarHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        # TODO further checking
        name = str(self.name)
        value = self.value
        return MetaVariable(name, value)


class InterfaceStmt(BasicStmt):
    """ class represent the header interface statement"""

    def __init__(self, **kwargs):
        """
        Constructor of Interface statement

        name: str

        args: list of function names

        """

        self.name = kwargs.pop('name')
        self.args = kwargs.pop('args')
        super(InterfaceStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        return InterfaceHeader(self.name, self.args)

# ...
class MacroArg(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """

        self.arg = kwargs.pop('arg')
        self.value = kwargs.pop('value',None)

        super(MacroArg, self).__init__(**kwargs)

    @property
    def expr(self):
        arg_ = self.arg
        if isinstance(arg_, MacroList):
            return tuple(arg_.expr)
        arg = PyccelSymbol(arg_)
        value = self.value
        if not(value is None):
            if isinstance(value, (MacroStmt,StringStmt)):
                value = value.expr
            return ValuedArgument(arg, value)
        return arg


class MacroStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """
        self.arg = kwargs.pop('arg')
        self.macro = kwargs.pop('macro')
        self.parameter = kwargs.pop('parameter', None)

        super(MacroStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        name = str(self.macro)
        arg  = PyccelSymbol(self.arg)
        parameter = self.parameter
        return construct_macro(name, arg, parameter=parameter)

# ...

class MacroList(BasicStmt):
    """ reresent a MacroList statement"""
    def __init__(self, **kwargs):
        ls = []
        for i in kwargs.pop('ls'):
            if isinstance(i, MacroArg):
                ls.append(i.expr)
            else:
                ls.append(i)
        self.ls = ls

        super(MacroList, self).__init__(**kwargs)

    @property
    def expr(self):
        return self.ls


class FunctionMacroStmt(BasicStmt):
    """Base class representing an alias function statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a FunctionMacroStmt statement

        name: str
            function name
        master: str
            master function name
        """

        self.dotted_name = tuple(kwargs.pop('dotted_name'))
        self.results = kwargs.pop('results',None)
        if self.results:
            self.results = [PyccelSymbol(r) for r in self.results]
        self.args = kwargs.pop('args')
        self.master_name = tuple(kwargs.pop('master_name'))
        self.master_args = kwargs.pop('master_args')

        super(FunctionMacroStmt, self).__init__(**kwargs)

    @property
    def expr(self):

        if len(self.dotted_name)>1:
            name = DottedName(*self.dotted_name)
        else:
            name = str(self.dotted_name[0])

        args = []
        for i in self.args:
            if isinstance(i, MacroArg):
                args.append(i.expr)
            else:
                raise TypeError('argument must be of type MacroArg')


        if len(self.master_name)==1:
            master_name = str(self.master_name[0])
        else:
            raise NotImplementedError('TODO')

        master_args = []
        for i in self.master_args:
            if isinstance(i, MacroStmt):
                master_args.append(i.expr)
            else:
                master_args.append(PyccelSymbol(i))


        results = self.results
        if (results is None):
            results = []


        if len(args + master_args + results) == 0:
            return MacroVariable(name, master_name)

        if not isinstance(name, str):
            #we treat the other all the names except the last one  as arguments
            # so that we always have a name of type str
            args = list(name.name[:-1]) + list(args)
            name = name.name[-1]
        return MacroFunction(name, args, master_name, master_args, results=results)


#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
hdr_classes = [Header, TypeHeader,
               Type, ListType, UnionTypeStmt, FuncType,
               HeaderResults,
               FunctionHeaderStmt,
               TemplateStmt,
               ClassHeaderStmt,
               VariableHeaderStmt,
               MetavarHeaderStmt,
               InterfaceStmt,
               MacroStmt,
               MacroArg,
               MacroList,
               FunctionMacroStmt,StringStmt]

this_folder = dirname(__file__)

# Get meta-model from language description
grammar = join(this_folder, '../grammar/headers.tx')

meta = metamodel_from_file(grammar, classes=hdr_classes)

def parse(filename=None, stmts=None):
    """ Parse header pragmas

      Parameters
      ----------

      filename: str

      stmts   : list

      Results
      -------

      stmts  : list

    """
    # Instantiate model
    if filename:
        model = meta.model_from_file(filename)
    elif stmts:
        model = meta.model_from_str(stmts)
    else:
        raise ValueError('Expecting a filename or a string')
    # Ensure PyccelAstNode is in correct stage
    PyccelAstNode.stage = 'syntactic'

    stmts = []
    for stmt in model.statements:
        e = stmt.stmt.expr
        stmts.append(e)

    if len(stmts) == 1:
        return stmts[0]
    else:
        return stmts

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
if __name__ == '__main__':
    print(parse(stmts='#$ header variable x :: int'))
    print(parse(stmts='#$ header variable x float [:, :]'))
    print(parse(stmts='#$ header function f(float [:], int [:]) results(int)'))
    print(parse(stmts='#$ header function f(float|int, int [:]) results(int)'))
    print(parse(stmts='#$ header class Square(public)'))
    print(parse(stmts='#$ header method translate(Point, [double], [int], int[:,:], double[:])'))
    print(parse(stmts="#$ header metavar module_name='mpi'"))
    print(parse(stmts='#$ header interface funcs=fun1|fun2|fun3'))
    print(parse(stmts='#$ header function _f(int, int [:])'))
    print(parse(stmts='#$ header macro _f(x) := f(x, x.shape)'))
    print(parse(stmts='#$ header macro _g(x) := g(x, x.shape[0], x.shape[1])'))
    print(parse(stmts='#$ header macro (a, b), _f(x) := f(x.shape, x, a, b)'))
    print(parse(stmts='#$ header macro _dswap(x, incx) := dswap(x.shape, x, incx)'))
    print(parse(stmts="#$ header macro _dswap(x, incx=1) := dswap(x.shape, x, incx)"))
    print(parse(stmts='#$ header macro _dswap(x, y, incx=1, incy=1) := dswap(x.shape, x, incx, y, incy)'))
    print(parse(stmts="#$ header macro _dswap(x, incx=x.shape) := dswap(x.shape, x, incx)"))
    print(parse(stmts='#$ header macro Point.translate(alpha, x, y) := translate(alpha, x, y)'))
    print(parse(stmts="#$ header macro _dswap([data,dtype=data.dtype,count=count.dtype], incx=y.shape,M='M',d=incx) := dswap(y.shape, y, incx)"))
    print(parse(stmts='#$ header function _f(int, int [:,:](order = F))'))
    print(parse(stmts='#$ header function _f(int, int [:,:])'))

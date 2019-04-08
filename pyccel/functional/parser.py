# coding: utf-8

import os
from os.path import join, dirname

from sympy import Symbol, Lambda, Function, Dummy, Tuple
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction
from sympy import Integer, Float
from sympy import sympify
from sympy import FunctionClass

from textx.metamodel import metamodel_from_str


from pyccel.codegen.utilities import random_string
from pyccel.ast.utilities import build_types_decorator
from pyccel.ast.core import Variable, FunctionDef, Assign, AugAssign
from pyccel.ast.datatypes import Int, Real, Complex, Bool
from .ast import Reduce
from .ast import SeqMap, ParMap, BasicMap
from .ast import SeqTensorMap, ParTensorMap, BasicTensorMap
from .ast import SeqZip, SeqProduct
from .ast import ParZip, ParProduct
from .ast import assign_type, BasicTypeVariable
from .ast import TypeVariable, TypeTuple, TypeList
from .ast import VariableGenerator
from .ast import generator_as_block

_known_functions = {'map':      SeqMap,
                    'pmap':     ParMap,
                    'tmap':     SeqTensorMap,
                    'ptmap':    ParTensorMap,
                    'zip':      SeqZip,
                    'pzip':     ParZip,
                    'product':  SeqProduct,
                    'pproduct': ParProduct,
                    'reduce':   Reduce,
                   }

_functors_map_registery = ['map', 'pmap', 'tmap', 'ptmap']
_functors_registery = _functors_map_registery + ['reduce']

# TODO atan2, pow
_elemental_math_functions = ['acos',
                             'asin',
                             'atan',
                             'cos',
                             'cosh',
                             'exp',
                             'log',
                             'log10',
                             'sin',
                             'sinh',
                             'sqrt',
                             'tan',
                             'tanh',
                            ]

# TODO add cross, etc
_math_vector_functions = ['dot']

# TODO
_math_matrix_functions = ['matmul']

_math_functions = _elemental_math_functions + _math_vector_functions + _math_matrix_functions
#==============================================================================

#==============================================================================
# ...
def _attributs_from_type(t, d_var):
    if isinstance(t, TypeList):
        t = _attributs_from_type(t.parent, d_var)
        d_var['rank'] = d_var['rank'] + 1
        return t, d_var

    elif isinstance(t, TypeTuple):
        raise NotImplementedError()

    elif isinstance(t, TypeVariable):
        d_var['dtype']          = t.dtype
        d_var['rank']           = t.rank
        d_var['is_stack_array'] = t.is_stack_array
        d_var['order']          = t.order
        d_var['precision']      = t.precision

        return t, d_var
# ...

# ... default values
def _attributs_default():
    d_var = {}

    d_var['dtype']          = None
    d_var['rank']           = 0
    d_var['allocatable']    = False
    d_var['is_stack_array'] = False
    d_var['is_pointer']     = False
    d_var['is_target']      = False
    d_var['shape']          = None
    d_var['order']          = 'C'
    d_var['precision']      = None

    return d_var
# ...
#==============================================================================

#==============================================================================
# any argument
class AnyArgument(Symbol):
    pass

_ = AnyArgument('_')

#==============================================================================
class NamedAbstraction(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.abstraction = kwargs.pop('abstraction')

class Abstraction(object):
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')
        self.expr = kwargs.pop('expr')

class Application(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.args = kwargs.pop('args')

#==============================================================================
def to_sympy(stmt):

    if isinstance(stmt, NamedAbstraction):
        name = stmt.name
        expr = to_sympy(stmt.abstraction)
        return expr

    elif isinstance(stmt, Abstraction):
        args = [to_sympy(i) for i in stmt.args]
        expr = to_sympy(stmt.expr)

        return Lambda(args, expr)

    elif isinstance(stmt, Application):
        args = [to_sympy(i) for i in stmt.args]
        name = stmt.name

        return Function(name)(*args)

    elif isinstance(stmt, (int, float)):
        return stmt

    elif isinstance(stmt, str):
        if stmt == '_':
            return _

        else:
            return sympify(stmt)

    else:
        raise TypeError('Not implemented for {}'.format(type(stmt)))

#==============================================================================
def sanitize(expr):
    if isinstance(expr, Lambda):
        args = expr.variables
        expr = sanitize(expr.expr)

        return Lambda(args, expr)

    elif isinstance(expr, AppliedUndef):
        name = expr.__class__.__name__

        args = [sanitize(i) for i in expr.args]
        # first argument of Map & Reduce are functions
        if name in _functors_registery:
            first = args[0]
            if isinstance(first, Symbol):
                args[0] = Function(first.name)

        if name in _known_functions.keys():
            return _known_functions[name](*args)

        else:
            return Function(name)(*args)

    elif isinstance(expr, (int, float, Integer, Float, Symbol)):
        return expr

    else:
        raise TypeError('Not implemented for {}'.format(type(expr)))

#==============================================================================
def parse(inputs, debug=False, verbose=False):
    this_folder = dirname(__file__)

    classes = [NamedAbstraction, Abstraction, Application]

    # Get meta-model from language description
    grammar = join(this_folder, 'grammar.tx')

    from textx.metamodel import metamodel_from_file
    meta = metamodel_from_file(grammar, debug=debug, classes=classes)

    # Instantiate model
    if os.path.isfile(inputs):
        ast = meta.model_from_file(inputs)

    else:
        ast = meta.model_from_str(inputs)

    # ...
    expr = to_sympy(ast)
    if verbose:
        print('>>> stage 0 = ', expr)
    # ...

#    # ...
#    expr = sanitize(expr)
#    if verbose:
#        print('>>> stage 1 = ', expr)
#    # ...

    # ...
    if verbose:
        print('')
    # ...

    return expr

#==============================================================================
def _get_key(expr):
    # TODO to be replaced by domain
    if isinstance(expr, FunctionDef):
        return str(expr.name) + '_args'

    elif isinstance(expr, UndefinedFunction):
        return str(expr)

    elif isinstance(expr, Symbol):
        return expr.name

    else:
        raise NotImplementedError('for {}'.format(type(expr)))

#==============================================================================
# TODO add some verifications before starting annotating L
class SemanticParser(object):

    def __init__(self, expr, **kwargs):
        assert(isinstance(expr, Lambda))

        self._expr = expr

        # ...
        self._namespace = {}
        self._d_types   = {}
        self._d_expr    = {}
        self._tag       = random_string( 8 )

        # to store current typed expr
        # this must not be a private variable,
        # in order to modify it on the fly
        self.main = expr
        self.main_type = None
        # ...

        # ... add types for arguments and results
        #     TODO use domain and codomain optional args for functions
        self._typed_functions = kwargs.pop('typed_functions', {})
        for f in self.typed_functions.values():
            type_domain   = assign_type(f.arguments)
            type_codomain = assign_type(f.results)

            self._set_type(f, value=type_domain, domain=True)
            self._set_type(f, value=type_codomain, codomain=True)
        # ...

        # ... default Type
        prefix = kwargs.pop('prefix', 'd') # doubles as default
        dtype     = None
        precision = None
        if prefix == 'i':
            dtype     = Int
            precision = 4

        elif prefix == 's':
            dtype     = Real
            precision = 4

        elif prefix == 'd':
            dtype     = Real
            precision = 8

        elif prefix == 'c':
            dtype     = Complex
            precision = 8

        elif prefix == 'z':
            dtype     = Complex
            precision = 16

        else:
            raise ValueError('Wrong prefix. Available: i, s, d, c, z')

        var = Variable(dtype, 'dummy_' + self.tag, precision=precision)
        self._default_type = TypeVariable(var)
        # ...

        # ... get all functions
        calls = list(expr.atoms(AppliedUndef))
        map_funcs = [i.args[0] for i in calls if i.__class__.__name__ in _functors_map_registery]
        callables = [i.func for i in calls  if not i.__class__.__name__ in _functors_registery]
        functions = list(set(map_funcs + callables))

        for f in functions:
            if str(f) in _elemental_math_functions:
                type_domain   = self.default_type
                type_codomain = self.default_type

                self._set_type(f, value=type_domain, domain=True)
                self._set_type(f, value=type_codomain, codomain=True)

            elif not str(f) in list(_known_functions.keys()) + list(self.typed_functions.keys()):
                raise NotImplementedError('{} not available'.format(str(f)))
        # ...

    @property
    def expr(self):
        return self._expr

    @property
    def typed_functions(self):
        return self._typed_functions

    @property
    def default_type(self):
        return self._default_type

    @property
    def namespace(self):
        return self._namespace

    @property
    def d_types(self):
        return self._d_types

    @property
    def d_expr(self):
        return self._d_expr

    @property
    def tag(self):
        return self._tag

    def inspect(self):
        print('============ types =============')
        print(self.d_types)
        for k,v in self.d_types.items():
            print('  {k} = {v}'.format(k=k, v=v.view()))
        print('================================')

    def _get_label(self, target, domain=False, codomain=False):
        # TODO improve
        if codomain:
            assert(not domain)
            if (isinstance(target, FunctionClass)):
                name = str(target)

            else:
                name = str(target.name)

            return name

        if domain:
            assert(not codomain)
            if (isinstance(target, FunctionClass)):
                name = str(target)

            else:
                name = str(target.name)

            _avail_funcs = list(self.typed_functions.keys()) + _math_functions
            if name in _avail_funcs:
                return name + '_args'

        return _get_key(target)

    def _get_type(self, target, domain=False, codomain=False):
        label = self._get_label(target, domain=domain, codomain=codomain)

        if label in self.d_types.keys():
            return self.d_types[label]

        return None

    def _set_type(self, target, value, domain=False, codomain=False):
        label = self._get_label(target, domain=domain, codomain=codomain)

        self.d_types[label] = value
        self._set_expr(value, target)

    def _set_expr(self, t_var, expr):
        self._d_expr[t_var.name] = expr

    def build_namespace(self):
        """builds the namespace from types."""
        raise NotImplementedError('')

    def to_type(self, verbose=False):

        # ... compute type
        i_count = 0
        max_count = 2
        while(i_count < max_count and not isinstance(self.main, BasicTypeVariable)):
            if verbose:
                print('----> BEFORE ', self.main)

            self.main = self._to_type(self.main)

            if verbose:
                print('<---- AFTER', self.main)

            i_count += 1
        # ...

        return self.main

    def _to_type(self, stmt, value=None):

        cls = type(stmt)
        name = cls.__name__

        method = '_to_type_{}'.format(name)
        if hasattr(self, method):
            return getattr(self, method)(stmt, value=value)

        elif name in _known_functions.keys():
#            print('[{}]'.format(name))

            FUNCTION = 'function'
            FUNCTOR  = 'functor'

            if name in ['map', 'pmap', 'tmap', 'ptmap']:
                name = 'map'
                kind = FUNCTOR

            elif name in ['reduce']:
                name = 'reduce'
                kind = FUNCTOR

            elif name in ['zip', 'pzip']:
                name = 'zip'
                kind = FUNCTION

            elif name in ['product', 'pproduct']:
                name = 'product'
                kind = FUNCTION

            else:
                raise NotImplementedError('{}'.format(name))

            pattern = '_to_type_{kind}_{name}'
            method  = pattern.format(kind=kind, name=name)
            method = getattr(self, method)

            return method(stmt, value=value)

        # Unknown object, we raise an error.
        raise TypeError('{node} not yet available'.format(node=type(stmt)))

    def _to_type_Lambda(self, stmt, value=None):
        # TODO treat args
        self.main = self._to_type(stmt.expr)
        if isinstance(self.main, BasicTypeVariable):
            self.main_type = self.main

        return self.main

    def _to_type_TypeVariable(self, stmt, value=None):
        return stmt

    def _to_type_TypeTuple(self, stmt, value=None):
        return stmt

    def _to_type_TypeList(self, stmt, value=None):
        return stmt

    def _to_type_Symbol(self, stmt, value=None):
        assert(not( value is None ))
        self._set_type(stmt, value)

    def _to_type_functor_map(self, stmt, value=None):
        arguments = stmt.args

        assert( len(arguments) == 2 )
        func   = arguments[0]
        target = arguments[1]

        type_codomain = self._get_type(func, codomain=True)
        type_domain   = self._get_type(func, domain=True)

        if not type_codomain:
            print('> Unable to compute type for {} '.format(stmt))
            raise NotImplementedError('')

        # TODO improve
        if stmt.__class__.__name__ in ['tmap', 'ptmap']:
            # TODO check that rank is the same for all domain
            assert(isinstance(target, AppliedUndef))
            assert(target.__class__.__name__ in ['product', 'pproduct'])

            # we substruct 1 since we use a TypeList
            base_rank = len(target.args) - 1

            type_domain   = assign_type(type_domain, rank=base_rank)
            type_codomain = assign_type(type_codomain, rank=base_rank)

        type_domain   = TypeList(type_domain)
        type_codomain = TypeList(type_codomain)

        self._to_type(target, value=type_domain)
        self._set_expr(type_codomain, stmt)

        return type_codomain

    def _to_type_function_zip(self, stmt, value=None):
        arguments = stmt.args

        assert(not( value is None ))
        assert(isinstance(value, TypeList))
        assert(len(value.parent.types) == len(arguments))

        if not isinstance(value.parent, TypeTuple):
            msg = '{} not available yet'.format(type(value.parent))
            raise NotImplementedError(msg)

        values = value.parent.types

        for a,t in zip(arguments, values):
            type_domain  = TypeList(t)
            self._to_type(a, value=type_domain)

        type_codomain = value

        # update main expression
        self.main = self.main.xreplace({stmt: type_codomain})
        self._set_expr(type_codomain, stmt)

        return type_codomain

    def _to_type_function_product(self, stmt, value=None):
        arguments = stmt.args

        assert(not( value is None ))
        assert(isinstance(value, TypeList))
        assert(len(value.parent.types) == len(arguments))

        if not isinstance(value.parent, TypeTuple):
            msg = '{} not available yet'.format(type(value.parent))
            raise NotImplementedError(msg)

        values = value.parent.types

        for a,t in zip(arguments, values):
            type_domain  = TypeList(t)
            self._to_type(a, value=type_domain)

        type_codomain = value

        # update main expression
        self.main = self.main.xreplace({stmt: type_codomain})
        self._set_expr(type_codomain, stmt)

        return type_codomain

    def _to_type_functor_reduce(self, stmt, value=None):
        arguments = stmt.args

        assert( len(arguments) == 2 )
        op     = arguments[0]
        target = arguments[1]

        # we must first determine the number of arguments
        # TODO must be done in main lambdify:
        #      - we use atoms on AppliedUndef
        #      - then we take those for which we provide python implementations
        #      - then we subtitute the function call by the appropriate one
        #      - and we append its implementation to user_functions
        nargs = len(sanitize(target))
        precision = str(op)[0]
        # TODO check this as in BLAS
        assert( precision in ['i', 's', 'd', 'z', 'c'] )
        print(nargs)

        print('> ', op, type(op))

        import sys; sys.exit(0)

    def annotate(self, stmt=None):

        if stmt is None:
            stmt = self.expr

        cls = type(stmt)
        name = cls.__name__

        method = '_annotate_{}'.format(name)
        if hasattr(self, method):
            return getattr(self, method)(stmt)

        elif name in _known_functions.keys():
            if name == 'map':
                func, target = stmt.args

                # ... construct the generator
                generator = self.annotate(target)
                if isinstance(generator, Variable):
                    generator = VariableGenerator(generator)
                # ...

                # ... construct the results
                type_codomain = self.main_type
                results = self.annotate(type_codomain)
                # ...

                # ... apply the function to arguments
                args = generator.iterator

                if isinstance(args, Tuple):
                    call = func( *args )

                else:
                    call = func( args )
                # ...

                # ... create core statement
                stmts = [Assign(results, call)]
                # ...

                # TODO USE THIS
#                expr = self.get_expr_from_type()

                # return the associated for loops
                return generator_as_block(generator, stmts, parallel=False)

            else:
                raise NotImplementedError('')

        # Unknown object, we raise an error.
        raise TypeError('{node} not yet available'.format(node=type(stmt)))

    def _annotate_Lambda(self, stmt):
        args = [self.annotate(i) for i in stmt.variables]
        expr = self.annotate(stmt.expr)
        # TODO improve
        results = self.annotate(self.main)

        # TODO to be moved to a method .to_funcdef
#        # ...
#        decorators = {'types':         build_types_decorator(args),
#                      'external_call': []}
#
#        tag         = random_string( 6 )
#        g_name      = 'lambda_{}'.format( tag )
#        # ...
#
#        return FunctionDef(g_name, args, results, body,
#                           decorators=decorators)

        return expr

    def _annotate_Symbol(self, stmt):
        t_var = self.d_types[stmt.name]
        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, stmt.name, **d_var )

        return var

    def _annotate_Integer(self, stmt):
        return stmt

    def _annotate_Float(self, stmt):
        return stmt

    def _annotate_TypeVariable(self, stmt):
        name  = 'dummy_{}'.format(stmt.tag)
        t_var = stmt

        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, name, **d_var )

        return var

    def _annotate_TypeTuple(self, stmt):
        # TODO
        name  = 'dummy_{}'.format(stmt.tag)
        t_var = stmt

        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, name, **d_var )

        return var

    def _annotate_TypeList(self, stmt):
        # TODO
        name  = 'dummy_{}'.format(stmt.tag)
        t_var = stmt

        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, name, **d_var )

        return var

    def get_expr_from_type(self, t_var=None):
        if t_var is None:
            t_var = self.main_type

        return self.d_expr[t_var.name]

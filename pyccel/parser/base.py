# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module containing aspects of a parser which are in common over all stages.
"""

from collections import OrderedDict
import importlib
import os
import re

#==============================================================================
from pyccel.version import __version__

from pyccel.ast.builtins import Lambda

from pyccel.ast.core import SymbolicAssign
from pyccel.ast.core import FunctionDef, Interface, FunctionAddress
from pyccel.ast.core import PythonFunction, SympyFunction
from pyccel.ast.core import Import, AsName
from pyccel.ast.core import create_incremented_string, create_variable

from pyccel.ast.utilities import builtin_import_registery as pyccel_builtin_import_registery

from pyccel.parser.utilities import is_valid_filename_pyh, is_valid_filename_py

from pyccel.errors.errors import Errors

# TODO - remove import * and only import what we need
#      - use OrderedDict whenever it is possible

from pyccel.errors.messages import *

#==============================================================================

errors = Errors()
#==============================================================================

strip_ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]|[\n\t\r]')

# use this to delete ansi_escape characters from a string
# Useful for very coarse version differentiation.

#==============================================================================


def get_filename_from_import(module,input_folder=''):
    """Returns a valid filename with absolute path, that corresponds to the
    definition of module.
    The priority order is:
        - header files (extension == pyh)
        - python files (extension == py)
    """

    if (isinstance(module, AsName)):
        module = str(module.name)

    # Remove first '.' as it doesn't represent a folder change
    if module[0] == '.':
        module = module[1:]
    filename = module.replace('.','/')

    # relative imports
    folder_above = '../'
    while filename.startswith('/'):
        filename = folder_above + filename[1:]

    filename_pyh = '{}.pyh'.format(filename)
    filename_py  = '{}.py'.format(filename)

    if is_valid_filename_pyh(filename_pyh):
        return os.path.abspath(filename_pyh)
    if is_valid_filename_py(filename_py):
        return os.path.abspath(filename_py)
    folders = input_folder.split(""".""")
    for i in range(len(folders)):
        poss_dirname      = os.path.join( *folders[:i+1] )
        poss_filename_pyh = os.path.join( poss_dirname, filename_pyh )
        poss_filename_py  = os.path.join( poss_dirname, filename_py  )
        if is_valid_filename_pyh(poss_filename_pyh):
            return os.path.abspath(poss_filename_pyh)
        if is_valid_filename_py(poss_filename_py):
            return os.path.abspath(poss_filename_py)

    source = module
    if len(module.split(""".""")) > 1:

        # we remove the last entry, since it can be a pyh file

        source = """.""".join(i for i in module.split(""".""")[:-1])
        _module = module.split(""".""")[-1]
        filename_pyh = '{}.pyh'.format(_module)
        filename_py = '{}.py'.format(_module)

    try:
        package = importlib.import_module(source)
        package_dir = str(package.__path__[0])
    except ImportError:
        errors = Errors()
        errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE, symbol=source,
                      severity='fatal')

    filename_pyh = os.path.join(package_dir, filename_pyh)
    filename_py = os.path.join(package_dir, filename_py)
    if os.path.isfile(filename_pyh):
        return filename_pyh
    elif os.path.isfile(filename_py):
        return filename_py

    errors = Errors()
    errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE, symbol=module,
                  severity='fatal')


#==============================================================================

class Scope(object):
    """."""

    def __init__(self, *, headers=None, decorators=None, templates=None):

        self._imports = OrderedDict()

        self._imports['functions'] = OrderedDict()
        self._imports['variables'] = OrderedDict()
        self._imports['classes'  ] = OrderedDict()
        self._imports['imports'  ] = OrderedDict()

        self._imports['python_functions'  ] = OrderedDict()
        self._imports['symbolic_functions'] = OrderedDict()

        self._variables = OrderedDict()
        self._classes   = OrderedDict()
        self._functions = OrderedDict()
        self._macros    = OrderedDict()
        self._templates = templates or OrderedDict()
        self._headers   = headers    or OrderedDict()
        self._decorators= decorators or OrderedDict()

        # TODO use another name for headers
        #      => reserved keyword, or use __
        self.parent_scope        = None
        self._sons_scopes        = OrderedDict()
        self._static_functions   = []
        self._cls_constructs     = OrderedDict()
        self._symbolic_functions = OrderedDict()
        self._python_functions   = OrderedDict()

        self._is_loop = False
        # scoping for loops
        self._loops = []

    def new_child_scope(self, name, **kwargs):
        """
        Create a new child Scope object which has the current object as parent.

        The parent scope can access the child scope through the '_sons_scopes'
        dictionary, using the provided name as key. Conversely, the child scope
        can access the parent scope through the 'parent_scope' attribute.

        Parameters
        ----------
        name : str
            Name of the new scope, used as a key to retrieve the new scope.

        kwargs : dict
            Keyword arguments passed to __init__() for object initialization.

        Returns
        -------
        child : Scope
            New child scope, which has the current object as parent.

        """

        child = Scope(**kwargs)

        self._sons_scopes[name] = child
        child.parent_scope = self

        return child

    @property
    def imports(self):
        return self._imports

    @property
    def variables(self):
        return self._variables

    @property
    def classes(self):
        return self._classes

    @property
    def functions(self):
        return self._functions

    @property
    def macros(self):
        return self._macros

    @property
    def headers(self):
        return self._headers

    @property
    def templates(self):
        """A dictionary of user defined templates applied to all the functions in this scope"""
        return self._templates

    @property
    def decorators(self):
        """Dictionary of Pyccel decorators applied to a function definition."""
        return self._decorators

    @property
    def static_functions(self):
        return self._static_functions

    @property
    def cls_constructs(self):
        return self._cls_constructs

    @property
    def sons_scopes(self):
        return self._sons_scopes

    @property
    def symbolic_functions(self):
        return self._symbolic_functions

    @property
    def python_functions(self):
        return self._python_functions

    @property
    def is_loop(self):
        return self._is_loop

    @property
    def loops(self):
        return self._loops




#==============================================================================

class BasicParser(object):

    """ Class for a base Parser.
    This class contains functions and properties which are common to SyntacticParser and SemanticParser

    Parameters
    ----------

        debug: bool
            True if in debug mode.

        headers: list, tuple
            list of headers to append to the namespace

        static: list/tuple
            a list of 'static' functions as strings

        show_traceback: bool
            prints Traceback exception if True

    """

    def __init__(self,
                 debug=False,
                 headers=None,
                 static=None,
                 show_traceback=False):

        self._code = None
        self._fst  = None
        self._ast  = None

        self._filename  = None
        self._metavars  = OrderedDict()
        self._namespace = Scope()

        self._used_names = None

        # represent the namespace of a function

        self._current_class    = None
        self._current_function = None

        # the following flags give us a status on the parsing stage
        self._syntax_done   = False
        self._semantic_done = False

        # the next expected Dummy variable
        self._dummy_counter = 1

        # current position for errors

        self._current_fst_node = None

        # flag for blocking errors. if True, an error with this flag will cause
        # Pyccel to stop
        # TODO ERROR must be passed to the Parser __init__ as argument

        self._blocking = False

        # printing exception

        self._show_traceback = show_traceback

        if headers:
            if not isinstance(headers, dict):
                raise TypeError('Expecting a dict of headers')


            self.namespace.headers.update(headers)


        if static:
            if not isinstance(static, (list, tuple)):
                raise TypeError('Expecting a list/tuple of static')

            for i in static:
                if not isinstance(i, str):
                    raise TypeError('Expecting str. given {}'.format(type(i)))

            self._namespace.static_functions.extend(static)

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, namespace):
        assert isinstance(namespace, Scope)
        self._namespace = namespace

    @property
    def filename(self):
        return self._filename

    @property
    def code(self):
        return self._code

    @property
    def fst(self):
        return self._fst

    @property
    def ast(self):
        if self._ast is None:
            self._ast = self.parse()
        return self._ast

    @ast.setter
    def ast(self, ast):
        self._ast = ast

    @property
    def metavars(self):
        return self._metavars

    @property
    def current_class(self):
        return self._current_class

    @property
    def current_function(self):
        """Name of current function, if any."""
        return self._current_function

    @property
    def syntax_done(self):
        return self._syntax_done

    @property
    def semantic_done(self):
        return self._semantic_done

    @property
    def is_header_file(self):
        """Returns True if we are treating a header file."""

        if self.filename:
            return self.filename.split(""".""")[-1] == 'pyh'
        else:
            return False

    @property
    def current_fst_node(self):
        return self._current_fst_node

    @property
    def blocking(self):
        return self._blocking

    @property
    def show_traceback(self):
        return self._show_traceback

    @property
    def used_names(self):
        """Returns a set of all names used in the current file.
        The set is used to prevent name collisions when creating new variables
        """
        return self._used_names

    def get_new_name(self, current_name = None):
        """
        Creates a new name. A current_name can be provided indicating the name the
        user would like to use if possible. If this name is not available then it
        will be used as a prefix for the new name.
        If no current_name is provided, then the standard prefix is used, and the
        dummy counter is used and updated to facilitate finding the next value of
        this common case

          Parameters
          ----------
          current_name : str

          Returns
          -------
          new_name     : str
        """
        if current_name is not None and current_name not in self.used_names:
            self.used_names.add(current_name)
            return current_name

        if current_name is not None:
            new_name, self._dummy_counter = create_incremented_string(self.used_names, prefix = current_name, counter = self._dummy_counter)
        else:
            new_name,_ = create_incremented_string(self.used_names, prefix = current_name)
        return new_name

    def get_new_variable(self, prefix = None):
        """
        Creates a new PyccelSymbol using the prefix provided. If this prefix is None,
        then the standard prefix is used, and the dummy counter is used and updated
        to facilitate finding the next value of this common case

          Parameters
          ----------
          prefix   : str

          Returns
          -------
          variable : PyccelSymbol
        """
        if prefix is not None:
            var,_ = create_variable(self._used_names, prefix)
        else:
            var, self._dummy_counter = create_variable(self._used_names, prefix, counter = self._dummy_counter)
        return var

    # TODO shall we need to export the Parser too?


    def insert_function(self, func):
        """."""

        if isinstance(func, SympyFunction):
            self.insert_symbolic_function(func)
        elif isinstance(func, PythonFunction):
            self.insert_python_function(func)
        elif isinstance(func, (FunctionDef, Interface, FunctionAddress)):
            container = self.namespace.functions
            container[func.name] = func
        else:
            raise TypeError('Expected a Function definition')

    def insert_symbolic_function(self, func):
        """."""

        container = self.namespace.symbolic_functions
        if isinstance(func, SympyFunction):
            container[func.name] = func
        elif isinstance(func, SymbolicAssign) and isinstance(func.rhs,
                Lambda):
            container[func.lhs] = func.rhs
        else:
            raise TypeError('Expected a symbolic_function')

    def insert_python_function(self, func):
        """."""

        container = self.namespace.python_functions

        if isinstance(func, PythonFunction):
            container[func.name] = func
        else:
            raise TypeError('Expected a python_function')

    def insert_import(self, expr):
        """."""

        # this method is only used in the syntatic stage

        if not isinstance(expr, Import):
            raise TypeError('Expecting Import expression')
        container = self.namespace.imports['imports']

        # if source is not specified, imported things are treated as sources
        if len(expr.target) == 0:
            if isinstance(expr.source, AsName):
                name   = expr.source
                source = expr.source.name
            else:
                name   = str(expr.source)
                source = name

            if not source in pyccel_builtin_import_registery:
                container[name] = []
        else:
            source = str(expr.source)
            if source not in pyccel_builtin_import_registery:
                if not source in container.keys():
                    container[source] = []
                container[source] += expr.target

    def dump(self, filename=None):
        """
        Dump the current ast using Pickle.

          Parameters
          ----------
          filename: str
            output file name. if not given `name.pyccel` will be used and placed
            in the Pyccel directory ($HOME/.pyccel)
        """
        import pickle
        import hashlib

        if not filename:
            if not self.filename:
                raise ValueError('Expecting a filename to load the ast')

            path , name  = os.path.split(self.filename)

            if not name.split('.')[-1] == 'pyh':
                return
            else:
                name = name[:-4]

            name     = '{}.pyccel'.format(name)
            filename = os.path.join(path, name)
        # check extension

        if not filename.split(""".""")[-1] == 'pyccel':
            raise ValueError('Expecting a .pyccel extension')

#        print('>>> home = ', os.environ['HOME'])
        # ...

        # we are only exporting the AST.
        try:
            code = self.code.encode('utf-8')
            hs   = hashlib.md5(code)
            with open(filename, 'wb') as f:
                pickle.dump((hs.hexdigest(), __version__, self), f, pickle.HIGHEST_PROTOCOL)
        except (FileNotFoundError, PermissionError, pickle.PickleError):
            pass

    def load(self, filename=None):
        """ Load the current ast using Pickle.

          Parameters
          ----------
          filename: str
            output file name. if not given `name.pyccel` will be used and placed
            in the Pyccel directory ($HOME/.pyccel)
        """

        # ...
        import pickle

        if not filename:
            if not self.filename:
                raise ValueError('Expecting a filename to load the ast')

            path , name = os.path.split(self.filename)

            if not name.split('.')[-1] == 'pyh':
                return
            else:
                name = name[:-4]

            name     = '{}.pyccel'.format(name)
            filename = os.path.join(path, name)

        if not filename.split(""".""")[-1] == 'pyccel':
            raise ValueError('Expecting a .pyccel extension')

        try:
            with open(filename, 'rb') as f:
                hs, version, parser = pickle.load(f)
        except (FileNotFoundError, PermissionError, pickle.PickleError):
            return

        import hashlib
        code = self.code.encode('utf-8')
        if hashlib.md5(code).hexdigest() == hs and __version__ == version:
            self.copy(parser)

    def copy(self, parser):
        """
        Copy the parser attributes in self

          Parameters
          ----------
          parser : BasicParser

        """
        self._fst = parser.fst
        self._ast = parser.ast

        self._metavars  = parser.metavars
        self._namespace = parser.namespace

        self._used_names = parser.used_names

        # the following flags give us a status on the parsing stage
        self._syntax_done   = parser.syntax_done
        self._semantic_done = parser.semantic_done

#==============================================================================


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        raise ValueError('Expecting an argument for filename')

    parser = BasicParser(filename)

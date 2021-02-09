#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains classes and methods that manipilate the various errors and warnings
that could be shown by pyccel.
"""

from collections import OrderedDict
from os.path import basename
from ast import dump as ast_dump
from pyccel.ast.basic import Basic

# ...
#ERROR = 'error'
#INTERNAL = 'internal'
#WARNING = 'warning'
#FATAL = 'fatal'
#
#PYCCEL = 'pyccel'
#
#def make_symbol(s):
#    return str(s)

try:
    from termcolor import colored
    ERROR = colored('error', 'red', attrs=['blink', 'bold'])
    INTERNAL = colored('internal', attrs=['blink', 'bold'])
    WARNING = colored('warning', 'green', attrs=['blink'])
    FATAL = colored('fatal', 'red', attrs=['blink', 'bold'])

    PYCCEL = colored('pyccel', attrs=['bold'])

    def make_symbol(s):
        return colored(str(s), attrs=['bold'])
except ImportError:
    ERROR = 'error'
    INTERNAL = 'internal'
    WARNING = 'warning'
    FATAL = 'fatal'

    PYCCEL = 'pyccel'

    def make_symbol(s):
        return str(s)
# ...

_severity_registry = {'error': ERROR,
                      'internal': INTERNAL,
                      'fatal': FATAL,
                      'warning': WARNING}


class PyccelError(Exception):
    def __init__(self, message, errors=''):

        # Call the base class constructor with the parameters it needs
        super(PyccelError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors

class PyccelSyntaxError(PyccelError):
    pass

class PyccelSemanticError(PyccelError):
    pass

class PyccelCodegenError(PyccelError):
    pass


class ErrorInfo:
    """Representation of a single error message."""

    def __init__(self, *, stage, filename,
                 line=None,
                 column=None,
                 severity=None,
                 message='',
                 symbol=None,
                 blocker=False):
        # The parser stage
        self.stage = stage
        # The source file that was the source of this error.
        self.filename = basename(filename) if filename is not None else ''
        # The line number related to this error within file.
        self.line = line
        # The column number related to this error with file.
        if isinstance(column, (tuple, list)):
            column = '-'.join(str(i) for i in column)
        self.column = column
        # Either 'error', 'fatal', or 'warning'.
        self.severity = severity
        # The error message.
        self.message = message
        # Symbol associated to the message
        self.symbol = symbol
        # If True, we should halt build after the file that generated this error.
        self.blocker = blocker or (severity == 'fatal')

    def __str__(self):

        pattern = '|{severity} [{stage}]: {filename}{location}| {message}{symbol}'
        info = {
            'stage'   : self.stage,
            'severity': _severity_registry[self.severity],
            'filename': self.filename,
            'location': '',
            'message' : self.message,
            'symbol'  : ''
        }

        if self.line:
            if self.column:
                info['location'] = ' [{line},{column}]'.format(line=self.line, column=self.column)
            else:
                info['location'] = ' [{line}]'.format(line=self.line)

        if self.symbol:
            info['symbol'] = ' ({})'.format(self.symbol)

        return pattern.format(**info)


def _singleton(cls):
    """
    A Class representing a singleton. Python does not offer this pattern.
    """
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls() # Line 5
        return instances[cls]
    return getinstance


@_singleton
class ErrorsMode:
    """Developper or User mode.
    pyccel command line will set it.
    """
    def __init__(self):
        self._mode = 'user'

    @property
    def value(self):
        return self._mode

    def set_mode(self, mode):
        assert(mode in ['user', 'developer'])
        self._mode = mode


@_singleton
class Errors:
    """Container for compile errors.
    """

    def __init__(self):
        self.error_info_map = None
        self._target = None
        self._parser_stage = None
        self._mode = ErrorsMode().value

        self.initialize()

    @property
    def parser_stage(self):
        return self._parser_stage

    @property
    def target(self):
        return self._target

    @property
    def mode(self):
        return self._mode

    def initialize(self):
        self.error_info_map = OrderedDict()

        self._target = {}
        self._target['file'] = None
        self._target['module'] = None
        self._target['function'] = None
        self._target['class'] = None

    def reset(self):
        self.initialize()

    def set_parser_stage(self, stage):
        assert(stage in ['syntax', 'semantic', 'codegen'])
        self._parser_stage = stage

    def set_target(self, target, kind):
        assert(kind in ['file', 'module', 'function', 'class'])
        self._target[kind] = target

    def unset_target(self, kind):
        assert(kind in ['file', 'module', 'function', 'class'])
        self._target[kind] = None

    def reset_target(self):
        """."""
        self._target = {}
        self._target['file'] = None
        self._target['module'] = None
        self._target['function'] = None
        self._target['class'] = None

    def report(self,
               message,
               line = None,
               column = None,
               bounding_box = None,
               blocker = False,
               severity = 'error',
               symbol = None,
               filename = None,
               verbose = False):
        """Report message at the given line using the current error context.
        stage: 'syntax', 'semantic' or 'codegen'
        """
        # filter internal errors
        if (self.mode == 'user') and (severity == 'internal'):
            return

        if (severity == 'fatal'):
            blocker = True

        if filename is None:
            filename = self.target['file']

        # TODO improve. it is assumed here that tl and br have the same line
        if bounding_box:
            line   = bounding_box[0]
            column = bounding_box[1]

        if symbol is not None:
            if getattr(symbol, '__module__', '') == '_ast':
                line   = symbol.lineno
                column = symbol.col_offset
                symbol = ast_dump(symbol)
            elif isinstance(symbol, Basic):
                fst = getattr(symbol, 'fst', None)
                if fst is not None:
                    line   = fst.lineno
                    column = fst.col_offset

        info = ErrorInfo(stage=self._parser_stage,
                         filename=filename,
                         line=line,
                         column=column,
                         severity=severity,
                         message=message,
                         symbol=symbol,
                         blocker=blocker)

        if verbose: print(info)

        self.add_error_info(info)

        if blocker:
            if self._parser_stage == 'syntax':
                raise PyccelSyntaxError(message)
            elif self._parser_stage == 'semantic':
                raise PyccelSemanticError(message)
            elif self._parser_stage == 'codegen':
                raise PyccelCodegenError(message)
            else:
                raise PyccelError(message)

    def _add_error_info(self, file, info):
        if file not in self.error_info_map:
            self.error_info_map[file] = []
        self.error_info_map[file].append(info)

    def add_error_info(self, info):
        self._add_error_info(info.filename, info)

    def num_messages(self):
        """Return the number of generated messages."""
        return sum(len(x) for x in self.error_info_map.values())

    def has_warnings(self):
        """Are there any errors that are warnings?"""
        return any(err for errs in self.error_info_map.values() for err in errs if err.severity == 'warning')

    def has_errors(self):
        """Are there any generated errors?"""
        return any(err for errs in self.error_info_map.values() for err in errs if err.severity != 'warning')

    def has_blockers(self):
        """Are there any errors that are blockers?"""
        return any(err for errs in self.error_info_map.values() for err in errs if err.blocker)

    def blocker_filename(self):
        """Return the file with a blocking error, or None if not possible."""
        for errs in self.error_info_map.values():
            for err in errs:
                if err.blocker:
                    return err.filename
        return None

    def format_messages(self, error_info):
        """Return a string list that represents the error messages.
        Use a form suitable for displaying to the user.
        """
        pass

    def check(self):
        """."""
        if self.num_messages() > 0:
            print(self.__str__())

    def __str__(self):
        print_path = (len(self.error_info_map.keys()) > 1)
        text = '{}:\n'.format(PYCCEL)

        for path in self.error_info_map.keys():
            errors = self.error_info_map[path]
            if print_path: text += ' filename :: {path}\n'.format(path=path)
            for err in errors:
                text += ' ' + str(err) + '\n'

        return text

if __name__ == '__main__':
    from pyccel.errors.messages import NO_RETURN_VALUE_EXPECTED
    from pyccel.errors.messages import INCOMPATIBLE_RETURN_VALUE_TYPE
    from pyccel.errors.messages import UNDEFINED_VARIABLE

    errors = Errors()
    errors.set_parser_stage('semantic')

    errors.set_target('script.py', 'file')

    errors.report(NO_RETURN_VALUE_EXPECTED, severity='warning', line=24)

    errors.set_target('make_knots', 'function')
    errors.report(INCOMPATIBLE_RETURN_VALUE_TYPE, symbol='y', severity='error',
                 line=34, column=17)
    errors.unset_target('function')

    errors.check()

    errors.set_target('eval_bsplines', 'function')
    errors.report(UNDEFINED_VARIABLE, symbol='x', severity='error', blocker=True)
    errors.unset_target('function')


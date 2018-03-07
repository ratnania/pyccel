# coding: utf-8
#!/usr/bin/env python

import sys
import os
import argparse

# TODO add version
#  --version             show program's version number and exit


from pyccel.codegen import build_file

class MyParser(argparse.ArgumentParser):
    """
    Custom argument parser for printing help message in case of an error.
    See http://stackoverflow.com/questions/4042452/display-help-message-with-python-argparse-when-script-is-called-without-any-argu
    """
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def _which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def pyccel(files=None, openmp=None, openacc=None, output_dir=None, compiler='gfortran'):
    """
    pyccel console command.
    """
    parser = MyParser(description='pyccel command line')

    # ...
    parser.add_argument('--language', type=str, \
                        help='Target language')
    parser.add_argument('--compiler', type=str, \
                        help='Used compiler')
    parser.add_argument('--openmp', action='store_true', \
                        help='uses openmp')
    parser.add_argument('--openacc', action='store_true', \
                        help='uses openacc')
    parser.add_argument('--execute', action='store_true', \
                        help='executes the binary file')
    parser.add_argument('--show', action='store_true', \
                        help='prints the generated file.')
    parser.add_argument('--debug', action='store_true', \
                        help='compiles the code in a debug mode.')
    parser.add_argument('--output-dir', type=str, \
                        help='Output directory.')

    parser.add_argument('--syntax-only', action='store_true',
                        help='Using pyccel for Syntax Checking')
    parser.add_argument('--semantic-only', action='store_true',
                        help='Using pyccel for Semantic Checking')
    parser.add_argument('--convert-only', action='store_true',
                        help='Converts pyccel files only without build')
    parser.add_argument('--lint', action='store_true', \
                        help='Uses PyLint for static checking.')

    parser.add_argument('--no-modules', action='store_true',
                        help='adds used modules to the generated file')
    parser.add_argument('--verbose', action='store_true', \
                        help='enables verbose mode.')
    parser.add_argument('--analysis', action='store_true', \
                        help='enables code analysis mode.')
    # TODO: remove local_vars later, by using Annotated Comments
    parser.add_argument('--local_vars', type=str, \
                        help='local variables on fast memory.')
    parser.add_argument('--include', type=str, \
                        help='path to include directory.')
    parser.add_argument('--libdir', type=str, \
                        help='path to lib directory.')
    parser.add_argument('--libs', type=str, \
                        help='list of libraries to link with.')

    if not files:
        parser.add_argument('files', metavar='N', type=str, nargs='+',
                            help='a Pyccel file')
    # ...

    # ...
    args = parser.parse_args()
    # ...

    # ...
    if not files:
        files = args.files

    if args.compiler:
        compiler = args.compiler

    if not openmp:
        openmp = args.openmp

    if not openacc:
        openacc = args.openacc

    if not output_dir:
        output_dir = args.output_dir

    if args.convert_only or args.syntax_only or args.semantic_only:
        compiler = None
    # ...

    # ...
    if not files:
        raise ValueError("a python filename must be provided.")

    if len(files) > 1:
        raise ValueError('Expecting one single file for the moment.')
    # ...

    filename = files[0]

    if args.language:
        language = args.language
    else:
        language = 'fortran'

    if compiler:
        if _which(compiler) is None:
            raise ValueError('Could not find {0}'.format(compiler))

    execute = args.execute

    accelerator = None
    if openmp:
        accelerator = "openmp"
    if openacc:
        accelerator = "openacc"

    lint    =  args.lint
    debug    = args.debug
    verbose  = args.verbose
    show     = args.show
    analysis = args.analysis
    include  = args.include
    libdir   = args.libdir
    libs     = args.libs

    if not include:
        include = []
    if not libdir:
        libdir = []
    if not libs:
        libs = []

    no_modules = True
    if not(args.no_modules is None):
        no_modules = args.no_modules
    # ...

    # ...
    from pyccel.parser import Parser
    if args.syntax_only:
        pyccel = Parser(filename)
        ast = pyccel.parse()
    elif args.semantic_only:
        pyccel = Parser(filename)
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)
    elif not analysis:
        build_file(filename, language, compiler,
                   execute=execute, accelerator=accelerator,
                   debug=debug, lint=lint, verbose=verbose, show=show,
                   name=None, include=include,
                   output_dir=output_dir,
                   libdir=libdir, libs=libs,
                   single_file=not(no_modules))
    else:
        from pyccel.complexity.memory import MemComplexity

        local_vars = []
        if args.local_vars:
            local_vars = args.local_vars.split(',')
        complexity = MemComplexity(filename)
        complexity.intensity(verbose=True, local_vars=local_vars)
    # ...

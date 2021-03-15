# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Contains the execute_pyccel function which carries out the main steps required to execute pyccel
"""

import os
import sys
import shutil
from collections import OrderedDict

from pyccel.errors.errors          import Errors, PyccelError
from pyccel.errors.errors          import PyccelSyntaxError, PyccelSemanticError, PyccelCodegenError
from pyccel.errors.messages        import PYCCEL_RESTRICTION_TODO
from pyccel.parser.parser          import Parser
from pyccel.codegen.codegen        import Codegen
from pyccel.codegen.utilities      import construct_flags
from pyccel.codegen.utilities      import compile_files
from pyccel.codegen.python_wrapper import create_shared_library

import pyccel.stdlib as stdlib_folder

__all__ = ['execute_pyccel']

# map internal libraries to their folders inside pyccel/stdlib
internal_libs = {
    'ndarrays' : 'ndarrays',
    "pyc_math" : "math",
}

# map language to its file extension
lang_ext_dict = {
    "c" : ".c",
    "fortran": ".f90",
}

#==============================================================================
# NOTE:
# [..]_dirname is the name of a directory
# [..]_dirpath is the full (absolute) path of a directory

# TODO: change name of variable 'module_name', as it could be a program
# TODO [YG, 04.02.2020]: check if we should catch BaseException instead of Exception
def execute_pyccel(fname, *,
                   syntax_only   = False,
                   semantic_only = False,
                   convert_only  = False,
                   verbose       = False,
                   folder        = None,
                   language      = None,
                   compiler      = None,
                   mpi_compiler  = None,
                   fflags        = None,
                   includes      = (),
                   libdirs       = (),
                   modules       = (),
                   libs          = (),
                   debug         = False,
                   accelerator   = None,
                   output_name   = None):
    """
    Carries out the main steps required to execute pyccel
    - Parses the python file (syntactic stage)
    - Annotates the abstract syntax tree (semantic stage)
    - Generates the translated file(s) (codegen stage)
    - Compiles the files to generate an executable and/or a shared library

    Parameters
    ----------
    fname         : str
                    Name of python file to be translated

    syntax_only   : bool
                    Boolean indicating whether the pipeline should stop
                    after the syntax stage
                    Default : False

    semantic_only : bool
                    Boolean indicating whether the pipeline should stop
                    after the semantic stage
                    Default : False

    convert_only  : bool
                    Boolean indicating whether the pipeline should stop
                    after the codegen stage
                    Default : False

    verbose       : bool
                    Boolean indicating whether debugging messages should be printed
                    Default : False

    folder        : str
                    Path to the working directory
                    Default : folder containing the file to be translated

    language      : str
                    The language which pyccel is translating to
                    Default : fortran

    compiler      : str
                    The compiler used to compile the generated files
                    Default : GNU

    mpi_compiler  : str
                    The compiler used to compile the generated files when mpi is needed.
                    This value must be provided to compile with mpi
                    Default : None (compile with 'compiler')

    fflags        : str
                    The flags passed to the compiler
                    Default : provided by codegen.utilities.construct_flags

    includes      : list
                    list of include directories paths

    libdirs       : list
                    list of paths to directories containing the required libraries

    modules       : list
                    list of files which must also be compiled in order to compile this module

    libs          : list
                    list of required libraries

    debug         : bool
                    Boolean indicating whether the file should be compiled in debug mode
                    (currently this only implies that the flag -fcheck=bounds is added)
                    Default : False

    accelerator   : str
                    Tool used to accelerate the code (e.g. openmp openacc)

    output_name   : str
                    Name of the generated module
                    Default : Same name as the file which was translated
    """

    # Reset Errors singleton before parsing a new file
    errors = Errors()
    errors.reset()

    # TODO [YG, 03.02.2020]: test validity of function arguments

    # Copy list arguments to local lists to avoid unexpected behavior
    includes = [*includes]
    libdirs  = [*libdirs]
    modules  = [*modules]
    libs     = [*libs]

    # Store current directory and add it to sys.path
    # variable to imitate Python's import behavior
    base_dirpath = os.getcwd()
    sys.path.insert(0, base_dirpath)

    # Unified way to handle errors: print formatted error message, then move
    # to original working directory. Caller should then raise exception.
    def handle_error(stage):
        print('\nERROR at {} stage'.format(stage))
        errors.check()
        os.chdir(base_dirpath)

    # Identify absolute path, directory, and filename
    pymod_filepath = os.path.abspath(fname)
    pymod_dirpath, pymod_filename = os.path.split(pymod_filepath)

    # Extract module name
    module_name = os.path.splitext(pymod_filename)[0]

    # Define working directory 'folder'
    if folder is None or folder == "":
        folder = pymod_dirpath
    else:
        folder = os.path.abspath(folder)

    # Define directory name and path for pyccel & cpython build
    pyccel_dirname = '__pyccel__'
    pyccel_dirpath = os.path.join(folder, pyccel_dirname)

    # Create new directories if not existing
    os.makedirs(folder, exist_ok=True)
    os.makedirs(pyccel_dirpath, exist_ok=True)

    # Change working directory to 'folder'
    os.chdir(folder)

    if language is None:
        language = 'fortran'

    # Choose Fortran compiler
    if compiler is None:
        if language == 'fortran':
            compiler = 'gfortran'
        elif language == 'c':
            compiler = 'gcc'

    f90exec = mpi_compiler if mpi_compiler else compiler

    if (language == "c"):
        libs = libs + ['m']
    if accelerator == 'openmp':
        if compiler in ["gcc","gfortran"]:
            if sys.platform == "darwin" and compiler == "gcc":
                libs = libs + ['omp']
            else:
                libs = libs + ['gomp']

        elif compiler == 'ifort':
            libs.append('iomp5')

    # ...
    # Construct flags for the compiler (if one is required)
    if fflags is None and compiler:
        fflags = construct_flags(f90exec,
                                 fflags=None,
                                 debug=debug,
                                 accelerator=accelerator,
                                 includes=())

    # Build position-independent code, suited for use in shared library
    fflags = ' {} -fPIC '.format(fflags)
    # ...

    # Parse Python file
    try:
        parser = Parser(pymod_filepath, show_traceback=verbose)
        parser.parse(verbose=verbose)
    except NotImplementedError as error:
        msg = str(error)
        errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
            severity='error')
    except PyccelError:
        handle_error('parsing (syntax)')
        raise
    if errors.has_errors():
        handle_error('parsing (syntax)')
        raise PyccelSyntaxError('Syntax step failed')

    if syntax_only:
        return

    # Annotate abstract syntax Tree
    try:
        settings = {'verbose':verbose}
        parser.annotate(**settings)
    except NotImplementedError as error:
        msg = str(error)
        errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
            severity='error')
    except PyccelError:
        handle_error('annotation (semantic)')
        # Raise a new error to avoid a large traceback
        raise PyccelSemanticError('Semantic step failed') from None

    if errors.has_errors():
        handle_error('annotation (semantic)')
        raise PyccelSemanticError('Semantic step failed')

    if semantic_only:
        return

    if parser.module_parser:
        parsers = [parser.module_parser, parser]
        program_name = os.path.basename(os.path.splitext(parser.filename)[0])
        module_names = [module_name, program_name]
    else:
        parsers = [parser]
        module_names = [module_name]

    # -------------------------------------------------------------------------
    # get path to pyccel/stdlib/lib_name
    stdlib_path = os.path.dirname(stdlib_folder.__file__)

    internal_libs_name = set()
    internal_libs_path = []
    internal_libs_files = []
    for parser, module_name in zip(parsers, module_names):
        semantic_parser = parser.semantic_parser
        # Generate .f90 file
        try:
            codegen = Codegen(semantic_parser, module_name)
            fname = os.path.join(pyccel_dirpath, module_name)
            fname = codegen.export(fname, language=language)
        except NotImplementedError as error:
            msg = str(error)
            errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
                severity='error')
        except PyccelError:
            handle_error('code generation')
            # Raise a new error to avoid a large traceback
            raise PyccelCodegenError('Code generation failed') from None

        if errors.has_errors():
            handle_error('code generation')
            raise PyccelCodegenError('Code generation failed')

        if language == 'python':
            output_file = (output_name + '.py') if output_name else os.path.basename(fname)
            new_location = os.path.join(folder, output_file)
            if verbose:
                print("cp {} {}".format(fname, new_location))
            shutil.copyfile(fname, new_location)
            continue

        #------------------------------------------------------
        # TODO: collect dependencies and proceed recursively
        # if recursive:
        #     for dep in parser.sons:
        #         # Call same function on 'dep'
        #         pass
        #------------------------------------------------------

        # Iterate over the internal_libs list and determine if the printer
        # requires an internal lib to be included.
        l = codegen.get_printer_imports()
        l.add("ndarrays")
        for lib in internal_libs:
            if lib in codegen.get_printer_imports():
                # get the include folder path and library files
                if lib not in internal_libs_name:
                    # get the library folder name
                    lib_name = internal_libs[lib]
                    # get lib path (stdlib_path/lib_name)
                    lib_path = os.path.join(stdlib_path, lib_name)
                    # remove library folder to avoid missing files and copy
                    # new one from pyccel stdlib
                    lib_dest_path = os.path.join(pyccel_dirpath, lib_name)
                    if os.path.exists(lib_dest_path):
                        shutil.rmtree(lib_dest_path)
                    try:
                        shutil.copytree(lib_path, lib_dest_path, copy_function = shutil.copy)
                    except:
                        print('[Error : copy folder]')

                    # stop after copying lib to __pyccel__ directory for
                    # convert only
                    if convert_only:
                        continue

                    # get library source files
                    source_files = []
                    for e in os.listdir(lib_dest_path):
                        if e[-2:] == '.c' or (e[-4:] == '.f90' and language == fortran):
                            source_files.append(os.path.join(lib_dest_path, e))
                    print('[HERE]', source_files)
                    # compile library source files
                    flags = construct_flags(f90exec,
                                            fflags=fflags,
                                            debug=debug,
                                            includes=[lib_dest_path])

                    try:
                        for f in source_files:
                            compile_files(f, f90exec, flags,
                                            binary=None,
                                            verbose=verbose,
                                            is_module=True,
                                            output=lib_dest_path,
                                            language=language)
                    except Exception:
                        handle_error('C {} library compilation'.format(lib))
                        raise
                    # Add internal lib to internal_libs_name set
                    internal_libs_name.add(lib)
                    # add source file without extension to internal_libs_files
                    internal_libs_files.extend(os.path.splitext(f)[0] for f in source_files)
                    # add library path to internal_libs_path
                    internal_libs_path.append(lib_dest_path)

        if convert_only:
            continue

        # ...
        # Determine all .o files and all folders needed by executable
        def get_module_dependencies(parser, mods=(), folders=()):
            mod_folder = os.path.join(os.path.dirname(parser.filename), "__pyccel__")
            mod_base = os.path.splitext(os.path.basename(parser.filename))[0]

            # Stop conditions
            if parser.metavars.get('ignore_at_import', False) or \
               parser.metavars.get('module_name', None) == 'omp_lib':
                return mods, folders

            # Update lists
            mods = [*mods, os.path.join(mod_folder, mod_base)]
            folders = [*folders, mod_folder]

            # Proceed recursively
            for son in parser.sons:
                mods, folders = get_module_dependencies(son, mods, folders)

            return mods, folders

        dep_mods, inc_dirs = get_module_dependencies(parser)

        # Add internal dependencies
        dep_mods = [*dep_mods, *internal_libs_files]
        inc_dirs = [*inc_dirs, *internal_libs_path]

        # Remove duplicates without changing order
        dep_mods = tuple(OrderedDict.fromkeys(dep_mods))
        inc_dirs = tuple(OrderedDict.fromkeys(inc_dirs))
        # ...

        includes += inc_dirs

        if codegen.is_program:
            modules += [os.path.join(pyccel_dirpath, m) for m in dep_mods[1:]]


        # Construct compiler flags
        flags = construct_flags(f90exec,
                                fflags=fflags,
                                debug=debug,
                                accelerator=accelerator,
                                includes=includes)

        # Compile Fortran code
        #
        # TODO: stop at object files, do not compile executable
        #       This allows for properly linking program to modules
        #
        try:
            compile_files(fname, f90exec, flags,
                            binary=None,
                            verbose=verbose,
                            modules=modules,
                            is_module=codegen.is_module,
                            output=pyccel_dirpath,
                            libs=libs,
                            libdirs=libdirs,
                            language=language)
        except Exception:
            handle_error('Fortran compilation')
            raise

        # For a program stop here
        if codegen.is_program:
            if verbose:
                exec_filepath = os.path.join(folder, module_name)
                print( '> Executable has been created: {}'.format(exec_filepath))
            os.chdir(base_dirpath)
            continue

        # Create shared library
        try:
            sharedlib_filepath = create_shared_library(codegen,
                                                       language,
                                                       pyccel_dirpath,
                                                       compiler,
                                                       mpi_compiler,
                                                       accelerator,
                                                       dep_mods,
                                                       libs,
                                                       libdirs,
                                                       includes,
                                                       flags,
                                                       output_name,
                                                       verbose)
        except NotImplementedError as error:
            msg = str(error)
            errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
                severity='error')
            handle_error('code generation (wrapping)')
            raise PyccelCodegenError(msg) from None
        except PyccelError:
            handle_error('code generation (wrapping)')
            raise
        except Exception:
            handle_error('shared library generation')
            raise

        if errors.has_errors():
            handle_error('code generation (wrapping)')
            raise PyccelCodegenError('Code generation failed')

        # Move shared library to folder directory
        # (First construct absolute path of target location)
        sharedlib_filename = os.path.basename(sharedlib_filepath)
        target = os.path.join(folder, sharedlib_filename)
        shutil.move(sharedlib_filepath, target)
        sharedlib_filepath = target

        if verbose:
            print( '> Shared library has been created: {}'.format(sharedlib_filepath))

    # Print all warnings now
    if errors.has_warnings():
        errors.check()

    # Change working directory back to starting point
    os.chdir(base_dirpath)

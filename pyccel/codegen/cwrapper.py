#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Functions necessary for creating the setup_X.py file which
uses python setuptools to compile a c file and generate the
corresponding shared library file"""
from numpy import get_include as get_numpy_include

def print_list(l):
    """ Convert a list of strings to a string that contains the
    python constructor of a list of strings """
    if isinstance(l,str):
        l = [l]
    return '[{0}]'.format(',\n\t\t\t\t'.join("r'{0}'".format(li) for li in l))

def create_c_setup(mod_name,
        wrapper_file,
        dependencies,
        compiler,
        include = (),
        libs    = (),
        libdirs = (),
        flags   = () ):
    """
    Create the code for the setup file which uses python setuptools
    to compile a c file and generate the corresponding shared
    library file

    Parameters
    ----------
    mod_name : str
            The name of the module that will be created
    dependencies : list
            A list of all files needed for the module
    include : list
            Include directories needed for compiling
            If there is only one then a string can be passed instead of a list
    libs : list
            Libraries needed for compiling
            If there is only one then a string can be passed instead of a list
    libdirs : list
            Library directories needed for compiling
            If there is only one then a string can be passed instead of a list
    flags : list
            Additional flags to pass to the compiler

    Returns
    -------
    code : str
            A string containing the contents of the setup file
    """

    code  = "from distutils.core import Extension, setup\n"
    if compiler in ('icc','ifort'):
        code += "import os"
        code += '\nos.environ["CC"]="icc"\n\n'
        code += '\nos.environ["LDSHARED"]="icc -shared"\n\n'
    else:
        code += "\n"

    wrapper_file = "[ r'{0}' ]".format(wrapper_file)

    deps  = ['{0}.o'.format(d) for d in dependencies]

    mod = '"{mod}"'.format(mod=mod_name)

    files       = ("extra_objects = {0}".format(print_list(deps))
                   if deps else None)

    if include is None:
        include_str = 'include_dirs = {0}'.format(print_list([get_numpy_include()]))
    else:
        include.append(get_numpy_include())
        include_str = ('include_dirs = {0}'.format(print_list(include))
                       if include else None)

    libs_str    = ('libraries = {0}'.format(print_list(libs))
                   if libs else None)
    libdirs_str = ('library_dirs = {0}'.format(print_list(libdirs))
                   if libdirs else None)

    flags_str   = ('extra_compile_args = {0}'.format(print_list(flags))
                   if flags else None)

    linker_flags = [f for f in flags if f.startswith("-Wl")]
    linker_flags_str   = ('extra_link_args = {0}'.format(print_list(linker_flags))
                   if flags else None)

    args = [mod, wrapper_file, files, include_str, libs_str, libdirs_str, flags_str, linker_flags_str]
    args = ',\n\t\t'.join(a for a in args if a is not None)

    code += "extension_mod = Extension({args})\n\n".format(args=args)
    code += "setup(name = \"" + mod_name + "\", ext_modules=[extension_mod])"
    return code


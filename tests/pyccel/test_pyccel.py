# pylint: disable=missing-function-docstring, missing-module-docstring/
import subprocess
import os
import shutil
import sys
import re
import pytest
import numpy as np

#==============================================================================
# UTILITIES
#==============================================================================

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c)
    ],
    scope='module'
)
def language(request):
    return request.param
#------------------------------------------------------------------------------

def get_abs_path(relative_path):
    relative_path = os.path.normpath(relative_path)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base_dir, relative_path)

#------------------------------------------------------------------------------
def get_exe(filename):
    exefile = os.path.splitext(filename)[0]
    if sys.platform == "win32":
        exefile = exefile + ".exe"
    if not os.path.isfile(exefile):
        dirname = os.path.dirname(filename)
        basename = "prog_"+os.path.basename(filename)
        exefile = os.path.join(dirname, os.path.splitext(basename)[0])
        if sys.platform == "win32":
            exefile = exefile + ".exe"
        assert(os.path.isfile(exefile))
    return exefile

#------------------------------------------------------------------------------
def insert_pyccel_folder(abs_path):
    base_dir = os.path.dirname(abs_path)
    base_name = os.path.basename(abs_path)
    return os.path.join(base_dir, "__pyccel__", base_name)

#------------------------------------------------------------------------------
def get_python_output(abs_path, cwd = None):
    if cwd is None:
        p = subprocess.Popen([sys.executable , "%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True)
    else:
        p = subprocess.Popen([sys.executable , "%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

#------------------------------------------------------------------------------
def compile_pyccel(path_dir,test_file, options = ""):
    if options != "":
        options = options.strip().split(' ')
        p = subprocess.Popen([shutil.which("pyccel"), "%s" % test_file] + options, universal_newlines=True, cwd=path_dir)
    else:
        p = subprocess.Popen([shutil.which("pyccel"), "%s" % test_file], universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

#------------------------------------------------------------------------------
def compile_c(path_dir,test_file,dependencies):
    root = insert_pyccel_folder(test_file)[:-3]

    assert(os.path.isfile(root+".c"))

    command = [shutil.which("gcc"), "-O3", "%s.c" % root]
    deps = [dependencies] if isinstance(dependencies, str) else dependencies
    for d in deps:
        d = insert_pyccel_folder(d)
        command.append(d[:-3]+".o")
        command.append("-I"+os.path.dirname(d))

    command.append("-o")
    command.append("%s" % test_file[:-3])

    p = subprocess.Popen(command, universal_newlines=True, cwd=path_dir)
    p.wait()

#------------------------------------------------------------------------------
def compile_fortran(path_dir,test_file,dependencies,is_mod=False):
    root = insert_pyccel_folder(test_file)[:-3]

    assert(os.path.isfile(root+".f90"))

    deps = [dependencies] if isinstance(dependencies, str) else dependencies
    if not is_mod:
        base_dir = os.path.dirname(root)
        base_name = os.path.basename(root)
        prog_root = os.path.join(base_dir, "prog_"+base_name)
        if os.path.isfile(prog_root+".f90"):
            compile_fortran(path_dir, test_file, dependencies, is_mod = True)
            dependencies.append(test_file)
            root = prog_root

    if is_mod:
        command = [shutil.which("gfortran"), "-c", "%s.f90" % root]
    else:
        command = [shutil.which("gfortran"), "-O3", "%s.f90" % root]

    for d in deps:
        d = insert_pyccel_folder(d)
        command.append(d[:-3]+".o")
        command.append("-I"+os.path.dirname(d))

    command.append("-o")
    if is_mod:
        command.append("%s.o" % root)
    else:
        command.append("%s" % test_file[:-3])
    p = subprocess.Popen(command, universal_newlines=True, cwd=path_dir)
    p.wait()

#------------------------------------------------------------------------------
def get_lang_output(abs_path):
    p = subprocess.Popen(["%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

#------------------------------------------------------------------------------
def get_value(string, regex, conversion):
    match = regex.search(string)
    assert(match)
    value = conversion(match.group())
    string = string[match.span()[1]:]
    return value, string

def compare_pyth_fort_output_by_type( p_output, f_output, dtype=float ):

    if dtype is str:
        p_list = [e.strip() for e in re.split('\n', p_output)]
        f_list = [e.strip() for e in re.split('\n', f_output)]
        assert(p_list==f_list)
    elif dtype is complex:
        rx = re.compile('[-0-9.eEj]+')
        p, p_output = get_value(p_output, rx, complex)
        if p.imag == 0:
            p2, p_output = get_value(p_output, rx, complex)
            p = p+p2

        rx = re.compile('[-0-9.eE]+')
        f, f_output  = get_value(f_output, rx, float)
        f2, f_output = get_value(f_output, rx, float)
        f = f+f2*1j
        assert(np.isclose(p,f))
    elif dtype is bool:
        rx = re.compile('TRUE|True|true|1|T|t|FALSE|False|false|F|f|0')
        bool_conversion = lambda m: m.lower() in ['true', 't', '1']
        p, p_output = get_value(p_output, rx, bool_conversion)
        f, f_output = get_value(f_output, rx, bool_conversion)
        assert(p==f)

    elif dtype is float:
        rx = re.compile('[-0-9.eE]+')
        p, p_output = get_value(p_output, rx, float)
        f, f_output = get_value(f_output, rx, float)
        assert(np.isclose(p,f))

    elif dtype is int:
        rx = re.compile('[-0-9eE]+')
        p, p_output = get_value(p_output, rx, int)
        f, f_output = get_value(f_output, rx, int)
        assert(p==f)
    else:
        raise NotImplementedError("Type comparison not implemented")
    return p_output,f_output

#------------------------------------------------------------------------------
def compare_pyth_fort_output( p_output, f_output, dtype=float ):

    if isinstance(dtype,list):
        for d in dtype:
            p_output,f_output = compare_pyth_fort_output_by_type(p_output,f_output,d)
    elif dtype is complex:
        while len(p_output)>0 and len(f_output)>0:
            p_output,f_output = compare_pyth_fort_output_by_type(p_output,f_output,complex)
    elif dtype is str:
        compare_pyth_fort_output_by_type(p_output,f_output,dtype)
    else:
        p_output = p_output.strip().split()
        f_output = f_output.strip().split()
        for p, f in zip(p_output, f_output):
            compare_pyth_fort_output_by_type(p,f,dtype)

#------------------------------------------------------------------------------
def pyccel_test(test_file, dependencies = None, compile_with_pyccel = True,
        cwd = None, pyccel_commands = "", output_dtype = float,
        language = None):
    test_file = os.path.normpath(test_file)

    if (cwd is None):
        cwd = os.path.dirname(test_file)

    cwd = get_abs_path(cwd)

    test_file = get_abs_path(test_file)
    pyth_out = get_python_output(test_file, cwd)

    if language:
        pyccel_commands += " --language="+language
    else:
        language='fortran'

    if dependencies:
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        for i, d in enumerate(dependencies):
            dependencies[i] = get_abs_path(d)
            if not compile_with_pyccel and language=='fortran':
                compile_pyccel (cwd, dependencies[i], pyccel_commands+" -t")
                compile_fortran(cwd, dependencies[i], [], is_mod = True)
            else:
                compile_pyccel(os.path.dirname(dependencies[i]), dependencies[i], pyccel_commands)

    if compile_with_pyccel:
        compile_pyccel(cwd, test_file, pyccel_commands)
    else:
        compile_pyccel (cwd, test_file, pyccel_commands+" -t")
        if not dependencies:
            dependencies = []
        if language=='fortran':
            compile_fortran(cwd, test_file, dependencies)
        else:
            compile_c(cwd, test_file, dependencies)

    lang_out = get_lang_output(get_exe(test_file))
    compare_pyth_fort_output(pyth_out, lang_out, output_dtype)

#==============================================================================
# UNIT TESTS
#==============================================================================
def test_relative_imports_in_project(language):

    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "project_rel_imports")
    pyth_out = get_python_output('runtest.py', cwd=path_dir)

    language_opt = '--language={}'.format(language)
    compile_pyccel(path_dir, 'project/folder1/mod1.py', language_opt)
    compile_pyccel(path_dir, 'project/folder2/mod2.py', language_opt)
    compile_pyccel(path_dir, 'project/folder2/mod3.py', language_opt)
    fort_out = get_python_output('runtest.py', cwd=path_dir)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
def test_absolute_imports_in_project(language):

    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "project_abs_imports")
    pyth_out = get_python_output('runtest.py', cwd=path_dir)

    language_opt = '--language={}'.format(language)
    compile_pyccel(path_dir, 'project/folder1/mod1.py', language_opt)
    compile_pyccel(path_dir, 'project/folder2/mod2.py', language_opt)
    compile_pyccel(path_dir, 'project/folder2/mod3.py', language_opt)
    fort_out = get_python_output('runtest.py', cwd=path_dir)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
def test_rel_imports_python_accessible_folder(language):
    # pyccel is called on scripts/folder2/runtest_rel_imports.py from the scripts folder
    # From this folder python understands relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.runtest_rel_imports import test_func

    pyth_out = str(test_func())

    language_opt = '--language={}'.format(language)
    compile_pyccel(os.path.join(path_dir, "folder2"), get_abs_path("scripts/folder2/folder2_funcs.py"), language_opt)
    compile_pyccel(path_dir, get_abs_path("scripts/folder2/runtest_rel_imports.py"), language_opt)

    p = subprocess.Popen([sys.executable , "%s" % os.path.join(base_dir, "run_import_function.py"), "scripts.folder2.runtest_rel_imports"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_imports_compile(language):
    pyccel_test("scripts/runtest_imports.py","scripts/funcs.py",
            compile_with_pyccel = False, language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_imports_in_folder(language):
    pyccel_test("scripts/runtest_folder_imports.py","scripts/folder1/folder1_funcs.py",
            compile_with_pyccel = False, language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_imports(language):
    pyccel_test("scripts/runtest_imports.py","scripts/funcs.py",
            language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_folder_imports_python_accessible_folder(language):
    # pyccel is called on scripts/folder2/runtest_imports2.py from the scripts folder
    # From this folder python understands relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.runtest_imports2 import test_func

    pyth_out = str(test_func())

    language_opt = '--language={}'.format(language)
    compile_pyccel(os.path.join(path_dir, "folder1"), get_abs_path("scripts/folder1/folder1_funcs.py"),
            language_opt)
    compile_pyccel(path_dir, get_abs_path("scripts/folder2/runtest_imports2.py"),
            language_opt)

    p = subprocess.Popen([sys.executable , "%s" % os.path.join(base_dir, "run_import_function.py"), "scripts.folder2.runtest_imports2"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_folder_imports(language):
    # pyccel is called on scripts/folder2/runtest_imports2.py from the scripts/folder2 folder
    # which is where the final .so file should be
    # From this folder python doesn't understand relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.runtest_imports2 import test_func

    pyth_out = str(test_func())

    language_opt = '--language={}'.format(language)
    compile_pyccel(os.path.join(path_dir,"folder1"), get_abs_path("scripts/folder1/folder1_funcs.py"),
            language_opt)
    compile_pyccel(os.path.join(path_dir,"folder2"), get_abs_path("scripts/folder2/runtest_imports2.py"),
            language_opt)

    p = subprocess.Popen([sys.executable , "%s" % os.path.join(base_dir, "run_import_function.py"), "scripts.folder2.runtest_imports2"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_funcs(language):
    pyccel_test("scripts/runtest_funcs.py", language = language)

#------------------------------------------------------------------------------
# Enumerate not supported in c
def test_inout_func():
    pyccel_test("scripts/runtest_inoutfunc.py")

#------------------------------------------------------------------------------
def test_bool(language):
    pyccel_test("scripts/bool_comp.py", output_dtype = bool, language = language)

#------------------------------------------------------------------------------
def test_expressions(language):
    types = [float, complex, int, float, float, int] + [float]*3 + \
            [complex, int, complex, complex, int, int, float] + [complex]*3 + \
            [float]*3 + [int] + [float]*2 + [int] + [float]*3 + [int] + \
            [float]*3 + [int]*2 + [float]*2 + [int]*5 + [complex] + [bool]*9
    pyccel_test("scripts/expressions.py", language=language,
                output_dtype = types)

#------------------------------------------------------------------------------
# See issue #756 for c problem
def test_generic_functions():
    pyccel_test("scripts/runtest_generic_functions.py",
            dependencies = "scripts/generic_functions.py",
            compile_with_pyccel = False,
            output_dtype = [float,float,float,float,float,float,
                    float,float,float,float,float,float,float,int,float,
                    int,int])

#------------------------------------------------------------------------------
# C does not handle functions in functions
def test_default_arguments():
    pyccel_test("scripts/runtest_default_args.py",
            dependencies = "scripts/default_args_mod.py",
            output_dtype = [int,int,float,float,float,
                float,float,float,float,bool,bool,bool,
                float,float,float,float])

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_pyccel_calling_directory(language):
    cwd = get_abs_path(".")

    test_file = get_abs_path("scripts/runtest_funcs.py")
    pyth_out = get_python_output(test_file)

    language_opt = '--language={}'.format(language)
    compile_pyccel(cwd, test_file, language_opt)

    fort_out = get_lang_output(get_exe(test_file))

    compare_pyth_fort_output( pyth_out, fort_out )

#------------------------------------------------------------------------------
def test_in_specified(language):
    pyccel_test("scripts/runtest_degree_in.py", language=language)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/hope_benchmarks/hope_fib.py",
                                        "scripts/hope_benchmarks/quicksort.py",
                                        "scripts/hope_benchmarks/hope_pisum.py",
                                        "scripts/hope_benchmarks/hope_ln_python.py",
                                        "scripts/hope_benchmarks/hope_pairwise_python.py",
                                        "scripts/hope_benchmarks/point_spread_func.py",
                                        "scripts/hope_benchmarks/simplify.py",
                                        pytest.param("scripts/hope_benchmarks_decorators/fib.py",
                                            marks = pytest.mark.xfail(reason="Issue 344 : Functions and modules cannot share the same name")),
                                        "scripts/hope_benchmarks_decorators/hope_ln_python.py",
                                        "scripts/hope_benchmarks_decorators/hope_pairwise_python.py",
                                        "scripts/hope_benchmarks_decorators/point_spread_func.py",
                                        "scripts/hope_benchmarks_decorators/simplify.py",
                                        "scripts/hope_benchmarks_decorators/hope_fib.py",
                                        "scripts/hope_benchmarks_decorators/quicksort.py",

                                        ] )
def test_hope_benchmarks( test_file ):
    pyccel_test(test_file)

#------------------------------------------------------------------------------
@pytest.mark.c
@pytest.mark.parametrize( "test_file", ["scripts/hope_benchmarks/hope_fib.py",
                                        "scripts/hope_benchmarks/quicksort.py",
                                        "scripts/hope_benchmarks/hope_pisum.py",
                                        "scripts/hope_benchmarks/hope_ln_python.py",
                                        "scripts/hope_benchmarks/hope_pairwise_python.py",
                                        pytest.param("scripts/hope_benchmarks/point_spread_func.py",
                                            marks = pytest.mark.skip(reason="Numpy sum not implemented in c")),
                                        "scripts/hope_benchmarks/simplify.py",
                                        "scripts/hope_benchmarks_decorators/fib.py",
                                        "scripts/hope_benchmarks_decorators/hope_ln_python.py",
                                        "scripts/hope_benchmarks_decorators/hope_pairwise_python.py",
                                        pytest.param("scripts/hope_benchmarks_decorators/point_spread_func.py",
                                            marks = pytest.mark.skip(reason="Numpy sum not implemented in c")),
                                        "scripts/hope_benchmarks_decorators/simplify.py",
                                        "scripts/hope_benchmarks_decorators/quicksort.py",
                                        ] )
def test_hope_benchmarks_c( test_file ):
    pyccel_test(test_file, language='c')

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import.py",
                                        "scripts/import_syntax/from_mod_import_as.py",
                                        "scripts/import_syntax/import_mod.py",
                                        "scripts/import_syntax/import_mod_as.py",
                                        "scripts/import_syntax/from_mod_import_func.py",
                                        "scripts/import_syntax/from_mod_import_as_func.py",
                                        "scripts/import_syntax/import_mod_func.py",
                                        "scripts/import_syntax/import_mod_as_func.py",
                                        "scripts/import_syntax/collisions.py",
                                        "scripts/import_syntax/collisions3.py",
                                        "scripts/import_syntax/collisions5.py",
                                        "scripts/import_syntax/collisions6.py",
                                        ] )
def test_import_syntax( test_file ):
    pyccel_test(test_file)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import_as_user_func.py",
                                        "scripts/import_syntax/from_mod_import_as_user.py",
                                        "scripts/import_syntax/collisions2.py"
                                        ] )
@pytest.mark.xdist_incompatible
def test_import_syntax_user_as( test_file ):
    pyccel_test(test_file, dependencies = "scripts/import_syntax/user_mod.py")

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import_user.py",
                                        "scripts/import_syntax/import_mod_user.py",
                                        "scripts/import_syntax/import_mod_as_user.py",
                                        "scripts/import_syntax/from_mod_import_user_func.py",
                                        "scripts/import_syntax/import_mod_user_func.py",
                                        "scripts/import_syntax/import_mod_as_user_func.py",
                                        ] )
@pytest.mark.xdist_incompatible
def test_import_syntax_user( test_file, language ):
    pyccel_test(test_file, dependencies = "scripts/import_syntax/user_mod.py", language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_import_collisions():
    pyccel_test("scripts/import_syntax/collisions4.py",
            dependencies = ["scripts/import_syntax/user_mod.py", "scripts/import_syntax/user_mod2.py"])

#------------------------------------------------------------------------------
# Numpy sum required
def test_numpy_kernels_compile():
    cwd = get_abs_path(".")
    compile_pyccel(os.path.join(cwd, "scripts/numpy/"), "numpy_kernels.py")

#------------------------------------------------------------------------------
def test_multiple_results(language):
    pyccel_test("scripts/runtest_multiple_results.py",
            output_dtype = [int,float,complex,bool,int,complex,
                int,bool,float,float,float,float,float,float,
                float,float,float,float,float,float
                ,float,float,float,float], language=language)

#------------------------------------------------------------------------------
def test_elemental(language):
    pyccel_test("scripts/decorators_elemental.py", language = language)

#------------------------------------------------------------------------------
def test_print_strings(language):
    types = str
    pyccel_test("scripts/print_strings.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="formated string not implemented in fortran"),
            pytest.mark.fortran]
        )
    )
)
def test_print_sp_and_end(language):
    types = str
    pyccel_test("scripts/print_sp_and_end.py", language=language, output_dtype=types)


#------------------------------------------------------------------------------
def test_c_arrays(language):
    types = [int]*15 + [float]*5 + [int]*25 + [float]* 20 * 5 + \
            [complex] * 3 * 10 + [complex] * 5 + [float] * 10 + [float] * 6 + \
            [float] * 2 * 3 + [complex] * 3 * 10 + [float] * 2 * 3 + [int] * 3
    pyccel_test("scripts/c_arrays.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_arrays_view(language):
    types = [int] * 10 + [int] * 10 + [int] * 4 + [int] * 4 + [int] * 10 + \
            [int] * 6 + [int] * 10
    pyccel_test("scripts/arrays_view.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_headers(language):
    test_file = "scripts/runtest_headers.py"
    test_file = os.path.normpath(test_file)
    test_file = get_abs_path(test_file)

    header_file = 'scripts/headers.pyh'
    header_file = os.path.normpath(header_file)
    header_file = get_abs_path(header_file)

    with open(test_file, 'w') as f:
        code = ("from headers import f\n"
                "def f(x):\n"
                "    y = x\n"
                "    return y\n"
                "print(f(1))\n")

        f.write(code)

    with open(header_file, 'w') as f:
        code =("#$ header metavar ignore_at_import=True\n"
               "#$ header function f(int)")

        f.write(code)

    test_file = os.path.normpath(test_file)
    cwd = os.path.dirname(test_file)
    cwd = get_abs_path(cwd)

    pyccel_commands = " --language="+language

    compile_pyccel(cwd, test_file, pyccel_commands)


    lang_out = get_lang_output(get_exe(test_file))
    assert int(lang_out)

    with open(test_file, 'w') as f:
        code = ("from headers import f\n"
                "def f(x):\n"
                "    y = x\n"
                "    return y\n"
                "print(f(1.5))\n")

        f.write(code)

    with open(header_file, 'w') as f:
        code =("#$ header metavar ignore_at_import=True\n"
               "#$ header function f(float)")

        f.write(code)

    compile_pyccel(cwd, test_file, pyccel_commands)

    lang_out = get_lang_output(get_exe(test_file))
    assert float(lang_out) == 1.5

    with open(test_file, 'w') as f:
        code = ("")
        f.write(code)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/classes/classes.py",
                                        "scripts/classes/classes_1.py",
                                        "scripts/classes/classes_5.py",
                                        ] )
def test_classes( test_file ):
    pyccel_test(test_file, compile_with_pyccel = False)

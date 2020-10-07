# pylint: disable=missing-function-docstring, missing-module-docstring/
import os
import shutil
import pytest

__all__ = ['setup', 'teardown']

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.c]
        )
    ]
)
def language(request):
    return request.param

def pytest_runtest_setup(item):
    teardown()

def pytest_runtest_teardown(item, nextitem):
    teardown()

def teardown(path_dir = None):
    if path_dir is None:
        path_dir = os.path.dirname(os.path.realpath(__file__))

    for root, _, files in os.walk(path_dir):
        for name in files:
            if name.startswith(".coverage"):
                shutil.copyfile(os.path.join(root,name),os.path.join(os.getcwd(),name))

    files = os.listdir(path_dir)
    for f in files:
        file_name = os.path.join(path_dir,f)
        if f in  ("__pyccel__", "__epyccel__"):
            shutil.rmtree( file_name )
        elif not os.path.isfile(file_name):
            teardown(file_name)
        elif not f.endswith(".py") and not f.endswith(".rst"):
            os.remove(file_name)

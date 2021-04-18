Welcome to Pyccel
=================

 |build-status| |codacy|

**Pyccel** stands for Python extension language using accelerators.

The aim of **Pyccel** is to provide a simple way to generate automatically, parallel low level code. The main uses would be:

1. Convert a *Python* code (or project) into a Fortran or C code.

2. Accelerate *Python* functions by converting them to *Fortran* or *C* functions.

**Pyccel** can be viewed as:

- *Python-to-Fortran/C* converter

- a compiler for a *Domain Specific Language* with *Python* syntax

Pyccel comes with a selection of **extensions** allowing you to convert calls to some specific python packages to Fortran/C. The following packages will be covered (partially):

- numpy
- scipy
- mpi4py
- h5py (not available yet)

If you are eager to try Pyccel out, we recommend reading our `quick-start guide <https://github.com/pyccel/pyccel/blob/master/tutorial/quickstart.md>`_!

Pyccel Installation Methods
***************************

Pyccel can be installed on virtually any machine that provides Python 3, the pip package manager, a C/Fortran compiler, and an Internet connection.
Some advanced features of Pyccel require additional non-Python libraries to be installed, for which we provide detailed instructions below.

Alternatively, Pyccel can be deployed through a **Linux Docker image** that contains all dependencies, and which can be setup with any version of Pyccel.
For more information, please read the section on `Pyccel container images`_.


Requirements
============

First of all, Pyccel requires a working Fortran/C compiler

For Fortran it supports

-   GFortran <https://gcc.gnu.org/fortran/>
-   Intel® Fortran Compiler <https://software.intel.com/en-us/fortran-compilers>
-   PGI Fortran <https://www.pgroup.com/index.htm>

For C it supports

-   Gcc <https://gcc.gnu.org/>
-   Intel® Compiler <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html>
-   PGI <https://www.pgroup.com/index.htm>

In order to perform fast linear algebra calculations, Pyccel uses the following libraries:

- BLAS (Basic Linear Algebra Subprograms) <http://www.netlib.org/blas/>
- LAPACK (Linear Algebra PACKage) <http://www.netlib.org/lapack/>

Finally, Pyccel supports distributed-memory parallel programming through the Message Passing Interface (MPI) standard; hence it requires an MPI library like

- Open-MPI <https://www.open-mpi.org/>
- MPICH <https://www.mpich.org/>
- Intel® MPI Library <https://software.intel.com/en-us/mpi-library>

We recommend using GFortran/Gcc and Open-MPI.

Pyccel also depends on several Python3 packages, which are automatically downloaded by pip, the Python Package Installer, during the installation process. In addition to these, unit tests require the *scipy*, *mpi4py*, *pytest* and *coverage* packages, while building the documentation requires Sphinx <http://www.sphinx-doc.org/>.



Linux Debian/Ubuntu/Mint
************************

To install all requirements on a Linux Ubuntu machine, just use APT, the Advanced Package Tool::

  sudo apt update
  sudo apt install gcc
  sudo apt install gfortran
  sudo apt install libblas-dev liblapack-dev
  sudo apt install libopenmpi-dev openmpi-bin

Linux Fedora/CentOS/RHEL
************************

Install all requirements using the DNF software package manager::

  su
  dnf check-update
  dnf install gcc
  dnf install gfortran
  dnf install blas-devel lapack-devel
  dnf install openmpi-devel
  exit

Similar commands work on Linux openSUSE, just replace ``dnf`` with ``zypper``.

Mac OS X
********

On an Apple Macintosh machine we recommend using Homebrew <https://brew.sh/>::

  brew update
  brew install gcc
  brew install openblas
  brew install lapack
  brew install open-mpi

This requires that the Command Line Tools (CLT) for Xcode are installed.

Windows
*******

Support for Windows is still experimental, and the installation of all requirements is more cumbersome.
We recommend using Chocolatey <https://chocolatey.org/> to speed up the process, and we provide commands that work in a git-bash shell.
In an Administrator prompt install git-bash (if needed), a Python3 Anaconda distribution, and a GCC compiler::

  choco install git
  choco install anaconda3
  choco install mingw

Download x64 BLAS and LAPACK DLLs from https://icl.cs.utk.edu/lapack-for-windows/lapack/::

  WEB_ADDRESS=https://icl.cs.utk.edu/lapack-for-windows/libraries/VisualStudio/3.7.0/Dynamic-MINGW/Win64
  LIBRARY_DIR=/c/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/lib
  curl $WEB_ADDRESS/libblas.dll -o $LIBRARY_DIR/libblas.dll
  curl $WEB_ADDRESS/liblapack.dll -o $LIBRARY_DIR/liblapack.dll

Generate static MS C runtime library from corresponding dynamic link library::

  cd "$LIBRARY_DIR"
  cp $SYSTEMROOT/SysWOW64/vcruntime140.dll .
  gendef vcruntime140.dll
  dlltool -d vcruntime140.def -l libmsvcr140.a -D vcruntime140.dll
  cd -

Download MS MPI runtime and SDK, then install MPI::

  WEB_ADDRESS=https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1
  curl -L $WEB_ADDRESS/msmpisetup.exe -o msmpisetup.exe
  curl -L $WEB_ADDRESS/msmpisdk.msi -o msmpisdk.msi
  ./msmpisetup.exe
  msiexec //i msmpisdk.msi

**At this point, close and reopen your terminal to refresh all environment variables!**

In Administrator git-bash, generate mpi.mod for gfortran according to https://abhilashreddy.com/writing/3/mpi_instructions.html::

  cd "$MSMPI_INC"
  sed -i 's/mpifptr.h/x64\/mpifptr.h/g' mpi.f90
  sed -i 's/mpifptr.h/x64\/mpifptr.h/g' mpif.h
  gfortran -c -D_WIN64 -D INT_PTR_KIND\(\)=8 -fno-range-check mpi.f90
  cd -

Generate static libmsmpi.a from msmpi.dll::

  cd "$MSMPI_LIB64"
  cp $SYSTEMROOT/SysWOW64/msmpi.dll .
  gendef msmpi.dll
  dlltool -d msmpi.def -l libmsmpi.a -D msmpi.dll
  cd -

Before installing Pyccel and using it, the Anaconda environment should be activated with::

  source /c/tools/Anaconda3/etc/profile.d/conda.sh
  conda activate

On Windows and/or Anaconda Python, use `pip` instead of `pip3` for the Installation of pyccel below.

Installation
============

From PyPi
*********

Simply run, for a user-specific installation::

  pip3 install --user pyccel

or::

  sudo pip3 install pyccel

for a system-wide installation.

From sources
************

* **Standard mode**::

    git clone git@github.com:pyccel/pyccel.git
    cd pyccel
    pip3 install --user .

* **Development mode**::

    git clone git@github.com:pyccel/pyccel.git
    cd pyccel
    pip3 install --user -e .

this will install a *python* library **pyccel** and a *binary* called **pyccel**.
Any required Python packages will be installed automatically from PyPI.


Additional packages
===================

In order to run the unit tests and to get a coverage report, five additional Python packages should be installed:::

  pip3 install --user scipy
  pip3 install --user mpi4py
  pip3 install --user tblib
  pip3 install --user pytest
  pip3 install --user coverage

Most of the unit tests can also be run in parallel. This can be done by installing one additional package::

  pip3 install --user pytest-xdist

Testing
=======

To test your Pyccel installation please run the script *tests/run_tests_py3.sh* (Unix), or *tests/run_tests.bat* (Windows).

Continuous testing runs on github actions: <https://github.com/pyccel/pyccel/actions?query=branch%3Amaster>


Pyccel Container Images
=======================

Pyccel container images are available through both Docker Hub (docker.io) and the GitHub Container Registry (ghcr.io).

The images:

- are based on ubuntu:latest
- use distro packaged python3, gcc, gfortran, blas and openmpi
- support all pyccel releases except the legacy "0.1"

Image tags match pyccel releases.

In order to implement your pyccel accelerated code, you can use a host based volume during the pyccel container creation.

For example::

  docker pull pyccel/pyccel:v1.0.0
  docker run -it -v $PWD:/data:rw  pyccel/pyccel:v1.0.0 bash

If you are using SELinux, you will need to set the right context for your host based volume.
Alternatively you may have docker or podman set the context using -v $PWD:/data:rwz instead of -v $PWD:/data:rw .

.. |build-status| image:: https://github.com/pyccel/pyccel/workflows/master_tests/badge.svg
    :alt: build status
    :scale: 100%
    :target: https://github.com/pyccel/pyccel/actions?query=workflow%3Amaster_tests

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/9723f47b95db491886a0e78339bd4698
    :alt: Codacy Badge
    :scale: 100%
    :target: https://www.codacy.com/gh/pyccel/pyccel?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pyccel/pyccel&amp;utm_campaign=Badge_Grade

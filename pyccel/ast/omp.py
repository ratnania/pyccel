# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
OpenMP has several constructs and directives, and this file contains the OpenMP types that are supported.
We represent some types with the OmpAnnotatedComment type.
These types are detailed on our documentation:
https://github.com/pyccel/pyccel/blob/master/tutorial/openmp.md
"""

from .basic import Basic

class OmpAnnotatedComment(Basic):

    """Represents an OpenMP Annotated Comment in the code.

    Parameters
    ----------

    txt: str
        statement to print

    combined: List (Optional)
        constructs to be combined with the current construct

    Examples
    --------
    >>> from pyccel.ast.omp import OmpAnnotatedComment
    >>> OmpAnnotatedComment('parallel')
    OmpAnnotatedComment(parallel)
    """
    _attribute_nodes = ()
    _is_multiline = False

    def __init__(self, txt, has_nowait=False, combined=None):
        self._txt = txt
        self._combined = combined
        self._has_nowait = has_nowait
        super().__init__()

    @property
    def is_multiline(self):
        """Used to check if the construct needs brackets."""
        return self._is_multiline

    @property
    def name(self):
        """Name of the construct."""
        return ''

    @property
    def txt(self):
        """Used to store clauses."""
        return self._txt

    @property
    def combined(self):
        """Used to store the combined construct of a directive."""
        return self._combined

    def __getnewargs__(self):
        """Used for Pickling self."""

        args = (self.txt, self.combined)
        return args

class OMP_For_Loop(OmpAnnotatedComment):
    """ Represents an OpenMP Loop construct. """
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

    @property
    def name(self):
        """Name of the construct."""
        return 'for'

class OMP_Simd_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Simd construct"""
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Distribute_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Distribute construct"""
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Parallel_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Parallel construct. """
    _is_multiline = True
    @property
    def name(self):
        """Name of the construct."""
        return 'parallel'

class OMP_Task_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Task construct. """
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Single_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Single construct. """
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Critical_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Critical construct. """
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Master_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Master construct. """
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Masked_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Masked construct. """
    _is_multiline = True
    @property
    def name(self):
        """Name of the construct."""
        return 'masked'

class OMP_Cancel_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Cancel construct. """
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Target_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Target construct. """
    _is_multiline = True
    @property
    def name(self):
        """Name of the construct."""
        return 'target'

class OMP_Teams_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Teams construct. """
    _is_multiline = True
    @property
    def name(self):
        """Name of the construct."""
        return 'teams'

class OMP_Sections_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Sections construct. """
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Section_Construct(OmpAnnotatedComment):
    """ Represent OpenMP Section construct. """
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class Omp_End_Clause(OmpAnnotatedComment):
    """ Represents the End of an OpenMP block. """
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

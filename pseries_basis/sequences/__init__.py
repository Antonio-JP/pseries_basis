r'''
    Sequences module.

    This module contains the main structures and algorithms to handle generic infinite sequences.
    This module differs from the module :mod:`sage.structure.sequence:Sequence` in the sense that
    the data of the sequence can be infinite.

    A sequence is a map from a cartesian product of the natural numbers to a *universe*. 
    The universe has to be a Parent structure in SageMath.

    .. MATH::

        s: \mathbb{N}^k \longrightarrow \mathbb{U}

    As for any usual function, there is a natural algebraic structure inherited from the universe
    `\mathbb{U}`. If `\mathbb{U}` is a ring, then the set of sequences inherits a ring structure
    using termwise operations.

    .. MATH::

        (s_1 + s_2)(n) = s_1(n) + s_2(n),\qquad (s_1 s_2)(n) = s_1(n)s_2(n).

    This module have several submodules:

    * :mod:`~.base`: this module contains the base structure for sequences (see :class:`.base.Sequence`) that
      is used later for further specializations and special cases of sequences. In particular, it includes
      the connection to the Sage categories and Sage Parent-Element framework. 
    * :mod:`~.element`: this module contains the main classes that implements the sequences in different ways.
      These subclasses will define a poset (partially-ordered set) of subclasses that will be used when
      computing with sequences to decide the final class of a sequence.
    * :mod:`~.qsequences`: implement all the necessary classes to handle and manipulate `q`-sequences.
'''
from .base import *
from .element import *
from .qsequences import *
r'''
    Module with the specific type of sequences that are used in the module :mod:`.base`.

    Since each of these are extension of the class :class:`~.base.Sequence`, they all inherit
    from this calss and fills the requirements to work properly in the Sequences setting:

    * Override the class method :func:`register_class` with the desired classes we want to be directly below.
        * Implement the following methods:
          - :func:`_change_class`: receives a class (given when registering the sequence class) and cast the current sequence to the new class.
          - :func:`_neg_`: implement the negation of a sequence for a given class.
          - :func:`_final_add`: implement the addition for two sequences of the same parent and class.
          - :func:`_final_sub`: implement the difference for two sequences of the same parent and class.
          - :func:`_final_mul`: implement the hadamard product for two sequences of the same parent and class.
          - :func:`_final_div`: implement the hadamard division for two sequences of the same parent and class.
          - :func:`_final_mod`: implement the hadamard module for two sequences of the same parent and class.
          - :func:`_final_floordiv`: implement the hadamard floor division for two sequences of the same parent and class.
        * Consider updating the following methods:
          - :func:`_element` to adjust how the sequence is computed.
          - :func:`_shift` to adjust the output of the shifted sequence.
          - :func:`_subsequence` to adjust the output of the subsequence.
          - :func:`_slicing` to adjust the output of the sliced sequence.
'''
from __future__ import annotations

from pseries_basis.sequences.base import Sequence
from .base import Sequence

class ExpressionSequence(Sequence):
    pass

class RationalSequence(Sequence):
    pass
    
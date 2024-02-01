r'''
    Module to define q-sequences

    A q-sequence is a sequence whose value is define for `q^n` instead of simply `n`. This, in particular,
    has a main implication on how things are computed. More precisely, we need an element in the universe 
    that will act as `q`.

    Usually, `q` is a transcendental variable over the original universe, however, we may allow any element 
    in the universe of a sequence to take the role of `q`.

    Since the behavior of these sequences are quite different to the standard sequences, the relation 
    in terms of class casting (see :class:`.base.Sequence`) is reset for q-sequences, meaning that 
    besides the basic relation with base sequences and constant sequences, they are not related with any 
    other classical sequences. This imply that any operation will fall to the basic callable sequences.

    These classes inherit from the main class :class:`.base.Sequence`, hence they need to implement the 
    following method in order to interact properly with other sequences:

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
'''
from __future__ import annotations

import logging

from collections.abc import Callable
from typing import Any
from sage.all import Expression, SR

from .base import Sequence
from .element import ExpressionSequence, RationalSequence

logger = logging.getLogger(__name__)

def is_QSequence(element: Sequence):
    return isinstance(element, Sequence) and hasattr(element, "q")

def QSequence(sequence: Callable[..., Any], universe=None, dim: int = 1, *, q, _extend_by_zero=False, **kwds):
    output = Sequence(sequence, universe, dim, _extend_by_zero=_extend_by_zero, q=q)
    if not q in output.universe:
        raise TypeError("[QSequence] The element `q` must be in the universe.")
    return output

def QExpressionSequence(expression: Expression, variables=None, universe=None, *, q, power: int = 1, _extend_by_zero=False, **kwds):
    meanings = QSequence(lambda n : SR(q)**(power*n), universe, 1, q=q, **kwds)
    output = ExpressionSequence(expression, variables, universe, meanings=meanings, _extend_by_zero=_extend_by_zero, q=q, **kwds)
    if not q in output.universe:
        raise TypeError("[QSequence] The element `q` must be in the universe.")
    return output

def QRationalSequence(rational, variables=None, universe=None, *, q, power: int = 1, _extend_by_zero=False, **kwds):
    meanings = QSequence(lambda n : SR(q)**(power*n), universe, 1, q=q, **kwds)
    output = RationalSequence(rational, variables, universe, meanings=meanings, _extend_by_zero=_extend_by_zero, q=q, **kwds)
    if not q in output.universe:
        raise TypeError("[QSequence] The element `q` must be in the universe.")
    return output

__all__ = ["QSequence", "QExpressionSequence", "QRationalSequence", "is_QSequence"]
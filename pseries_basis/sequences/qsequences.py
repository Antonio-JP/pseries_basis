r'''
    Module to define q-sequences

    A q-sequence is a sequence whose value is define for `q^n` instead of simply `n`. This, in particular,
    has a main implication on how things are computed. More precisely, we need an element in the universe 
    that will act as `q`.

    Usually, `q` is a transcendental variable over the original universe, however, we may allow any element 
    in the universe of a sequence to take the role of `q`.

    Since the behavior of these sequences are quite different to the standard sequences, the relation 
    in terms of class casting (see :class:`.sequences.Sequence`) is reset for q-sequences, meaning that 
    besides the basic relation with base sequences and constant sequences, they are not related with any 
    other classical sequences. This imply that any operation will fall to the basic callable sequences.
'''
from __future__ import annotations

import logging

from .base import Sequence
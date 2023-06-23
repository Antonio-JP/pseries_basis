r'''
    Sage package for Power Series Basis.

    This module introduces the basic structures in Sage for computing with *Power
    Series Basis*. We based this work in the paper :arxiv:`2202.05550`
    by M. Petkovšek, where all definitions and proofs for the algorithms here can be found.

    A Power Series basis is defined as a sequence `\{f_n\}_{n\in\mathbb{N}} \subset \mathbb{K}[[x]]`
    that form a `\mathbb{K}`-basis of the whole ring of formal power series. We distinguish
    between two basic type of basis:

    * Polynomial basis: here `f_n \in \mathbb{K}[x]` with degree equal to `n`.
    * Order basis: here `ord(f_n) = n`, meaning that `f_n = x^n g_n(x)` such that `g(0) \neq 0`.

    Any formal power series `g(x)` can be expressed in terms of the power series basis:

    .. MATH::

        g(x) = \sum_{n \in \mathbb{N}} \alpha_n f_n.

    The main aim of this work is to understand which `\mathbb{K}`-linear operators over the
    ring of formal power series are *compatible* with a power series basis, meaning that, 
    `L\cdot g(x) = 0` if and only if the sequence `\alpha_n` is P-recursive.

    EXAMPLES::

        sage: from pseries_basis import *

    This package includes no example since all the structures it offers are abstract, so they should
    never be instantiated. For particular examples and test, look to the modules :mod:`~pseries_basis.factorial.factorial_basis`
    and :mod:`~pseries_basis.factorial.product_basis`.
'''
from __future__ import annotations
from collections.abc import Callable

import logging
logger = logging.getLogger(__name__)

from .sequences.base import Sequence, SequenceSet

class PSBasis(Sequence):
    def __init__(self, sequence: Callable[[int], Sequence], universe=None, *, _extend_by_zero=False):
        # We check the case we provide a 2-dimensional sequence
        if isinstance(sequence, Sequence) and sequence.dim == 2:
            universe = sequence.universe if universe is None else universe; or_sequence = sequence
            sequence = lambda n : or_sequence.slicing((0,n))
        else:
            if universe is None:
                raise ValueError(f"When argument is callable we require a universe argument for the basis.")
        self.__inner_universe = universe
        super().__init__(sequence, SequenceSet(1, universe), 1, _extend_by_zero=_extend_by_zero)

    @property
    def base(self):
        return self.__inner_universe
    
    def _element(self, *indices: int):
        return super()._element(*indices).change_universe(self.base)
    
    def as_2dim(self) -> Sequence:
        return Sequence(lambda n,k : self(n)(k), self.base, 2, _extend_by_zero = self._Sequence__extend_by_zero)

        

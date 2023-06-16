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

from collections.abc import Callable

from pseries_basis.sequences.base import Sequence
from .base import Sequence

class ExpressionSequence(Sequence):
    pass

class RationalSequence(Sequence):
    pass

class ConstantSequence(Sequence):
    def __init__(self, value, universe=None, dim: int = 1, *, _extend_by_zero=True):
        super().__init__(None, universe, dim, _extend_by_zero=_extend_by_zero)
        self.__value = self.universe(value)

    @classmethod
    def _resgister_class(cls, *super_classes):
        return super()._resgister_class(Sequence, ExpressionSequence, RationalSequence)
    
    def _change_class(self, cls):
        if cls == ExpressionSequence:
            raise NotImplementedError("Cast to ExpressionSequence not yet implemented")
        elif cls == RationalSequence:
            raise NotImplementedError("Cast to RationalSequence not yet implemented")
        elif cls == Sequence:
            return Sequence(lambda *n: self.__value, self.universe, self.dim)
        else:
            raise NotImplementedError(f"Class {cls} not recognized from a ConstantSequence")

    def _neg_(self) -> ConstantSequence:
        return ConstantSequence(-self.__value, self.universe, self.dim)
    
    def _final_add(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value + other.__value, self.universe, self.dim)
    def _final_sub(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value - other.__value, self.universe, self.dim)
    def _final_mul(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value * other.__value, self.universe, self.dim)
    def _final_div(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value / other.__value, self.universe, self.dim)
    def _final_mod(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value % other.__value, self.universe, self.dim)
    def _final_floordiv(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value // other.__value, self.universe, self.dim)

    def _element(self, *indices: int):
        return self.__value
    def _shift(self, *shifts):
        return self
    def _subsequence(self, final_input: dict[int, Sequence]):
        return self
    def _slicing(self, values: dict[int, int]):
        if len(values) >= self.dim:
            return self.__value
        return ConstantSequence(self.__value, self.universe, self.dim - len(values))
    
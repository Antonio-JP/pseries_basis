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

from sage.all import SR, ZZ
from sage.misc.functional import log

from pseries_basis.sequences.base import Sequence
from .base import Sequence, ConstantSequence
from .element import RationalSequence

logger = logging.getLogger(__name__)

def logq(a,q):
    try:
        return ZZ(log(a,q))
    except:
        return ZZ(log(SR(a), SR(q)))

class QSequence(Sequence):
    r'''
        Base class for q-sequences.

        A `q`-sequence is a normal sequence where the elements are generated by evaluating `n -> q^n` for 
        some given value of `q` in the universe of the sequence.

        In general, `q`-sequences appear when taking a transcendental element in the universe (i.e.,
        we can decompose the universe into a field of rational fractions over this `q`). However, the 
        implementation provided by this class is not limited to this.

        INPUT:

        * ``sequence``: a callable such that, evaluated on `q^n` gives the sequence defined.
        * ``universe``: a Parent structure on SageMath as in :class:`.base.Sequence`.
        * ``dim``: number of variables defining the sequence, as in :class:`.base.Sequence`.
        * ``q``: indicates the element to be used as `q`.

        TODO: add examples and tests.
    '''
    def __init__(self, sequence: Callable[..., Any], universe=None, dim: int = 1, *, q, _extend_by_zero=False):
        Sequence.__init__(self, sequence, universe, dim, _extend_by_zero=_extend_by_zero)
        
        self.__q = self.universe(q)

    @property
    def q(self):
        return self.__q
    
    ## Methods for the casting of sequences
    def args_to_self(self):
        return [self._Sequence__sequence], {"universe": self.universe, "dim": self.dim, "q": self.q, "_extend_by_zero": self._Sequence__extend_by_zero}

    @classmethod
    def _change_from_class(cls, sequence: Sequence, **extra_info):
        if isinstance(sequence, ConstantSequence):
            return QSequence(
                lambda *_ : sequence._element(*(sequence.dim*[0])), 
                universe = sequence.universe,
                dim = sequence.dim,
                q = extra_info.get("q", sequence.universe.gens()[0]),
                _extend_by_zero=sequence._Sequence__extend_by_zero
            )
        else:
            raise NotImplementedError(f"The class {sequence.__class__} not recognized as subclass of {cls}")
        
    def extra_info(self) -> dict:
        dict = super().extra_info()
        dict["q"] = self.q
        return dict

    ## Methods for sequence arithmetic
    def _neg_(self) -> Sequence:
        return QSequence(lambda *n : (-1)*self._Sequence__sequence(*n), self.universe, dim = self.dim, q = self.q) 
    
    def _final_add(self, other:Sequence) -> Sequence:
        return QSequence(lambda *n: self._Sequence__sequence(*n) + other._Sequence__sequence(*n), self.universe, self.dim, q = self.q)
    def _final_sub(self, other:Sequence) -> Sequence:
        return QSequence(lambda *n: self._Sequence__sequence(*n) - other._Sequence__sequence(*n), self.universe, self.dim, q = self.q)
    def _final_mul(self, other:Sequence) -> Sequence:
        return QSequence(lambda *n: self._Sequence__sequence(*n) * other._Sequence__sequence(*n), self.universe, self.dim, q = self.q)
    def _final_div(self, other:Sequence) -> Sequence:
        return QSequence(lambda *n: self._Sequence__sequence(*n) / other._Sequence__sequence(*n), self.universe, self.dim, q = self.q)
    def _final_mod(self, other:Sequence) -> Sequence:
        return QSequence(lambda *n: self._Sequence__sequence(*n) % other._Sequence__sequence(*n), self.universe, self.dim, q = self.q)
    def _final_floordiv(self, other:Sequence) -> Sequence:
        return QSequence(lambda *n: self._Sequence__sequence(*n) // other._Sequence__sequence(*n), self.universe, self.dim, q = self.q)

    ## Methods for operations on Sequences
    def _element(self, *indices: int):
        return super()._element(*[self.q**ZZ(i) if i in ZZ else SR(self.q)**i for i in indices])

    def _shift(self, *shifts):
        return QSequence(lambda *n : self._Sequence__sequence(*[self.q**shifts[i] * n[i] for i in range(self.dim)]), self.universe, dim=self.dim, q = self.q)
    def _subsequence(self, final_input: dict[int, Sequence]):
        return QSequence(
            lambda *n : self._Sequence__sequence(*[self.q**(final_input[i]._element(logq(n[i], self.q))) if i in final_input else n[i] for i in range(self.dim)]), 
            self.universe, 
            self.dim,
            q = self.q
        )
    def _slicing(self, values: dict[int, int]):
        def to_original_input(n: list):
            result = []; read = 0
            for i in range(self.dim):
                if i in values:
                    result.append(self.q**values[i])
                else:
                    result.append(n[read]); read += 1
            return result
        return QSequence(lambda *n: self._Sequence__sequence(*to_original_input(n)), self.universe, self.dim - len(values), q = self.q)
    def _swap(self, src: int, dst: int):
        def __swap_index(*n):
            n = list(n)
            n[src], n[dst] = n[dst], n[src]
            return tuple(n)

        return QSequence(lambda *qn : self._Sequence__sequence(*__swap_index(*qn)), self.universe, self.dim, q = self.q)
    
    ## Other methods
    def is_hypergeometric(self, index: int = None, *, _bound=30) -> tuple[bool, Sequence]:
        raise NotImplementedError(f"is_hypergeometric with guessing is not working as expected.")
        # if self.dim > 1:
        #     raise NotImplementedError(f"is_hypergeometric not implemented for multidimensional q-sequences")
        # elif index != None and index != 0:
        #     raise IndexError(f"{index=} not valid for a sequence of dimension {self.dim}")
        # from ore_algebra import OreAlgebra, guess
        # from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
        # R = PolynomialRing(self.universe, "q_n"); qn = R.gens()[0]
        # OQ = OreAlgebra(R, ("Q", R.Hom(R)(self.q*qn), lambda _ : 0))
        # try:
        #     guessed = guess(self[:_bound], OQ, order=1)
        # except ValueError: #no relations found
        #     return False,None
        # if guessed.order() != 1:
        #     return False,None
        # rat_func = -guessed[0]/guessed[1]
        # return True, QSequence(lambda n : rat_func(**{str(qn): n}), self.universe, 1, q=self.q)

class QRationalSequence(QSequence, RationalSequence):
    r'''
        Class for representing rational `q`-sequences

        A rational `q`-sequences is a `q`-sequence that can be expressed as a rational function in `q^n`
        for each of the dimensions of the sequence. For example, the bidimensional sequence `q^{n-k}` is
        rational, since we can write it as `q^n / q^k \leftrightarrow x/y`.

        The actual implementation of the arithmetic operations of this class is based on the implementation of 
        :class:`.element.RationalSequence`. On the other hand, other operations are implemented
        following the ideas from :class:`QSequence`, since the behavior of the shifts are different between
        the classical sequences and the `q`-sequences.

        TODO: add examples of rational q-sequences.
    '''
    def __init__(self, rational, variables=None, universe=None, *, q, _extend_by_zero=False):
        # we set up the generic expression and check the arguments
        RationalSequence.__init__(self, rational, variables=variables, universe=universe, _extend_by_zero=_extend_by_zero) 
        # we set up the value for q
        QSequence.__init__(self, None, self.universe, q=q, _extend_by_zero=_extend_by_zero)

    ## Methods for the casting of sequences
    @classmethod
    def register_class(cls):
        return cls._register_class([QSequence], [])
    
    def args_to_self(self):
        return [self.generic()], {"variables": self.variables(), "universe": self.universe, "q": self.q, "_extend_by_zero": self._Sequence__extend_by_zero}
    
    def _change_class(self, goal_class, **_):
        if goal_class != QSequence:
            raise NotImplementedError
        return QSequence(
            lambda *n : self.generic()(**{str(v): n[i] for (i,v) in enumerate(self.variables())}),
            self.universe, 
            self.dim, 
            q = self.q,
            _extend_by_zero = self._Sequence__extend_by_zero
        )
    
    @classmethod
    def _change_from_class(cls, sequence: Sequence, **extra_info):
        if not isinstance(sequence, ConstantSequence):
            raise NotImplementedError(f"The class {sequence.__class__} not recognized as subclass of {cls}")
    
        return QRationalSequence(
            extra_info["field"](sequence(*(sequence.dim*[0]))),
            variables=extra_info.get("variables", [f"qn_{i}" for i in range(sequence.dim)] if sequence.dim > 1 else ["qn"]),
            universe=sequence.universe,
            q = extra_info.get("q"),
            _extend_by_zero=sequence._Sequence__extend_by_zero
        )
       
    def extra_info(self) -> dict:
        dict = QSequence.extra_info(self)
        dict.update(RationalSequence.extra_info(self))
        return dict
    
    ## Methods fro sequence arithmetic
    def _neg_(self) -> Sequence:
        return RationalSequence._neg_(self)
    def _final_add(self, other: Sequence) -> Sequence:
        return RationalSequence._final_add(self,other)
    def _final_sub(self, other: Sequence) -> Sequence:
        return RationalSequence._final_sub(self,other)
    def _final_mul(self, other: Sequence) -> Sequence:
        return RationalSequence._final_mul(self,other)
    def _final_div(self, other: Sequence) -> Sequence:
        return RationalSequence._final_div(self,other)
    def _final_mod(self, other: Sequence) -> Sequence:
        return RationalSequence._final_mod(self,other)
    def _final_floordiv(self, other: Sequence) -> Sequence:
        return RationalSequence._final_floordiv(self,other)
    
    ## Methods for operations on Sequences
    def _element(self, *indices: int):
        return RationalSequence._element(self, *[self.q**i if i in ZZ else SR(self.q)**i for i in indices])

    def _shift(self, *shifts):
        return QRationalSequence(
            self.generic()(**{str(v): v * self.q**(i) for (v,i) in zip(self.variables(), shifts)}), 
            variables=self.variables(), 
            q = self.q,
            _extend_by_zero=self._Sequence__extend_by_zero
        )
    
    def _subsequence_input(self, input):
        if isinstance(input, (list, tuple)):
            if len(input) != 2:
                    raise TypeError(f"[subsequence - linear] Error in format for a linear subsequence. Expected a pair of integers")
            elif any((not el in ZZ) for el in input):
                raise TypeError(f"[subsequence - linear] Error in format for a linear subsequence. Expected a pair of integers")
            a, b = input
            if a != 1 or b != 0:
                from .element import RationalSequence
                R = ZZ[self.q, 'n']; n = R.gens()[1]
                return RationalSequence(R(f"({self.q}**{b})*n**{a}"), variables=['n'], universe=ZZ)
            else:
                return False
        return super()._subsequence_input(input)
    
    def _subsequence(self, final_input: dict[int, Sequence]):
        try:
            vars = self.variables()
            generics = {i : seq.generic(str(vars[i])) for (i,seq) in final_input.items()}
            return QRationalSequence(
                self._RationalSequence__generic(**{str(vars[i]): gen for (i,gen) in generics.items()}), 
                variables=self.variables(), 
                universe=self.universe, 
                q = self.q,
                _extend_by_zero=all(el._Sequence__extend_by_zero for el in final_input.values())
            )
        except Exception as e:
            if isinstance(e, RuntimeError): raise e
            return super()._subsequence(final_input)

    def _slicing(self, values: dict[int, int]):
        vars = self.variables()
        rem_vars = [vars[i] for i in range(self.dim) if not i in values]
        return QRationalSequence(
            self.__generic(**{str(vars[i]): self.q**(val) for (i,val) in values.items()}), 
            variables=rem_vars, 
            universe=self.universe, 
            q = self.q,
            _extend_by_zero=self._Sequence__extend_by_zero
        )
    def _swap(self, src: int, dst: int):
        new_vars = list(self.variables())
        new_vars[src], new_vars[dst] = new_vars[dst], new_vars[src]
        return QRationalSequence(self.__generic, new_vars, self.universe, q = self.q)

__all__ = ["logq", "QSequence", "QRationalSequence"]
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

from .base import ConstantSequence, Sequence
from sage.all import Expression, SR

class ExpressionSequence(Sequence):
    r'''
        Class for sequences defined by a SageMath expression.

        These sequences are the easiest sequences after the callable sequences (defined in :class:`.base.Sequence`). In 
        this case, the sequence is defined using a SageMath expression (see :class:`~sage.symbolic.expression.Expression`)
        where the variables indicate the dimension of the sequence.

        Expression sequences are particularly interesting because the generic expression is always available (it is defined through
        such expression), which allows to perform different operations in a straightforward manner.

        This class includes the method :func:`variables` that defines the symbolic elements that are the actual variables of the 
        expression. This is used as default values for the names when using the method :func:`generic`.

        EXAMPLES::

            sage: from pseries_basis.sequences import *
            sage: poly = ExpressionSequence(x^2 + 1, variables=(x,), universe=ZZ)
            sage: poly[:5]
            [1, 2, 5, 10, 17]

        On top of this class we fin the callable sequences. When we operate whe these, we lose the information of the expression 
        and, maybe, the generic expression::

            sage: poly + Fibonacci(1,1)
            Sequence over [Integer Ring]: (2, 3, 7,...)
            sage: (poly + Fibonacci(1,1)).__class__
            <class 'pseries_basis.sequences.base.Sequence'>
            sage: (poly + Fibonacci(1,1)).generic("x")
            Traceback (most recent call last)
            ...
            ValueError: Impossible to compute generic expression for a sequence.
            sage: poly * Factorial
            Sequence over [Integer Ring]: (1, 2, 10,...)
            sage: (poly * Factorial).__class__
            <class 'pseries_basis.sequences.base.Sequence'>
            sage: (poly * Factorial).generic("x")
            (x^2 + 1)*factorial(x)

        Conversely, on the bottom of this class we always find the class :class:`ConstantSequence` which will always convert into 
        an ExpressionSequence with the same variables::

            TODO: add examples with constant sequences.

        Additionally with the usual restrictions for computing with sequences, :class:`ExpressionSequence` has an extra restriction
        to compute with these sequences. Namely, we require that the variables that define the sequence must coincide in name and 
        order. Otherwise, an usual coercion error is raised::

            TODO: add examples with arithmetics on ExpressionSequence. Show error when variables do not match.

        We can also compue any other operation over sequences as defined in :class:`Sequence`::

            TODO: add examples using :func:`~.base.Sequence.subsequence`, :func:`~.base.Sequence.slicing`, :func:`~.base.Sequence.shift`
    '''
    def __init__(self, expression: Expression, variables=None, universe=None, *, _extend_by_zero=False):
        if variables == None:
            variables = expression.variables()
        dim = len(variables)
        self.__generic = expression
        self.__variables = variables

        super().__init__(None, universe, dim, _extend_by_zero=_extend_by_zero)

    ## Methods for the casting of seqeunces
    @classmethod
    def resgister_class(cls):
        return cls._resgister_class([Sequence], [ConstantSequence])

    @classmethod
    def _change_from_class(self, sequence: Sequence):
        if isinstance(sequence, ConstantSequence):
            return ExpressionSequence(
                SR(sequence.element(*[0 for _ in range(sequence.dim)])), 
                variables=[f"n_{i}" for i in range(sequence.dim)] if sequence.dim > 1 else ["n"], 
                universe=sequence.universe, 
                _extend_by_zero=sequence._Sequence__extend_by_zero
            )
        else:
            raise NotImplementedError(f"The class {sequence.__class__} not recognized as subclass of ExpressionSequence")

    ## Methods fro sequence arithmetic
    def _neg_(self) -> ExpressionSequence:
        return ExpressionSequence(
            self.generic()._neg_(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero
        )

    def _final_add(self, other: ExpressionSequence) -> ExpressionSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return ExpressionSequence(
            self.generic() + other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_sub(self, other: ExpressionSequence) -> ExpressionSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return ExpressionSequence(
            self.generic() - other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_mul(self, other: ExpressionSequence) -> ExpressionSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return ExpressionSequence(
            self.generic() * other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_div(self, other: ExpressionSequence) -> ExpressionSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return ExpressionSequence(
            self.generic() / other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_mod(self, other: ExpressionSequence) -> ExpressionSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return ExpressionSequence(
            self.generic() % other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_floordiv(self, other: ExpressionSequence) -> ExpressionSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return ExpressionSequence(
            self.generic() // other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )

    ## Methods for operations on Sequences
    def _element(self, *indices: int):
        return self.__generic.subs(**{str(v) : i for (v,i) in zip(self.variables(), indices)})

    def _shift(self, *shifts):
        return ExpressionSequence(
            self.__generic.subs(**{str(v): v + i for (v,i) in zip(self.variables(), shifts)}), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero
        )
    def _subsequence(self, final_input: dict[int, Sequence]):
        try:
            vars = self.variables()
            generics = {i : seq.generic(str(vars[i])) for (i,seq) in final_input.items()}
            return ExpressionSequence(
                self.__generic.subs(**{str(vars[i]): gen for (i,gen) in generics.items()}), 
                variables=self.variables(), 
                universe=self.universe, 
                _extend_by_zero=all(el._Sequence__extend_by_zero for el in final_input.values())
            )
        except:
            return super()._subsequence(final_input)
    def _slicing(self, values: dict[int, int]):
        vars = self.variables()
        rem_vars = [vars[i] for i in range(self.dim) if not i in values]
        return ExpressionSequence(
            self.__generic.subs(**{str(vars[i]): val for (i,val) in values.items()}), 
            variables=rem_vars, 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero
        )

    ## Other overriden methods
    def generic(self, *names: str):
        if len(names) == 0:
            return self.__generic
        return super().generic(*names)

    ## Other methods
    def variables(self):
        return self.__variables

class RationalSequence(Sequence):
    pass

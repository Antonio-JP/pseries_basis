r'''
    Module with the specific type of sequences that are used in the module :mod:`.base`.

    Since each of these are extension of the class :class:`~.base.Sequence`, they all inherit
    from this class and fills the requirements to work properly in the Sequences setting:

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

from .base import ConstantSequence, Sequence
from sage.all import Expression, PolynomialRing, var, SR #pylint: disable=no-name-in-module
from sage.rings.fraction_field import is_FractionField
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing

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
            Traceback (most recent call last):
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

            sage: n = var('n')
            sage: Seq = ExpressionSequence(sin(pi*n/8), [n], QQbar)
            sage: CS = ConstantSequence(1/sqrt(2), QQbar, 1)
            sage: Seq * CS
            Sequence over [Algebraic Field]: (0, 0.2705980500730985?, 0.500000000000000?,...)
            sage: Seq - QQbar(sqrt(2))
            Sequence over [Algebraic Field]: (-1.414213562373095?, -1.031530130008006?, -0.7071067811865475?,...)
            sage: Seq * QQbar(sqrt(2))
            Sequence over [Algebraic Field]: (0, 0.5411961001461970?, 1.000000000000000?,...)
            sage: (Seq + Factorial)/QQbar(sqrt(2)) - 1
            Sequence over [Algebraic Field]: (-0.2928932188134525?, -0.02229516874035399?, 0.9142135623730951?,...)
            sage: ((Seq + Factorial)/QQbar(sqrt(2)) - 1).generic()
            0.7071067811865475?*factorial(n) + 0.7071067811865475?*sin(1/8*pi*n) - 1
            sage: Sin2 = ExpressionSequence(sin(x)**2, [x], SR); Cos2 = ExpressionSequence(cos(x)**2, [x], SR)
            sage: Sin2 + Cos2 == 1
            True

        Additionally with the usual restrictions for computing with sequences, :class:`ExpressionSequence` has an extra restriction
        to compute with these sequences. Namely, we require that the variables that define the sequence must coincide in name and 
        order. Otherwise, an usual coercion error is raised::

            sage: ExpX = ExpressionSequence(exp(x+n), [n], SR); ExpX
            Sequence over [Symbolic Ring]: (e^x, e^(x + 1), e^(x + 2),...)
            sage: ExpN = ExpressionSequence(exp(x+n), [x], SR); ExpN
            Sequence over [Symbolic Ring]: (e^n, e^(n + 1), e^(n + 2),...)
            sage: ExpX + ExpN
            Traceback (most recent call last):
            ...
            TypeError: unsupported operand type(s) for +: 'ExpressionSequence' and 'ExpressionSequence'
            sage: y = var('y')
            sage: ExpYX= ExpressionSequence(exp(x+y+n), [n], SR); ExpYX
            Sequence over [Symbolic Ring]: (e^(x + y), e^(x + y + 1), e^(x + y + 2),...)
            sage: ExpYX / ExpX
            Sequence over [Symbolic Ring]: (e^y, e^y, e^y,...)
            sage: ExpYX / ExpX == e^y
            True

        We can also compute any other operation over sequences as defined in :class:`Sequence`::

            sage: Exp = ExpressionSequence(exp(x-y), universe=SR); Exp
            Sequence with 2 variables over [Symbolic Ring]
            sage: Exp.generic()
            e^(x - y)
            sage: Exp.shift(2, 1).generic()
            e^(x - y + 1)
            sage: Exp.shift(1, 5).generic()
            e^(x - y - 4)
            sage: Exp.subsequence((0, Factorial), (1, ExpressionSequence(y**2, [y], universe=ZZ))).generic()
            e^(-y^2 + factorial(x))
            sage: Exp.slicing((1,10)).generic()
            e^(x - 10)
    '''
    def __init__(self, expression: Expression, variables=None, universe=None, *, _extend_by_zero=False):
        if variables == None:
            variables = expression.variables()
        dim = len(variables)
        self.__generic = expression
        self.__variables = variables

        super().__init__(None, universe, dim, _extend_by_zero=_extend_by_zero)

    ## Methods for the casting of sequences
    @classmethod
    def register_class(cls):
        return cls._register_class([Sequence], [ConstantSequence])

    @classmethod
    def _change_from_class(cls, sequence: Sequence, **extra_info):
        if isinstance(sequence, ConstantSequence):
            return ExpressionSequence(
                SR(sequence.element(*[0 for _ in range(sequence.dim)])), 
                variables=extra_info.get("variables", [var(f"n_{i}") for i in range(sequence.dim)] if sequence.dim > 1 else [var("n")]), 
                universe=sequence.universe, 
                _extend_by_zero=sequence._Sequence__extend_by_zero
            )
        else:
            raise NotImplementedError(f"The class {sequence.__class__} not recognized as subclass of ExpressionSequence")
        
    def extra_info(self) -> dict:
        return {"variables": self.variables()}

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

    ## Other overridden methods
    def generic(self, *names: str):
        if len(names) == 0:
            return self.__generic
        return super().generic(*names)

    ## Other methods
    def variables(self):
        return self.__variables

class RationalSequence(Sequence):
    r'''
        Class for sequences defined by a Rational function.

        These sequences are defined using a rational function that, when it is evaluated, we obtain the corresponding element 
        in the sequence. If the sequence contains several variables, then higher dimensional sequences can be generated.
        We also allow to only specify some of the variables to left other variables as parameters.

        INPUT:

        * ``rational``: rational expression defining the sequence. It must be either in a polynomial ring or a fraction
          field of a polynomial ring. The base field is not important for defining the sequence.
        * ``variables`` (optional): a list/tuple of variables in the parent of ``rational`` that specifies the variables
          that are considered indices of the sequence. If ``None`` is given, all variables in the parent are considered.
        * ``universe`` (optional): the parent structure that contains the elements of the sequence. By default, the universe 
          is not necessary and it is set to the remaining parent structure from the parent of ``rational`` after removing
          the variables in ``variables``. If given, it may raise errors when evaluating the sequence.
        
        EXAMPLES::

            sage: from pseries_basis.sequences import *
            sage: R.<x,y> = QQ[]
            sage: P = (x+y)/(x-y)
            sage: poly = RationalSequence(P, [x]); poly
            Sequence over [Fraction Field of Multivariate Polynomial Ring in x, y over Rational Field]: (-1, (y + 1)/(-y + 1), (y + 2)/(-y + 2),...)
            sage: poly[:5]
            [-1, (y + 1)/(-y + 1), (y + 2)/(-y + 2), (y + 3)/(-y + 3), (y + 4)/(-y + 4)]

        If we do not provide the list of variables, then we considered all variables as variables of the sequence::

            sage: poly2 = RationalSequence(P); poly2
            Sequence with 2 variables over [Fraction Field of Multivariate Polynomial Ring in x, y over Rational Field]
            sage: poly2((10, 5)) # (10+5)/(10-5)
            3

        On top of this class we find the Expression sequences. When we operate whe these, we lose the information of the rational expression 
        and fall into the symbolic ring::

            TODO: add examples with Expression sequences

        Conversely, on the bottom of this class we always find the class :class:`ConstantSequence` which will always convert into 
        an ExpressionSequence with the same variables::

            TODO: add examples with constant sequences.

        Additionally with the usual restrictions for computing with sequences, :class:`RationalSequence` has an extra restriction
        to compute with these sequences. Namely, we require that the polynomial rings used to define the sequence are compatible
        for operations::

            TODO: add examples with arithmetics on RationalSequence. Show error when variables do not match.

        We can also compute any other operation over sequences as defined in :class:`Sequence`::

            TODO: add examples using :func:`~.base.Sequence.subsequence`, :func:`~.base.Sequence.slicing`, :func:`~.base.Sequence.shift`
    '''
    def __init__(self, rational, variables=None, universe=None, *, _extend_by_zero=False):
        ## Checking rational is a rational expression
        R = rational.parent()
        if is_FractionField(R):
            R = R.base()
        if not (is_PolynomialRing(R) or is_MPolynomialRing(R)):
            if universe is None or variables is None:
                raise TypeError(f"Incomplete information: 'rational' was not a rational expression and 'universe'/'variables' was not given.")
            else:
                R = PolynomialRing(universe, [str(v) for v in variables])
                variables = R.gens()
        else:
            variables = [R(v) for v in variables] if variables != None else R.gens()
            if any(v not in R.gens() for v in variables):
                raise TypeError("Incorrect information: a variable requested is not a generator of the polynomial ring.")
        
        F = R.fraction_field()
        rational = F(rational)

        ## Storing the data for the sequence
        self.__generic = rational
        self.__F = F
        self.__variables = variables
        dim = len(variables)

        ## Computing the remaining universe
        if universe == None:
            universe = R.remove_var(*variables)
            universe = R.fraction_field() if not R.is_field() else R
            
        super().__init__(None, universe, dim, _extend_by_zero=_extend_by_zero)

    ## Methods for the casting of sequences
    @classmethod
    def register_class(cls):
        return cls._register_class([ExpressionSequence], [ConstantSequence])
    
    def extra_info(self) -> dict:
        return {"variables": self.variables(), "F": self.__F}
    
    def _change_class(self, cls, **extra_info):
        if cls == ExpressionSequence:
            return ExpressionSequence(
                self.__generic,
                self.variables(),
                self.universe,
                _extend_by_zero = self._Sequence__extend_by_zero
            )
        else:
            return super()._change_class(cls, **extra_info)

    @classmethod
    def _change_from_class(cls, sequence: Sequence, **extra_info):
        if isinstance(sequence, ConstantSequence):
            return RationalSequence(
                (sequence.element(*[0 for _ in range(sequence.dim)])), 
                variables = extra_info.get("variables", [f"n_{i}" for i in range(sequence.dim)] if sequence.dim > 1 else ["n"]), 
                universe=sequence.universe, 
                _extend_by_zero=sequence._Sequence__extend_by_zero
            )
        else:
            raise NotImplementedError(f"The class {sequence.__class__} not recognized as subclass of ExpressionSequence")

    ## Methods fro sequence arithmetic
    def _neg_(self) -> RationalSequence:
        return RationalSequence(
            self.generic()._neg_(), 
            variables=self.variables(),  
            _extend_by_zero=self._Sequence__extend_by_zero
        )

    def _final_add(self, other: RationalSequence) -> RationalSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return RationalSequence(
            self.generic() + other.generic(), 
            variables=self.variables(), 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_sub(self, other: RationalSequence) -> RationalSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return RationalSequence(
            self.generic() - other.generic(), 
            variables=self.variables(), 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_mul(self, other: RationalSequence) -> RationalSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return RationalSequence(
            self.generic() * other.generic(), 
            variables=self.variables(), 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_div(self, other: RationalSequence) -> RationalSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return RationalSequence(
            self.generic() / other.generic(), 
            variables=self.variables(), 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_mod(self, other: RationalSequence) -> RationalSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return RationalSequence(
            self.generic() % other.generic(), 
            variables=self.variables(), 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )
    def _final_floordiv(self, other: RationalSequence) -> RationalSequence:
        if self.variables() != other.variables():
            return NotImplemented

        return RationalSequence(
            self.generic() // other.generic(), 
            variables=self.variables(), 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero
        )

    ## Methods for operations on Sequences
    def _element(self, *indices: int):
        return self.__generic(**{str(v) : i for (v,i) in zip(self.variables(), indices)})

    def _shift(self, *shifts):
        return RationalSequence(
            self.__generic(**{str(v): v + i for (v,i) in zip(self.variables(), shifts)}), 
            variables=self.variables(), 
            _extend_by_zero=self._Sequence__extend_by_zero
        )
    def _subsequence(self, final_input: dict[int, Sequence]):
        try:
            vars = self.variables()
            generics = {i : seq.generic(str(vars[i])) for (i,seq) in final_input.items()}
            return RationalSequence(
                self.__generic(**{str(vars[i]): gen for (i,gen) in generics.items()}), 
                variables=self.variables(), 
                universe=self.universe, 
                _extend_by_zero=all(el._Sequence__extend_by_zero for el in final_input.values())
            )
        except:
            return super()._subsequence(final_input)
    def _slicing(self, values: dict[int, int]):
        vars = self.variables()
        rem_vars = [vars[i] for i in range(self.dim) if not i in values]
        return RationalSequence(
            self.__generic(**{str(vars[i]): val for (i,val) in values.items()}), 
            variables=rem_vars, 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero
        )

    ## Other overridden methods
    def generic(self, *names: str):
        if len(names) == 0:
            return self.__generic
        return super().generic(*names)

    ## Other methods
    def variables(self):
        return self.__variables

__all__=["ExpressionSequence", "RationalSequence"]
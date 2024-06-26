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

from typing import Collection, Mapping

from sage.calculus.var import var
from sage.categories.pushout import pushout
from sage.rings.integer_ring import ZZ
from sage.rings.fraction_field import is_FractionField 
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.element import parent
from sage.symbolic.expression import Expression
from sage.symbolic.ring import SR

from .base import ConstantSequence, Sequence, IdentitySequence

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

        On top of this class we find the callable sequences. When we operate with these, we lose the information of the expression 
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
    def __init__(self, expression: Expression, variables=None, universe=None, *, meanings: Sequence | Collection[Sequence] | Mapping[str,Sequence] = None, _extend_by_zero=False, **kwds):
        if variables is None:
            variables = expression.variables()
        dim = len(variables)
        self.__generic = expression
        self.__SR = expression.parent()
        self.__variables = [self.__SR(el) for el in variables]

        super().__init__(None, universe, dim, _extend_by_zero=_extend_by_zero, **kwds)
        
        ## Now we check argument "meaning"
        if meanings is None:
            meanings = dim*[IdentitySequence(universe, **kwds)]
        elif isinstance(meanings, Sequence):
            meanings = dim*[meanings]

        if not isinstance(meanings, Collection):
            raise TypeError(f"[ExpressionSequence] The meanings of the variables is ill-formatted: incorrect type")
        if not isinstance(meanings, Mapping): # If it is not a map, we convert it
            if len(meanings) != dim:
                raise ValueError(f"[ExpressionSequence] The meanings of the variables is ill-formated: incorrect size")
            meanings = {str(var): val for (var,val) in zip(self.__variables, meanings)}
        
        if any(self.__SR(v) not in self.__variables for v in meanings):
            raise ValueError(f"[ExpressionSequence] The meanings of the variables is ill-formatted: incorrect names")
        if any(not (isinstance(el, Sequence) or (el in self.universe)) for el in meanings.values()):
            raise ValueError(f"[ExpressionSequence] The meanings of the variables is ill-formatted: incorrect values for meanings")
        
        # We guarantee that meanings are some sequences
        self.__meanings = {str(var): val if isinstance(val, Sequence) else ConstantSequence(val, universe, 1, **kwds) for (var,val) in meanings.items()}

    ## Methods for the casting of sequences
    @classmethod
    def register_class(cls):
        return cls._register_class([Sequence], [ConstantSequence])
    
    def args_to_self(self):
        return ([self.__generic], 
                {
                    "universe": self.universe, 
                    "variables": self.variables(), 
                    "meanings": self.meanings(), 
                    "_extend_by_zero": self._Sequence__extend_by_zero,
                    **super().extra_info()["extra_args"]
                })
    
    def _change_dimension(self, new_dim: int, old_dims: list[int], new_variables = None, new_meanings: Sequence | Collection[Sequence] | Mapping[Sequence] = None):
        args, kwds = self.args_to_self()
        generic = args[0]
        variables = kwds.pop("variables")
        universe=kwds.pop("universe")
        extend = kwds.pop("_extend_by_zero")
        meanings = kwds.pop("meanings")

        ## Checking the new variable names
        if not isinstance(new_variables, (list,tuple)) or len(new_variables) != new_dim - self.dim:
            raise TypeError(f"[change_dim] The new variables must be a valid list of valid length")
        
        ## Checking the argument for new meanings
        if new_meanings is None:
            new_meanings = len(new_variables)*[IdentitySequence(universe, **kwds)]
        elif isinstance(new_meanings, Sequence):
            new_meanings = len(new_variables)*[new_meanings]

        if not isinstance(new_meanings, Collection):
            raise TypeError(f"[change_dim] The new meanings must be of appropriate type")
        if not isinstance(new_meanings, Mapping): # If it is not a map, we convert it
            if len(new_meanings) != len(new_variables):
                raise ValueError(f"[change_dim] The new meanings must be of appropriate length")
            new_meanings = {str(var): val for (var,val) in zip(new_variables, new_meanings)}
        
        if any(v not in new_variables for v in new_meanings):
            raise ValueError(f"[change_dim] The new meanings are given for wrong names")
        if any(not (isinstance(el, Sequence) or (el in self.universe)) for el in new_meanings.values()):
            raise ValueError(f"[change_dim] The new meanings must be of appropriate shape")
        ## We add the new variables and their meanings
        for (i, v) in sorted(zip(old_dims, variables), key=lambda k : k[0]): new_variables.insert(i, v)
        meanings.update(**new_meanings)

        return self.__class__(generic, new_variables, universe=universe, meanings=meanings, _extend_by_zero=extend, **kwds)

    @classmethod
    def _change_from_class(cls, sequence: Sequence, **extra_info):
        if isinstance(sequence, ConstantSequence):
            variables = extra_info.get("variables", [var(f"n_{i}") for i in range(sequence.dim)] if sequence.dim > 1 else [var("n")])
            meanings = extra_info.get("meanings", dict())
            return ExpressionSequence(
                SR(sequence.element(*[0 for _ in range(sequence.dim)])), 
                variables=variables, 
                universe=sequence.universe, 
                meanings=meanings,
                _extend_by_zero=sequence._Sequence__extend_by_zero,
                **{**extra_info["extra_args"], **sequence.extra_info()["extra_args"]}
            )
        else:
            raise NotImplementedError(f"The class {sequence.__class__} not recognized as subclass of ExpressionSequence")
        
    def extra_info(self) -> dict:
        dict = super().extra_info()
        dict["variables"] = self.variables()
        dict["meanings"] = self.meanings()
        return dict

    ## Methods for sequence arithmetic
    def _compatible_(self, other: ExpressionSequence):
        r'''Method that checks whether two ExpressionSequences are valid to be operated'''
        ## We need that the names of the variables coincide
        if self.variables() != other.variables():
            return False
        ## We also need that the sequence they represent are the same
        self_meanings = self.meanings()
        other_meanings = other.meanings()

        if any(self_meanings[str(v)] != other_meanings[str(v)] for v in self.variables()):
            return False
        return True

    def _neg_(self) -> ExpressionSequence:
        gen, args = self.args_to_self()
        return self.__class__(gen[0]._neg_(), **args)

    def _final_add(self, other: ExpressionSequence) -> ExpressionSequence:
        if not self._compatible_(other):
            return NotImplemented
        
        return self.__class__(
            self.generic() + other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero,
            meanings=self.meanings(),
            **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]}
        )
    def _final_sub(self, other: ExpressionSequence) -> ExpressionSequence:
        if not self._compatible_(other):
            return NotImplemented

        return self.__class__(
            self.generic() - other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero,
            meanings=self.meanings(),
            **{**self.extra_info()["extra_args"], **self.extra_info()["extra_args"]}
        )
    def _final_mul(self, other: ExpressionSequence) -> ExpressionSequence:
        if not self._compatible_(other):
            return NotImplemented

        return self.__class__(
            self.generic() * other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero,
            meanings=self.meanings(),
            **{**self.extra_info()["extra_args"], **self.extra_info()["extra_args"]}
        )
    def _final_div(self, other: ExpressionSequence) -> ExpressionSequence:
        if not self._compatible_(other):
            return NotImplemented

        return self.__class__(
            self.generic() / other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero,
            meanings=self.meanings(),
            **{**self.extra_info()["extra_args"], **self.extra_info()["extra_args"]}
        )
    def _final_mod(self, other: ExpressionSequence) -> ExpressionSequence:
        if not self._compatible_(other):
            return NotImplemented

        return self.__class__(
            self.generic() % other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero,
            meanings=self.meanings(),
            **{**self.extra_info()["extra_args"], **self.extra_info()["extra_args"]}
        )
    def _final_floordiv(self, other: ExpressionSequence) -> ExpressionSequence:
        if not self._compatible_(other):
            return NotImplemented

        return self.__class__(
            self.generic() // other.generic(), 
            variables=self.variables(), 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero and other._Sequence__extend_by_zero,
            meanings=self.meanings(),
            **{**self.extra_info()["extra_args"], **self.extra_info()["extra_args"]}
        )

    ## Methods for operations on Sequences
    def _element(self, *indices: int):
        self_meanings = self.meanings()
        return self._eval_generic(**{str(v) : self_meanings[str(v)]._element(i) for (v,i) in zip(self.variables(), indices)})

    def _subsequences_input(self, *vals) -> dict[int, Sequence]:
        original_inputs = dict(vals)
        result = super()._subsequences_input(*vals)
        ## We check if all were expressions
        if any(isinstance(seq, Sequence) for seq in result.values()): # we need to fall-back to sequences
            for (i, seq) in result.items():
                if not isinstance(seq, Sequence):
                    result[i] = super()._subsequence_input(original_inputs[i])
        
        return result

    def _subsequence_input(self, index: int, input: tuple[int,int] | Sequence) -> Expression | Sequence | bool: # TODO
        vars = self.variables()
        inner = self.meanings()[str(vars[index])]
        if isinstance(input, (list,tuple)): # case of a linear subsequence
            if len(input) != 2:
                raise TypeError(f"[subsequence - linear] Error in format for a linear subsequence. Expected a pair of integers")
            elif any((el not in ZZ) for el in input):
                raise TypeError(f"[subsequence - linear] Error in format for a linear subsequence. Expected a pair of integers")
            
            a, b = [ZZ(el) for el in input]

            if a != 1 or b != 0:
                from .qsequences import is_QSequence
                if inner == IdentitySequence(self.universe, **self.extra_info()["extra_args"]):
                    return vars[index]*a + b
                elif is_QSequence(inner):
                    if hasattr(inner, "power"): # This is a special q-sequence representing q^{en}
                        # q^{e(an+b)} = (q^{en})^a * q^{e*b} = q^{eb} * var^a
                        # Since `inner` is a q-sequence, it has attribute `q`
                        return (vars[index]**a) * (inner.q ** b)
        ## Any other behavior will fall into the original method
        elif isinstance(input, Sequence):
            try:
                return input.generic(str(vars[index]))
            except Exception: 
                pass

        return super()._subsequence_input(index, input)
    
    def _shift(self, *shifts):
        try:
            subseqs = [self._subsequence_input(i, (1,shifts[i])) for i in range(len(shifts))] # this take into account the meaning
            if any(isinstance(subseq, Sequence) for subseq in subseqs):
                raise TypeError(f"Falling back to usual shifting")
            subseqs = {str(v): subseq for (v,subseq) in zip(self.variables(), subseqs) if subseq}
            return self.__class__(
                self._eval_generic(**subseqs), 
                variables=self.variables(), 
                universe=self.universe, 
                _extend_by_zero=self._Sequence__extend_by_zero,
                meanings=self.meanings(),
                **self.extra_info()["extra_args"]
            )
        except Exception:
            return super()._shift(*shifts)
        
    def _subsequence(self, final_input: dict[int, Sequence] | dict[int, Expression]):
        try:
            if isinstance(next(iter(final_input.values())), Sequence):
                raise TypeError(f"Falling back to usual subsequencing")
            vars = self.variables()
            subseqs = {str(vars[i]): final_input[i] for i in final_input}
            return self.__class__(
                self._eval_generic(**subseqs), 
                variables=self.variables(), 
                universe=self.universe, 
                _extend_by_zero=self._Sequence__extend_by_zero,
                meanings=self.meanings(),
                **self.extra_info()["extra_args"]
            )
        except TypeError:
            return super()._subsequence(final_input)
        
    def _slicing(self, values: dict[int, int]):
        vars = self.variables()
        rem_vars = [vars[i] for i in range(self.dim) if i not in values]
        self_meanings = self.meanings()
        meanings = {str(v): self_meanings[str(v)] for v in rem_vars if str(v) in self_meanings}
        return self.__class__(
            self._eval_generic(**{str(vars[i]): self_meanings[str(vars[i])]._element(val) for (i,val) in values.items()}), 
            variables=rem_vars, 
            universe=self.universe, 
            _extend_by_zero=self._Sequence__extend_by_zero,
            meanings=meanings,
            **self.extra_info()["extra_args"]
        )
    
    def _swap(self, src: int, dst: int):
        new_vars = list(self.variables())
        new_vars[src], new_vars[dst] = new_vars[dst], new_vars[src]
        new_meanings = self.meanings()
        new_meanings[new_vars[src]], new_meanings[new_vars[dst]] = new_meanings[new_vars[dst]], new_meanings[new_vars[src]]
        return self.__class__(
            self.generic(), 
            new_vars, 
            self.universe, 
            meanings=new_meanings, 
            _extend_by_zero = self._Sequence__extend_by_zero, 
            **self.extra_info()["extra_args"]
        )

    ## Other overridden methods
    def generic(self, *names: str):
        if len(names) == 0:
            return self.__generic
        return super().generic(*names)
    def _eval_generic(self, **values):
        return self.__generic.subs(**values)
    def _generic(self, *names: str):
        return self._eval_generic(**{str(v) : self.__meanings[str(v)].generic(names[i]) for (i,v) in enumerate(self.variables())})

    def is_polynomial(self) -> bool:
        try:
            expr = SR(self.generic()).simplify_full()
            return all(expr.is_polynomial(x) for x in self.variables())
        except Exception:
            return False
    def as_polynomial(self) -> RationalSequence:
        if self.is_polynomial():
            from .element import RationalSequence
            return RationalSequence(SR(self.generic()).simplify_full(), self.variables(), self.universe, meanings=self.meanings())
        else:
            raise ValueError(f"{self} is not a polynomial sequence.")
    def is_rational(self) -> bool:
        try:
            return SR(self.generic()).simplify_full().is_rational_expression()
        except Exception:
            return False
    def as_rational(self) -> RationalSequence:
        if self.is_rational():
            from .element import RationalSequence
            return RationalSequence(SR(self.generic()).simplify_full(), self.variables(), self.universe, meanings=self.meanings())
        else:
            raise ValueError(f"{self} is not a polynomial sequence.")
      
    ## Other methods
    def variables(self):
        return self.__variables
    def meanings(self):
        return self.__meanings.copy()

class RationalSequence(ExpressionSequence):
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
            Sequence over [Fraction Field of Univariate Polynomial Ring in y over Rational Field]: (-1, (-y - 1)/(y - 1), (-y - 2)/(y - 2),...)
            sage: poly[:5]
            [-1, (-y - 1)/(y - 1), (-y - 2)/(y - 2), (-y - 3)/(y - 3), (-y - 4)/(y - 4)]

        If we do not provide the list of variables, then we considered all variables as variables of the sequence::

            sage: poly2 = RationalSequence(P); poly2
            Sequence with 2 variables over [Rational Field]
            sage: poly2((10, 5)) # (10+5)/(10-5)
            3

        On top of this class we find the Expression sequences. When we operate with these, we lose the information of the rational expression 
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
    def __init__(self, rational, variables=None, universe=None, *, meanings: Sequence | Collection[Sequence] | Mapping[str,Sequence] = None, _extend_by_zero=False, **kwds):
        ## Checking rational is a rational expression
        R = rational.parent() if hasattr(rational, "parent") else parent(rational)
        if is_FractionField(R):
            R = R.base()
        if not (is_PolynomialRing(R) or is_MPolynomialRing(R)):
            if universe is None or variables is None:
                raise TypeError(f"Incomplete information: 'rational' was not a rational expression and 'universe'/'variables' was not given.")
            else:
                R = PolynomialRing(universe, [str(v) for v in variables])
                variables = R.gens()
        else:
            if universe is not None and variables is not None:
                min_R = PolynomialRing(universe, variables)
            elif universe is not None:
                min_R = universe
            elif variables is not None:
                min_R = PolynomialRing(R.base(), list(R.variable_names()) + [str(v) for v in variables if (v not in R.gens())])
            else:
                min_R = R
            R = pushout(R, min_R)

            variables = [R(v) for v in variables] if variables is not None else R.gens()
            if any(v not in R.gens() for v in variables):
                raise TypeError("Incorrect information: a variable requested is not a generator of the polynomial ring.")
        
        F = R.fraction_field()
        rational = F(rational)
        # dim = len(variables) TODO: remove

        ## Computing the remaining universe
        if universe is None:
            if is_PolynomialRing(R) and len(variables) > 0:
                universe = R.base()
            elif is_PolynomialRing(R):
                universe = R
            else:
                universe = R.remove_var(*variables)
            universe = universe.fraction_field() if not universe.is_field() else universe

        # ## Now we check argument "meaning"  TODO: remove
        # if meanings is None:
        #     meanings = dim*[IdentitySequence(universe, **kwds)]
        # elif isinstance(meanings, Sequence):
        #     meanings = dim*[meanings]

        # if not isinstance(meanings, Collection):
        #     raise TypeError(f"[ExpressionSequence] The meanings of the variables is ill-formatted: incorrect type")
        # if not isinstance(meanings, Mapping): # If it is not a map, we convert it
        #     if len(meanings) != dim:
        #         raise ValueError(f"[ExpressionSequence] The meanings of the variables is ill-formated: incorrect size")
        #     meanings = {str(var): val for (var,val) in zip(variables, meanings)}
        
        # if any(SR(v) not in variables for v in meanings):
        #     raise ValueError(f"[ExpressionSequence] The meanings of the variables is ill-formatted: incorrect names")
        # if any(not (isinstance(el, Sequence) or (el in self.universe)) for el in meanings.values()):
        #     raise ValueError(f"[ExpressionSequence] The meanings of the variables is ill-formatted: incorrect values for meanings")
        
        ## Storing the data for the sequence
        self.__F = F
        super().__init__(rational, tuple(variables), universe, meanings=meanings, _extend_by_zero=_extend_by_zero, **kwds)
        # self.__generic = rational TODO: remove
        # self.__variables = tuple(variables) TODO: remove
        # self.__meanings = meanings TODO: remove
            
        # super().__init__(None, universe, dim, _extend_by_zero=_extend_by_zero, **kwds) TODO: remove

    ## Methods for the casting of sequences
    @classmethod
    def register_class(cls):
        return cls._register_class([ExpressionSequence], [ConstantSequence])

    def _change_class(self, cls, **extra_info):
        if cls == ExpressionSequence:
            return ExpressionSequence(
                SR(self.generic()),
                [str(v) for v in self.variables()],
                self.universe,
                _extend_by_zero = self._Sequence__extend_by_zero,
                meanings=self.meanings(),
                **self.extra_info()["extra_args"]
            )
        else:
            return super()._change_class(cls, **extra_info)

    @classmethod
    def _change_from_class(cls, sequence: Sequence, **extra_info):
        if isinstance(sequence, ConstantSequence):
            return RationalSequence(
                extra_info["field"](sequence.element(*[0 for _ in range(sequence.dim)])), 
                variables = extra_info.get("variables", [f"n_{i}" for i in range(sequence.dim)] if sequence.dim > 1 else ["n"]), 
                universe=sequence.universe, 
                _extend_by_zero=sequence._Sequence__extend_by_zero,
                meanings=extra_info.get("meanings", None),
                **{**extra_info["extra_args"], **sequence.extra_info()["extra_args"]}
            )
        else:
            raise NotImplementedError(f"The class {sequence.__class__} not recognized as subclass of ExpressionSequence")

    def extra_info(self) -> dict:
        dict = super().extra_info()
        dict["field"] = self.__F
        return dict

    def _eval_generic(self, **values):
        return self.generic()(**values)

    def is_polynomial(self) -> bool:
        return self.generic().denominator() == 1
    def as_polynomial(self) -> Sequence:
        if self.is_polynomial():
            return self
        else:
            raise ValueError(f"{self} is not a polynomial sequence.")
    def is_rational(self) -> bool:
        return True
    def as_rational(self) -> Sequence:
        return self
    
__all__=["ExpressionSequence", "RationalSequence"]
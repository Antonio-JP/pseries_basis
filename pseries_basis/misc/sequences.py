r'''
    Module containing methods, strucutres and functions related with Sequences.

    A sequence is a function `f: \mathbb{N} \rightarrow \mathbb{K}` where the space `\mathbb{K}` is 
    called the *universe* of the sequence. The package :mod:`pseries_basis` is oriented towards 
    formal power series, which are equivalent to sequences. 
    
    This module will create a framework for the whole package to manage, access and manipulate sequences
    in a generic fashion.
'''
## Sage imports
from sage.all import (cached_method, oo, NaN, cartesian_product, ZZ, parent, SR)
from sage.categories.homset import Hom
from sage.categories.pushout import CoercionException, pushout
from sage.categories.morphism import SetMorphism # pylint: disable=no-name-in-module
from sage.categories.sets_cat import Sets
from sage.rings.semirings.non_negative_integer_semiring import NN

class Sequence(SetMorphism):
    r'''
        Main calss for sequences. It defines a universe for its elements and a general interface to access the sequence.
    '''
    def __init__(self, universe, dim : int =1, allow_sym = False):
        self.__universe = universe
        self.__dim = dim
        self.__alls = allow_sym
        if dim == 1:
            super().__init__(Hom(NN,universe,Sets()), lambda n : self.element(n))
        else:
            super().__init__(Hom(cartesian_product(dim*[NN]), universe, Sets()), lambda n : self.element(*n))

    @property
    def universe(self): 
        r'''
            Attribute with the common parent for all the elements of this sequence
        '''
        return self.__universe
    @property
    def dim(self) -> int:
        r'''
            Attribute with the number of variables for the sequence.
        '''
        return self.__dim
    @property
    def allow_sym(self) -> bool:
        return self.__alls

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            return [self[k] for k in key]
        elif isinstance(key, slice):
            res = []
            st = 0 if key.start is None else key.start # we start at 0 if not given beginning
            # the ending must be given, otherwise, we raise an error
            if key.stop is None: raise TypeError("The ending of a slice must be fixed (to guarantee the end of the method)")
            end = key.stop
            jump = 1 if key.step is None else key.step # the step is, by default, 1
            
            for i in range(st, end, jump):
                res.append(self[i])
            return res
        return self.element(key)

    def __call__(self, *input):
        return self.element(*input)

    @cached_method
    def element(self, *indices : int):
        try:
            output = self._element(*indices)
        except ZeroDivisionError:
            return oo
        except:
            return NaN
        if (self.universe is None) or (self.allow_sym and not output in self.universe): # we allow None universe and also evaluation that do nto lie in the universe
            return output
        else:
            return self.universe(output)

    def _element(self, *indices : int):
        raise NotImplementedError("Method '_element' not implemented")

    @cached_method
    def shift(self, *shifts : int) -> "Sequence":
        r'''
            Method to compute the shifted sequence given some shifting indices.

            Given a sequence `(a_n)_n`, we can always consider the `k`-shifted sequence
            for any `k \mathbb{Z}` defined by `b_n = a_{n+k}`. This method allows a 
            sequence in any number of variables to obtained its shifted version for any tuple of 
            shifts.

            INPUT:

            * ``shifts``: a list of integers providing the shifts for each of the dimensions of 
              the sequence. It must be only integers and have same length as ``self.dim``.
        '''
        if len(shifts) == 0:
            shifts = [1 for _ in range(self.dim)]
        elif len(shifts) == 1 and isinstance(shifts[0], (tuple, list)):
            shifts = shifts[0]

        if any(not sh in ZZ for sh in shifts):
            raise TypeError("The shift must be integers")
        if len(shifts) != self.dim:
            raise ValueError(f"We need {self.dim} shifts but {len(shifts)} were given")
        return self._shift(*shifts)

    def _shift(self, *shifts):
        r'''
            Return the actual shifted sequence. Can assume ``shifts`` is a list of appropriate length and type.
        '''
        return LambdaSequence(lambda *n : self(*[n[i]+shifts[i] for i in range(self.dim)]), self.universe, dim=self.dim, allow_sym=self.allow_sym)

    # basic arithmethic methods
    def __add__(self, other):
        if not isinstance(other, Sequence) or self.dim != other.dim:
            return NotImplemented

        universe = pushout(self.universe, other.universe)
        alls = self.allow_sym and other.allow_sym
        return LambdaSequence(lambda *n : self(*n) + other(*n), universe, dim = self.dim, allow_sym=alls)

    def __sub__(self, other):
        if not isinstance(other, Sequence) or self.dim != other.dim:
            return NotImplemented

        universe = pushout(self.universe, other.universe)
        alls = self.allow_sym and other.allow_sym
        return LambdaSequence(lambda *n : self(*n) - other(*n), universe, dim = self.dim, allow_sym=alls)

    def __mul__(self, other):
        if isinstance(other, Sequence) and self.dim == other.dim:
            universe = pushout(self.universe, other.universe)
            alls = self.allow_sym and other.allow_sym
            return LambdaSequence(lambda *n : self(*n) * other(*n), universe, dim = self.dim, allow_sym=alls)
        elif not isinstance(other, Sequence):
            try:
                universe = pushout(self.universe, parent(other))
                return LambdaSequence(lambda *n : self(*n) * other, universe, dim = self.dim, allow_sym=self.allow_sym)
            except CoercionException:
                return NotImplemented
        return NotImplemented
        
    def __neg__(self):
        return LambdaSequence(lambda *n : -self(*n), self.universe, dim = self.dim, allow_sym=self.allow_sym) # pylint: disable=invalid-unary-operand-type
        
    # reverse arithmethic methods
    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return self.__sub__(other)
    def __rmul__(self, other):
        return self.__mul__(other)

    def almost_zero(self, order=10) -> bool:
        r'''
            Method that checks if a sequence is zero at the beginning.

            This method receives a number of elements to be checked and then it proceeds to check if all the first 
            elements of the sequence are equal to zero. This is helpful in some sequence to guarantee that the whole 
            sequence is exactly zero.
        '''
        return all(self[i] == 0 for i in range(order))

    def almost_equals(self, other : 'Sequence', order=10) -> bool:
        r'''
            Method that checks if two sequences are equals at the beginning.

            This method receives another :class:`Sequence` and a number of elements to be checked and then it 
            proceeds to check if all the first elements are equal for both sequences. This is helpful in some sequences
            to guarantee equality through the whole sequence.

            INPUT:

            * ``sequence``: a :class:`Sequence` that will be compare with ``self``.
            * ``order``: number of elements that will be compared.

            OUTPUT:

            ``True`` if the first ``order`` elements of both sequences are equal (even in different 
            universes). ``False`` otherwise.

            EXAMPLES::

                sage: from pseries_basis.misc.sequences import *
                sage: def fib(n): return 1 if n == 0 else (1 if n==1 else fib(n-1) + fib(n-2))
                sage: Fib = LambdaSequence(fib, ZZ)
                sage: Fac = LambdaSequence(lambda n : factorial(n), ZZ)
                sage: Fib[:10]
                [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
                sage: Fac[:10]
                [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
                sage: Fib.almost_equals(Fac, 2)
                True
                sage: Fib.almost_equals(Fac, 3)
                True
                sage: Fib.almost_equals(Fac, 4)
                False
        '''
        return self[:order] == other[:order]

    def subsequence(self, *vals : int) -> "Sequence":
        r'''
            Method to obtain a subsequence when having multiple arguments.

            When we have a sequence `f: \mathbb{N}^k \rightarrow R`, we can fix the first 
            arguments `(a_1,\ldots, a_t)` and build a new sequence by:

            .. MATH::

                \tilde{f}(n_1,\ldots, n_{k-t}) = f(a_1,\ldots,a_t,n_1,\ldots,n_{k-t})

            This method build the corresponding subsequence for the given values in ``vals``.
            If the number of values is more than the number of inputs, then we return the value
            of the sequence at the given position.
        '''
        if len(vals) >= self.dim:
            return self.element(*vals[:self.dim])
        elif any(not v in ZZ for v in vals):
            raise TypeError("The values for subsequences must be integers")
        return self._subsequence(*vals)

    def _subsequence(self, *vals): 
        r'''
            Return the actual subsequence (see :func:`subsequence`). Can assume ``vals`` is a list of appropriate length and type.
        '''
        return LambdaSequence(lambda *n: self.element(*vals, *n), self.universe, dim=self.dim-len(vals), allow_sym=True)

    def __repr__(self) -> str:
        if self.dim == 1:
            return f"Sequence over [{self.universe}]: ({self[0]}, {self[1]}, {self[2]},...)"
        else:
            return f"Sequence with {self.dim} variables over [{self.universe}]"
        
class LambdaSequence(Sequence):
    r'''
        Simplest implementation of :class:`Sequence`.

        This class computes a sequence by calling a function with some integer coefficient. It does not check whether this 
        defining function will always work.

        INPUT:

        * ``func``: a callable that defines the sequence as their output with natural input.
        * ``universe``: a parent structure where all the elements of the sequence will be (a specific casting is performed)
        * ``allow_sym``: a flag to set a sequence to allow symbolic inputs to the sequence to obtain 'generic elements'.

        EXAMPLES::

            sage: from pseries_basis.misc.sequences import LambdaSequence
            sage: C = LambdaSequence(lambda n : catalan_number(n), ZZ)
            sage: C[:10]
            [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]
            sage: (3*C)[:5]
            [3, 3, 6, 15, 42]
            sage: (C + C)[:5]
            [2, 2, 4, 10, 28]
            sage: F = LambdaSequence(fibonacci, ZZ)
            sage: F[:10]
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    '''
    def __init__(self, func, universe, dim = 1, allow_sym = False):
        super().__init__(universe, dim=dim, allow_sym=allow_sym)

        self.__func = func

    def _element(self, *indices : int):
        return self.__func(*indices)

    # equality and hash methods
    def __eq__(self, other):
        if not isinstance(other, LambdaSequence):
            return False
        return self.__func == other.__func
    
    def __hash__(self):
        return hash(self.__func)

class ExpressionSequence(Sequence):
    r'''
        Implementation of :class:`Sequence` using expressions from Sage.

        This class computes a sequence by calling setting up an expression an a list of variables to be the variables of the
        sequence. This class can always be evaluated symbolically and a general expression can be obtained.

        Several methods from :class:`Sequence`are overriden to take into account the use of :class:`ExpressionSequence` for 
        arithmetic operation and similar methods that obtain new :class:`Sequence`

        INPUT:

        * ``expr``: an object of SageMath that can be casted into an expression in the symbolic ring.
        * ``universe``: a parent structure where all the elements (after evaluation) should belong.
        * ``variables``: list of variables that are evaluated for the sequence. If ``None`` is given,
          then all variables in ``expr`` are used in the order given by ``expr.variables()``. We encourage 
          the use of this input to define the ordering of the variables.

        REMARK:

        * The value for ``dim`` in :class:`Sequence` can be obtained from the number of variables.
        * The value for ``allow_sym`` in :class:`Sequence` is always ``True`` for a :class:`ExpressionSequence`

        EXAMPLES::

            sage: from pseries_basis.misc.sequences import ExpressionSequence, LambdaSequence
            sage: F = ExpressionSequence(factorial(n), ZZ)
            sage: F2 = LambdaSequence(lambda n : factorial(n), ZZ)
            sage: F.almost_equals(F2, 100)
            True
            sage: F
            Sequence over [Integer Ring]: factorial(n)
    '''
    def __init__(self, expr, universe, variables=None):
        if not expr in SR:
            raise ValueError("The expression must be something we can convert to an expression")
        expr = SR(expr)
        variables = expr.variables() if variables is None else tuple(variables)

        super().__init__(universe, len(variables), True)
        self.__generic = expr
        self.__vars = variables

    def generic(self):
        r'''
            Method that returns the generic expression for this sequence.

            This method returns the expression that defines this sequence. Use method :func:`variables` 
            to obtain the list (in appropriate order) of variables that are used for this sequence.
        '''
        return self.__generic

    def variables(self):
        r'''
            Method that returns the variables of this sequence.

            This method returns the (sorted) tuple of variables that defines this sequence. Use method :func:`generic` 
            to obtain the defining expression of this sequence.
        '''
        return self.__vars

    def _element(self, *indices: int):
        vars = self.variables()
        return self.generic()(**{str(vars[i]) : indices[i] for i in range(len(indices))})

    def _shift(self, *shifts):
        vars = self.variables()
        evaluations = [vars[i] + shifts[i] for i in range(self.dim)]
        new_expr = self.generic()(**{str(vars[i]) : evaluations[i] for i in range(self.dim)})

        return ExpressionSequence(new_expr, self.universe, vars)

    # basic arithmethic methods
    def __add__(self, other):
        if not isinstance(other, ExpressionSequence) or self.dim != other.dim:
            return super().__add__(other)

        if [SR(v) for v in self.variables()] != [SR(v) for v in other.variables()]:
            return NotImplemented

        universe = pushout(self.universe, other.universe)
        return ExpressionSequence(self.generic() + other.generic(), universe, self.variables())

    def __sub__(self, other):
        if not isinstance(other, ExpressionSequence) or self.dim != other.dim:
            return super().__add__(other)

        if [SR(v) for v in self.variables()] != [SR(v) for v in other.variables()]:
            return NotImplemented

        universe = pushout(self.universe, other.universe)
        return ExpressionSequence(self.generic() - other.generic(), universe, self.variables())

    def __mul__(self, other):
        if isinstance(other, ExpressionSequence) and self.dim == other.dim:
            if [SR(v) for v in self.variables()] != [SR(v) for v in other.variables()]:
                return NotImplemented
            universe = pushout(self.universe, other.universe)
            return ExpressionSequence(self.generic() * other.generic(), universe, self.variables())
        elif not isinstance(other, Sequence):
            try:
                universe = pushout(self.universe, parent(other))
                return ExpressionSequence(SR(other)*self.generic(), universe, self.variables())
            except CoercionException:
                return NotImplemented
        return super().__mul__(other)
        
    def __neg__(self):
        return ExpressionSequence(-self.generic(), self.universe, self.variables())
    
    def _subsequence(self, *vals):
        vars = self.variables()
        new_expr = self.generic(**{str(vars[i]) : vals[i] for i in range(len(vals))}) 
        rem_vars = vars[len(vals):]
        return ExpressionSequence(new_expr, self.universe, rem_vars)

    # equality and hash methods
    def __eq__(self, other):
        if not isinstance(other, ExpressionSequence):
            return False
        if [SR(v) for v in self.variables()] != [SR(v) for v in other.variables()]:
            return False
        return bool(self.generic() == other.generic())
    
    def __hash__(self):
        return hash(self.__generic)

    def __repr__(self) -> str:
        return f"Sequence{f' in {self.__vars}' if self.dim > 1 else ''} over [{self.universe}]: {self.generic()}"

__all__ = ["Sequence", "LambdaSequence", "ExpressionSequence"]
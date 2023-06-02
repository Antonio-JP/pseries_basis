r'''
    Module containing methods, structures and functions related with Sequences.

    A sequence is a function `f: \mathbb{N} \rightarrow \mathbb{K}` where the space `\mathbb{K}` is 
    called the *universe* of the sequence. The package :mod:`pseries_basis` is oriented towards 
    formal power series, which are equivalent to sequences. 
    
    This module will create a framework for the whole package to manage, access and manipulate sequences
    in a generic fashion.
'''
from __future__ import annotations

from functools import reduce
from itertools import product
from typing import Collection

## Sage imports
from sage.all import (cached_method, oo, NaN, cartesian_product, latex, ZZ, parent, SR)
from sage.categories.homset import Homset
from sage.categories.pushout import pushout
from sage.categories.morphism import SetMorphism # pylint: disable=no-name-in-module
from sage.categories.sets_cat import Sets
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing
from sage.rings.semirings.non_negative_integer_semiring import NN
from sage.structure.unique_representation import UniqueRepresentation

_Sets = Sets.__classcall__(Sets)
       
###########################################################################################
### Base element and parent structures for sequences
###########################################################################################
class Sequence(SetMorphism):
    r'''
        Main class for sequences. It defines a universe for its elements and a general interface to access the sequence.

        Methods that need to be implemented in extended classes (abstract methods):

        * ``_element``
        
        Methods that can be extended for more refined results:
        
        * :func:`as_self`: allow better coercion between classes of sequences. If changing this method, these methods should also be updated:
          - :func:`_add_`
          - :func:`_sub_`
          - :func:`_mul_`
          - :func:`_truediv_`
        * :func:`_shift`: allow a way of creating shifts using different classes
        * :func:`_subsequence`: allow a way of building subsequences using different classes
        * :func:`_linear_subsequence`: allow a way of building subsequences using different classes
    '''
    def __init__(self, universe = None, dim : int = 1, allow_sym : bool = False):
        if universe is None:
            raise TypeError("The universe of a sequence must never be None")
        self.__universe = universe
        self.__dim = dim
        self.__alls = allow_sym
        parent = SequenceSet(dim, universe)
        func = (lambda n : self.element(*n)) if dim > 1 else (lambda n : self.element(n))
        self.__CACHE_ELEMENTS = {}

        super().__init__(parent, func)

    #############################################################################
    ## Static Properties
    #############################################################################
    EQUALITY_BOUND = 10

    #############################################################################
    ## Object properties
    #############################################################################
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

    #############################################################################
    ## Casting methods
    #############################################################################
    @staticmethod
    def coerced_op(seq1, seq2, operation: str):
        r'''
            Method to convert sequences to a common type in order to apply some operation 

            This method tries to convert sequences to different classes to get better 
            or more refined outputs of some operations.

            * If no sequences are given, we raise an error.
            * If only one sequence is given, we treat the other as a constant sequence (see :class:`ConstantSequence`)
            * Then we proceed as follows:
              1. We ask ``seq1`` if it recognizes ``seq2``. If it does, we apply the operation with the transformed sequence.
              2. Otherwise, we ask ``seq2`` if it recognizes ``seq1``. We do as in (1).
              3. If we can not relate the sequence, we fall back to default implementation in :class:`Sequence`.
        '''
        if not isinstance(seq1, Sequence) and not isinstance(seq2, Sequence):
            raise TypeError("At least one sequence must be given for applying an operation")
        elif not isinstance(seq1, Sequence):
            universe = pushout(seq1.parent(), seq2.universe)
            seq1 = ConstantSequence(seq1, universe=universe, dim = seq2.dim, allow_sym = True)
        elif not isinstance(seq2, Sequence):
            universe = pushout(seq1.universe, seq2.parent())
            seq2 = ConstantSequence(seq2, universe=universe, dim = seq1.dim, allow_sym = True)

        ## Now ``seq1`` and ``seq2`` are :class:`Sequence`
        if seq1.dim != seq2.dim:
            raise TypeError(f"The dimensions between sequences do not match ({seq1.dim} vs {seq2.dim})")
        
        try: # seq1 tries to recognize seq2
            seq1, seq2 = seq1.as_self(seq2)
            return getattr(seq1, operation)(seq2)
        except NotImplementedError:
            try: # seq2 tries to recognize seq1
                seq2, seq1 = seq2.as_self(seq1)
                return getattr(seq2, operation)(seq1)
            except NotImplementedError:
                # We fall back to default behavior
                return getattr(Sequence, operation)(seq1,seq2)

    def change_universe(self, new_universe):
        r'''
            Method that change the universe of a sequence. This can help to use the same universe in different 
            spaces or when it is required to force a specific universe.
        '''
        return LambdaSequence(lambda *n : self._element(*n), new_universe, dim=self.dim, allow_sym=self.allow_sym)

    def as_self(self, other: Sequence) -> tuple[Sequence, Sequence]:
        r'''
            Method to try and recognize a sequence of the same type so the operations in ``self.__class__`` work properly.
        '''
        raise NotImplementedError("By default, a sequence does not recognize anyone.")
    
    #############################################################################
    ## Element methods
    #############################################################################
    def element(self, *indices : int):
        if not tuple(indices) in self.__CACHE_ELEMENTS:
            try:
                output = self._element(*indices)
            except ZeroDivisionError:
                output = oo
            except:
                output = NaN

            if not output in (NaN, oo):
                try:
                    output = self.universe(output) #pylint: disable=not-callable
                except Exception as e:
                    if not self.allow_sym:
                        raise e
            
            self.__CACHE_ELEMENTS[tuple(indices)] = output
        return self.__CACHE_ELEMENTS[tuple(indices)]

    def _element(self, *indices : int):
        raise NotImplementedError("Method '_element' not implemented")

    #############################################################################
    ## Operation methods
    #############################################################################
    @cached_method
    def shift(self, *shifts : int) -> Sequence:
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

    def subsequence(self, *vals : int) -> Sequence:
        r'''
            Method to obtain a subsequence when having multiple arguments.

            When we have a sequence `f: \mathbb{N}^k \rightarrow R`, we can fix the first 
            arguments `(a_1,\ldots, a_t)` and build a new sequence by:

            .. MATH::

                \tilde{f}(n_1,\ldots, n_{k-t}) = f(a_1,\ldots,a_t,n_1,\ldots,n_{k-t})

            This method build the corresponding subsequence for the given values in ``vals``.
            If the number of values is more than the number of inputs, then we return the value
            of the sequence at the given position.

            TODO: Add the option for more sparse subsequences: if vals is given as a sequence over
            `\mathbb{N}, then we can still build a subsequence from it. This is more on the line of 
            a sparse subsequence.
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

    def interlace(self, *others : Sequence, dim_to_interlace: int = 0):
        return InterlacingSequence(self, *others, dim_to_interlace=dim_to_interlace)

    def linear_subsequence(self, index: int, scale: int, shift: int):
        r'''Method to compute a linear subsequence of ``self``'''
        if index < 0 or index >= self.dim: raise IndexError(f"The index given must be in the dimension ({self.dim}). Got {index}")
        if scale <= 0 or not scale in ZZ: raise ValueError(f"The scale must be a positive integer. Got {scale}")
        if shift < 0 or not shift in ZZ: raise ValueError(f"The shift given must be a non-negative integer. Got {shift}")

        return self._linear_subsequence(index,scale,shift)
    
    def _linear_subsequence(self, index:int, scale:int, shift: int) -> Sequence:
        r'''Method to actually compute the new subsequence. It assumes the arguments to be valid.'''
        change_index = lambda *n : tuple(n[i] if i != index else n[i]*scale+shift for i in range(len(n)))

        return LambdaSequence(lambda *n : self(*change_index(*n)), self.universe, self.dim, self.allow_sym)

    ## Arithmetic methods
    ## These methods defines the default behavior when the sequences do not recognize each other
    def _add_(self, other: Sequence) -> Sequence:
        new_universe = pushout(self.universe, other.universe)
        return LambdaSequence(lambda *n: self(*n) + other(*n), universe = new_universe, dim = self.dim)
    
    def _sub_(self, other: Sequence) -> Sequence:
        new_universe = pushout(self.universe, other.universe)
        return LambdaSequence(lambda *n: self(*n) - other(*n), universe = new_universe, dim = self.dim)
    
    def _neg_(self) -> Sequence:
        return LambdaSequence(lambda *n : -self(*n), self.universe, dim = self.dim, allow_sym=self.allow_sym) # pylint: disable=invalid-unary-operand-type
    
    def _mul_(self, other: Sequence) -> Sequence:
        new_universe = pushout(self.universe, other.universe)
        return LambdaSequence(lambda *n: self(*n) * other(*n), universe = new_universe, dim = self.dim)
        
    def _truediv_(self, other: Sequence) -> Sequence:
        new_universe = pushout(self.universe, other.universe)
        return LambdaSequence(lambda *n: self(*n) / other(*n), universe = new_universe, dim = self.dim)
    
    #############################################################################
    ## Magic methods for Python
    #############################################################################
    # Arithmetic methods
    def __add__(self, other):
        try:
            return Sequence.coerced_op(self, other, "_add_")
        except:
            return NotImplemented

    def __sub__(self, other):
        try:
            return Sequence.coerced_op(self, other, "_sub_")
        except:
            return NotImplemented

    def __mul__(self, other):
        try:
            return Sequence.coerced_op(self, other, "_mul_")
        except:
            return NotImplemented

    def __truediv__(self, other):
        try:
            return Sequence.coerced_op(self, other, "_truediv_")
        except:
            return NotImplemented
        
    def __neg__(self):
        return self._neg_()
        
    # Reverse arithmetic methods
    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__sub__(other)
    def __rmul__(self, other): return self.__mul__(other)
    def __rtruediv__(self, other): 
        try:
            return Sequence.coerced_op(other, self, "_truediv_")
        except:
            return NotImplemented

    # Methods to get elements
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

    #############################################################################
    ## Equality and checking methods
    #############################################################################
    def almost_zero(self, order=10) -> bool:
        r'''
            Method that checks if a sequence is zero at the beginning.

            This method receives a number of elements to be checked and then it proceeds to check if all the first 
            elements of the sequence are equal to zero. This is helpful in some sequence to guarantee that the whole 
            sequence is exactly zero.
        '''
        first_terms = product(range(order), repeat=self.dim)
        if isinstance(self.universe, SequenceSet):
            return all(self(*term).almost_zero(order) for term in first_terms)
        else:
            return all(self(*term) == 0 for term in first_terms)

    def almost_equals(self, other : Sequence, order=10) -> bool:
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
        first_terms = product(range(order), repeat=self.dim)
        if isinstance(self.universe, SequenceSet): # equality if the codomain are more sequences
            return all(self(*term).almost_equals(other(*term), order) for term in first_terms)
        else: # equality when elements lied in other ring
            return all(self(*term) == other(*term) for term in first_terms)

    def __eq__(self, other: Sequence) -> bool:
        r'''Checking for partial equality'''
        if not isinstance(other, Sequence):
            return False
        return self.almost_equals(other, order=Sequence.EQUALITY_BOUND)
    
    #############################################################################
    ## Representation methods
    #############################################################################
    def __repr__(self) -> str:
        if self.dim == 1:
            return f"Sequence over [{self.universe}]: ({self[0]}, {self[1]}, {self[2]},...)"
        else:
            return f"Sequence with {self.dim} variables over [{self.universe}]"

class SequenceSet(Homset,UniqueRepresentation):
    r'''
        Class for the set of sequences. We implement more coercion methods to allow more operations for sequences.
    '''
    Element = Sequence

    def __init__(self, dimension, codomain):
        domain = NN if dimension == 1 else cartesian_product(dimension*[NN])
        super().__init__(domain, codomain, category=_Sets)

    def dimension(self):
        try:
            return len(self.domain().construction()[1])
        except TypeError:
            return 1

    ## Category and coercion methods
    def _element_constructor_(self, x, check=None, **options):
        if x in self.codomain():
            return LambdaSequence(lambda *_ : x, self.codomain(), dim=self.dimension(), allow_sym=True)
        elif parent(x).has_coerce_map_from(self.codomain()):
            return LambdaSequence(lambda *_: x, pushout(self.codomain(), parent(x)), dim=self.dimension(), allow_sym=True)
        
        return super()._element_constructor_(x, check, **options)

    def _coerce_map_from_(self, S):
        return super().has_coerce_map_from(S) or self.codomain().has_coerce_map_from(S)

    def __repr__(self):
        return f"Set of Sequences from NN{'' if self.dimension()==1 else f'^{self.dimension()}'} to {self.codomain()}"

###########################################################################################
### Specific implementations of the class Sequence
###########################################################################################
class ConstantSequence(Sequence):
    r'''Class defining constant sequences'''
    def __init__(self, constant, universe=None, dim: int = 1, **kwds):
        super().__init__(universe, dim, kwds.pop("allow_sym", True), **kwds)
        self.__constant = self.universe(constant) #pylint: disable=not-callable

    ### ABSTRACT METHODS
    def _element(self, *indices : int):
        return self.__constant
    
    ### RECOMMENDED METHODS TO OVERWRITE
    def _shift(self, *shifts) -> ConstantSequence:
        return self
    
    def _subsequence(self, *vals) -> ConstantSequence: 
        return self
    
    def _linear_subsequence(self, index:int, scale:int, shift: int) -> ConstantSequence:
        return self
    
    ### OTHER METHODS OVERWRITTEN
    def change_universe(self, new_universe) -> ConstantSequence:
        return ConstantSequence(self.__constant, new_universe, dim=self.dim)
    
    #############################################################################
    ## Equality methods
    #############################################################################
    def __eq__(self, other):
        if not isinstance(other, ConstantSequence):
            return super().__eq__(other)
        return self.__constant == other.__constant
    
    def __hash__(self):
        return hash(self.__constant)

    #############################################################################
    ## Representation methods
    #############################################################################
    def __repr__(self) -> str:
        if self.dim == 1:
            return f"Constant sequence over [{self.universe}]: {self.__constant}"
        else:
            return f"Constant sequence with {self.dim} variables over [{self.universe}]: {self.__constant}"
        
    def _latex_(self) -> str:
        return latex(self.__constant)
    
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
    def __init__(self, func=None, universe=None, dim = 1, allow_sym = False, **kwds):
        super().__init__(
            universe, dim=dim, allow_sym=allow_sym, # arguments for Sequence
            **kwds # other arguments for builders (allowing multi-inheritance)
        )

        self.__func = func

    ### ABSTRACT METHODS
    def _element(self, *indices : int):
        return self.__func(*indices)

    #############################################################################
    ## Equality methods
    #############################################################################
    def __eq__(self, other):
        if not isinstance(other, LambdaSequence):
            return super().__eq__(other)
        return self.__func == other.__func
    
    def __hash__(self):
        return hash(self.__func)

class ExpressionSequence(Sequence):
    r'''
        Implementation of :class:`Sequence` using expressions from Sage.

        This class computes a sequence by calling setting up an expression an a list of variables to be the variables of the
        sequence. This class can always be evaluated symbolically and a general expression can be obtained.

        Several methods from :class:`Sequence`are overridden to take into account the use of :class:`ExpressionSequence` for 
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

            sage: from pseries_basis.misc.sequences import *
            sage: F = ExpressionSequence(factorial(x), ZZ)
            sage: F2 = LambdaSequence(lambda n : factorial(n), ZZ)
            sage: F.almost_equals(F2, 100)
            True
            sage: F
            Sequence over [Integer Ring]: factorial(x)
    '''
    def __init__(self, expr=None, universe=None, variables=None, **kwds):
        if expr == None:
            raise TypeError("An ExpressionSequence requires an expression different than 'None'")
        if not expr in SR:
            raise ValueError("The expression must be something we can convert to an expression")
        expr = SR(expr)
        variables = expr.variables() if variables is None else tuple(variables)

        super().__init__(
            universe, len(variables), True, # arguments for Sequence
            **kwds # arguments for other builders (allowing multi-inheritance)
        )
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

    ### ABSTRACT METHODS
    def _element(self, *indices: int):
        vars = self.variables()
        return self.generic()(**{str(vars[i]) : indices[i] for i in range(len(indices))})

    ### RECOMMENDED METHODS TO OVERWRITE
    def _shift(self, *shifts):
        vars = self.variables()
        evaluations = [vars[i] + shifts[i] for i in range(self.dim)]
        new_expr = self.generic()(**{str(vars[i]) : evaluations[i] for i in range(self.dim)})

        return ExpressionSequence(new_expr, self.universe, vars)

    def _subsequence(self, *vals):
        vars = self.variables()
        new_expr = self.generic(**{str(vars[i]) : vals[i] for i in range(len(vals))}) 
        rem_vars = vars[len(vals):]
        return ExpressionSequence(new_expr, self.universe, rem_vars)
    
    def _linear_subsequence(self, index:int, scale:int, shift: int) -> Sequence:
        r'''Method to actually compute the new subsequence. It assumes the arguments to be valid.'''
        variable = self.variables()[index]
        new_generic = self.generic()(**{str(variable) : scale*variable + shift})

        return ExpressionSequence(new_generic, self.universe, self.variables())
    
    ### OVERWRITING METHODS FOR COERCION
    def as_self(self, other: Sequence) -> tuple[Sequence, Sequence]:
        universe = pushout(self.universe, other.universe)
        if isinstance(other, ConstantSequence):
            return self.change_universe(universe), ExpressionSequence(other[0], universe, self.variables())
        elif isinstance(other, LambdaSequence):
            try:
                other_generic = other(*self.variables())
                if other_generic in (NaN, oo):
                    raise NotImplementedError("LambdaSequence not valid for an expression")
                return self.change_universe(universe), ExpressionSequence(other_generic, universe, self.variables())
            except:
                raise NotImplementedError("LambdaSequence not valid for an expression")
        elif isinstance(other, ExpressionSequence):
            if [SR(v) for v in self.variables()] != [SR(v) for v in other.variables()]:
                raise NotImplementedError("The variable names are different between ExpressionSequences")
            return self.change_universe(universe), other.change_universe(universe)
        else:
            raise NotImplementedError(f"Class {other.__class__} not recognized by ExpressionSequence")
        
    def _add_(self, other: ExpressionSequence) -> ExpressionSequence:
        return ExpressionSequence(self.generic() + other.generic(), self.universe, self.variables())
    def _sub_(self, other: ExpressionSequence) -> ExpressionSequence:
        return ExpressionSequence(self.generic() - other.generic(), self.universe, self.variables())
    def _mul_(self, other: ExpressionSequence) -> ExpressionSequence:
        return ExpressionSequence(self.generic() * other.generic(), self.universe, self.variables())
    def _truediv_(self, other: ExpressionSequence) -> ExpressionSequence:
        return ExpressionSequence(self.generic() / other.generic(), self.universe, self.variables())
    def _neg_(self):
        return ExpressionSequence(-self.generic(), self.universe, self.variables())

    ### OTHER METHODS OVERWRITTEN
    def change_universe(self, new_universe):
        return ExpressionSequence(self.generic(), new_universe, self.variables())

    #############################################################################
    ## Equality methods
    #############################################################################
    def __eq__(self, other):
        if not isinstance(other, ExpressionSequence):
            return super().__eq__(other)
        if [SR(v) for v in self.variables()] != [SR(v) for v in other.variables()]:
            return False
        return bool(self.generic() == other.generic())
    
    def __hash__(self):
        return hash(self.__generic)

    #############################################################################
    ## Representation methods
    #############################################################################
    def __repr__(self) -> str:
        return f"Sequence{f' in {self.__vars}' if self.dim > 1 else ''} over [{self.universe}]: {self.generic()}"
    
    def _latex_(self) -> str:
        return r"\left(" + latex(self.generic()) + r"\right)_{" + ",".join(latex(v) for v in self.variables()) + r" \in \mathbb{N}}"

class RationalSequence(Sequence):
    def __init__(self, func=None, ring=None, variables=None, **kwds):
        func, ring, variables, universe = RationalSequence._init_arguments(func, ring, variables)
        self.__generic = func
        self.__ring = ring
        self.__variables = variables

        # removing possible repeated arguments
        kwds.pop("universe",None); kwds.pop("dim", None); kwds.pop("allow_sym", None)
        super().__init__(
            universe=universe, dim=len(self.__variables), allow_sym=True, # arguments for Sequence
            **kwds # other arguments (allowing multi-inheritance)
        )

    @property
    def generic(self):
        r'''
            Method that returns the generic function for this sequence as a rational function (or polynomial)

            This method returns the function that defines this sequence. Use method :func:`variables` 
            to obtain the list (in appropriate order) of variables that are used for this sequence.
        '''
        return self.__generic

    @property
    def variables(self):
        r'''
            Method that returns the variables of this sequence.

            This method returns the (sorted) tuple of variables that defines this sequence. Use method :func:`generic` 
            to obtain the defining function of this sequence.
        '''
        return self.__variables

    @property
    def ring(self):
        return self.__ring

    @staticmethod
    def _init_arguments(func, ring, variables):
        if func is None:
            raise ValueError("A rational sequence require a function to be defined")
        # checking the input to decide the universe and the dimension of the sequence
        R = func.parent()
        if ring != None:
            R = pushout(R, ring)

        if R.is_field(): # it may be a rational function space
            if not (is_PolynomialRing(R.base()) or is_MPolynomialRing(R.base())):
                raise TypeError("The ring of the given function must be a polynomial field or its field of fractions")
        if variables == None:
            variables = R.gens()
        else:
            gen_names = [str(g) for g in R.gens()]
            
            if any(str(v) not in gen_names for v in variables):
                raise ValueError("The given variables are not in the ring provided with the function")
            variables = tuple([R(str(v)) for v in variables])
        universe = R.base_ring() if len(variables) == R.ngens() else R
        return func, ring, variables, universe

    def change_universe(self, new_universe):
        if len(self.variables) != self.ring.ngens():
            return RationalSequence(func=self.generic, ring=new_universe, variables=self.variables)
        else:
            return RationalSequence(func=self.generic.change_ring(new_universe), variables=self.variables)

    def _element(self, *indices: int):
        vars = self.variables
        return self.generic(**{str(vars[i]) : indices[i] for i in range(len(indices))})

    def _shift(self, *shifts):
        vars = self.variables
        evaluations = [vars[i] + shifts[i] for i in range(self.dim)]
        new_expr = self.generic(**{str(vars[i]) : evaluations[i] for i in range(self.dim)})

        return RationalSequence(func=new_expr, ring=self.ring, variables=vars)

    # basic arithmetic methods
    def __add__(self, other):
        if not isinstance(other, RationalSequence) or self.dim != other.dim:
            return super().__add__(other)

        if [str(v) for v in self.variables] != [str(v) for v in other.variables]:
            return NotImplemented

        new_ring = pushout(self.ring, other.ring)
        new_func = new_ring(self.generic) + new_ring(other.generic)
        return RationalSequence(func=new_func, ring=new_ring, variables=self.variables)

    def __sub__(self, other):
        if not isinstance(other, RationalSequence) or self.dim != other.dim:
            return super().__add__(other)

        if [str(v) for v in self.variables] != [str(v) for v in other.variables]:
            return NotImplemented

        new_ring = pushout(self.ring, other.ring)
        new_func = new_ring(self.generic) - new_ring(other.generic)
        return RationalSequence(func=new_func, ring=new_ring, variables=self.variables)

    def __mul__(self, other):
        if isinstance(other, RationalSequence) and self.dim == other.dim:
            if [str(v) for v in self.variables] != [str(v) for v in other.variables]:
                return NotImplemented

            new_ring = pushout(self.ring, other.ring)
            new_func = new_ring(self.generic) * new_ring(other.generic)
            return RationalSequence(func=new_func, ring=new_ring, variables=self.variables)
        elif not isinstance(other, Sequence):
            universe = pushout(self.universe, parent(other))
            if len(self.variables) != self.ring.ngens():
                new_ring = universe
            else:
                new_ring = self.ring.change_ring(universe)
            new_func = new_ring(other)*new_ring(self.generic)
            return RationalSequence(func=new_func, ring=new_ring, variables=self.variables)
        return super().__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, RationalSequence) and self.dim == other.dim:
            if [str(v) for v in self.variables] != [str(v) for v in other.variables]:
                return NotImplemented

            new_ring = pushout(self.ring, other.ring)
            new_func = new_ring(self.generic) / new_ring(other.generic)
            return RationalSequence(func=new_func, ring=new_ring, variables=self.variables)
        elif not isinstance(other, Sequence):
            universe = pushout(self.universe, parent(other))
            if len(self.variables) != self.ring.ngens():
                new_ring = universe
            else:
                new_ring = self.ring.change_ring(universe)
            new_func = new_ring(self.generic) / new_ring(other)
            return RationalSequence(func=new_func, ring=new_ring, variables=self.variables)
        return super().__truediv__(other)
        
    def __neg__(self):
        return RationalSequence(func=-self.generic, ring=self.ring, variables=self.variables)
    
    def _subsequence(self, *vals):
        vars = self.variables
        new_expr = self.generic(**{str(vars[i]) : vals[i] for i in range(len(vals))}) 
        rem_vars = vars[len(vals):]
        return RationalSequence(func=new_expr, ring=self.ring, variables=rem_vars)

    # equality and hash methods
    def __eq__(self, other):
        if not isinstance(other, RationalSequence):
            return False
        if [str(v) for v in self.variables()] != [str(v) for v in other.variables]:
            return False
        return self.generic == other.generic
    
    def __hash__(self):
        return hash(self.__generic)

    def __repr__(self) -> str:
        return f"Sequence{f' in {self.variables}' if self.dim > 1 else ''} over [{self.universe}]: {self.generic}"

class InterlacingSequence(Sequence):
    r'''
        Interlaced sequence.

        This sequence computes the interlacing among several sequences in one of its dimensions.

        EXAMPLES::

            sage: from sage.pseries_basis import *
            sage: fact = LambdaSequence(lambda n : factorial(n), QQ)
            sage: zero = LambdaSequence(lambda n : 0, ZZ)
            sage: InterlacingSequence(fact, zero)[:10]
            [1, 0, 1, 0, 2, 0, 6, 0, 24, 0]
    '''
    def __init__(self, *sequences: Sequence, dim_to_interlace: int = 0, **kwds):
        if len(sequences) == 1 and isinstance(sequences[0], Collection):
            sequences = sequences[0]

        if len(sequences) == 0:
            raise TypeError("Interlacing of sequences is defined for at least one sequence")

        # we check the dimension of all sequences
        dim = sequences[0].dim
        if [seq.dim for seq in sequences].count(dim) != len(sequences):
            raise TypeError("The dimension for interlacing sequences must be the same")
        
        # we compute a common universe        
        universe = reduce(lambda p,q : pushout(p,q), [seq.universe for seq in sequences])

        # we compute if we can compute the symbolic expression for the interlaced sequence
        allow_sym = all(seq.allow_sym for seq in sequences)

        # we check the dimension to interlace:
        if dim_to_interlace < 0 or dim_to_interlace >= len(sequences):
            raise ValueError("The dimension to interlace must be valid")
        self.__sequences = tuple(sequences)
        self.__to_interlace = dim_to_interlace

        # Removing possible duplicated arguments
        kwds.pop("universe", None); kwds.pop("dim", None); kwds.pop("allow_symb", None)
        super().__init__(
            universe, dim, allow_sym, # arguments for Sequence
            **kwds                    # arguments for possible multi-inheritance
        )

    @property
    def sequences(self) -> tuple[Sequence]: return self.__sequences
    @property
    def interlacing(self) -> int: return self.__to_interlace

    def _element(self, *indices: int):
        r = indices[self.__to_interlace]%len(self.sequences)
        indices = list(indices); indices[self.__to_interlace] //= len(self.sequences)
        return self.sequences[r].element(*indices)

__all__ = ["Sequence", "SequenceSet", "LambdaSequence", "ConstantSequence", "ExpressionSequence", "RationalSequence", "InterlacingSequence"]
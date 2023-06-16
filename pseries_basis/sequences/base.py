
from __future__ import annotations

import logging

from collections.abc import Callable
from functools import cached_property
from itertools import product
from sage.all import (cached_method, cartesian_product, parent, var, oo, NaN, SR, ZZ)
from sage.categories.all import CommutativeRings, Sets
from sage.categories.homset import Homset
from sage.categories.homsets import Homsets
from sage.categories.morphism import SetMorphism # pylint: disable=no-name-in-module
from sage.categories.pushout import ConstructionFunctor, pushout
from sage.graphs.digraph import DiGraph
from sage.structure.element import Element
from sage.structure.unique_representation import UniqueRepresentation

_Sets = Sets.__classcall__(Sets)
_CommutativeRings = CommutativeRings.__classcall__(CommutativeRings)
_Homsets = Homsets.__classcall__(Homsets)
logger = logging.getLogger(__name__)
       
###############################################################################################################################################
### Base class for sequences
###############################################################################################################################################
class Sequence(SetMorphism):
    r'''
        Main class for sequences. It defines a universe for its elements and a general interface to access the sequence.

        To extend this class to a more detailed type of sequence, we need to do the following:
        
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

        This class inherits from :class:`sage.categories.morphism.SetMorphism`allowing us to use methods such as
        ``domain`` and ``codomain``, ``__call__``, etc.

        However, we have overriden the method __mul__ to fit better to the parent-element Sage framework.
    '''
    def __init__(self, sequence: Callable, universe = None, dim : int = 1, *, _extend_by_zero=False):
        if universe is None:
            raise TypeError("The universe of a sequence must never be None")
        self.__sequence = sequence
        self.__extend_by_zero = _extend_by_zero
        parent = SequenceSet(dim, universe)
        func = (lambda n : self.element(*n)) if dim > 1 else (lambda n : self.element(n))
        self.__CACHE_ELEMENTS = {}

        super().__init__(parent, func)
        self.__class__.resgister_class()

    #############################################################################
    ## Methods related to the Graph of sequences classes
    #############################################################################
    CLASSES_GRAPH: DiGraph = DiGraph(loops=False, multiedges=False, weighted=False, data_structure="sparse")

    @classmethod
    def resgister_class(cls):
        cls._resgister_class()

    @classmethod
    def _resgister_class(cls, *super_classes):
        if len(super_classes) == 0 and cls != Sequence:
            super_classes = [Sequence]
        if not cls in Sequence.CLASSES_GRAPH.vertices(sort=False):
            Sequence.CLASSES_GRAPH.add_vertex(cls)
            for sup_class in super_classes:
                Sequence.CLASSES_GRAPH.add_edge(cls, sup_class)

    @staticmethod
    def MinimalCommonClass(cls1, cls2):
        if cls1 == cls2: return cls1

        if any(cls not in Sequence.CLASSES_GRAPH.vertices(sort=False) for cls in (cls1,cls2)):
            raise ValueError(f"One the the classes ({cls1} or {cls2}) not registered")
        
        # we search the class Sequence by breadth marking the depth of found vertices from cls1
        depth = {cls1 : 0}
        to_check = [cls1]
        while len(to_check) > 0:
            current = to_check.pop(0)
            for v in Sequence.CLASSES_GRAPH.neighbor_iterator(current):
                if not v in depth:
                    depth[v] = depth[current] + 1
                    to_check.append(v)
        
        # now we do a breadth search from cls2 until we find something like in 1
        to_check = [cls2]
        while len(to_check) > 0:
            found_intersection = [v for v in to_check if v not in depth]
            if len(found_intersection) > 0: # we return the minimum intersection
                depths = [depth[v] for v in found_intersection]
                return found_intersection[depths.index(min(depths))]
            else: # not found yet. we check next layer of vertices
                to_check = sum([[v for v in Sequence.CLASSES_GRAPH.neighbor_iterator(w)] for w in to_check],[])

        logger.warning(f"No common class found for {cls1} and {cls2}. Returning default class")
        return Sequence

    ### Other static attributes
    EQUALITY_BOUND: int = 50

    #############################################################################
    ## Object properties
    #############################################################################
    @cached_property
    def universe(self): 
        r'''
            Attribute with the common parent for all the elements of this sequence
        '''
        return self.codomain()
    
    @cached_property
    def dim(self) -> int:
        r'''
            Attribute with the number of variables for the sequence.
        '''
        return self.parent().dimension()
    
    #############################################################################
    ## Casting methods
    #############################################################################
    def change_universe(self, new_universe):
        r'''
            Method that change the universe of a sequence. This can help to use the same universe in different 
            spaces or when it is required to force a specific universe.
        '''
        return Sequence(lambda *n : self._element(*n), new_universe, dim=self.dim)

    def change_class(self, goal_class):
        r'''
            This method transforms the current sequence to an equivalent sequence
            in a different (but compatible) class. This method defines how to 
            convert a sequence into another class.
        '''
        if self.__class__ == goal_class:
            return self
        
        # we first check that the class is registered and is compatible with self.__class__
        if not goal_class in Sequence.CLASSES_GRAPH.vertices(sort=False):
            raise TypeError(f"Class {goal_class} not recognized in the Sequence framework.")
        
        # we do a search by breadth to find the goal_class from self.__class__
        to_check = [self.__class__]; prev_class = dict()
        while len(to_check) > 0 and (not goal_class in to_check):
            current = to_check.pop(0)
            for v in Sequence.CLASSES_GRAPH.neighbor_iterator(current):
                to_check.append(v); prev_class[v] = current
                if v == goal_class: break
            else: # if we do not break, we are not finished
                continue
            break # otherwise we simply get out of the loop
        else: # if we never break, it means we did not find "goal_class"
            raise TypeError(f"The class {goal_class} is not admissible for {self.__class}")
        path_to_goal = [goal_class]; current = goal_class
        while prev_class[current] != self.__class__:
            current = prev_class[current]
            path_to_goal.insert(0, current)
        
        current = self
        for cls in path_to_goal:
            current = current._change_class(cls)
        return current
    
    def _change_class(self, _):
        raise NotImplementedError(f"For base Sequence, no call to _change_class should be done.")
        
    #############################################################################
    ## Element methods
    #############################################################################
    def element(self, *indices : int, _generic=False):
        if len(indices) != self.dim:
            raise TypeError(f"Invalid number of arguments for a sequence. Provided {len(indices)} but required {self.dim}")
        
        if (not _generic) and any(index < 0 for index in indices):
            return 0 if self.__extend_by_zero else NaN
        
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
                    if not _generic:
                        raise e
            
            self.__CACHE_ELEMENTS[tuple(indices)] = output
        return self.__CACHE_ELEMENTS[tuple(indices)]

    def _element(self, *indices : int):
        return self.__sequence(*indices)
    
    def generic(self, *names: str):
        r'''
            Method that tries to create a generic evaluation for the sequence.

            The given names are used as variable names.
        '''
        
        if len(names) == 0: # if nothing provided we create default names
            names = ["n"] if self.dim == 1 else [f"n_{i}" for i in range(1,self.dim+1)]
        if len(names) < self.dim:
            raise TypeError(f"Insufficient variables names provided")
        names = var(names) # we create the variables
        result = SR(self.element(*names[:self.dim], _generic=True))
        if result is NaN:
            raise ValueError("Impossible to compute generic expression for a sequence.")
        return result
    
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

    #############################################################################
    ## Operation with sequences
    ##
    ## In this seciton we present the main methods that are defined over sequences
    ## including, but not being limited to:
    ##   * Shifts -> shifting the arguments of the sequence
    ##   * Subsequences -> getting a subsequence indexed by other sequence, but keeping the dimension
    ##   * Slicing -> getting a part of the sequence reducing the dimension
    ##   * Interlacing -> interlacing several sequences
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
        return Sequence(lambda *n : self._element(*[n[i]+shifts[i] for i in range(self.dim)]), self.universe, dim=self.dim)

    def slicing(self, *vals : tuple[int,int]) -> Sequence:
        r'''
            Method to obtain a slice of a sequence.

            When we have a sequence `f: \mathbb{N}^k \rightarrow R`, we can fix some of the arguments 
            arguments `(a_1,\ldots, a_t)` and build a new sequence by:

            .. MATH::

                \tilde{f}(n_1,\ldots, n_{k-t}) = f(a_1,\ldots,a_t,n_1,\ldots,n_{k-t})

            This method build the corresponding slicing sequence where the dimension has been reduced
            consequently.

            INPUT:

            * A list of tuples `(i,n)` such that we will fix the `i`-th entry to take the value `n`.
        '''
        if any(0 < i or i <= self.dim or not n in ZZ for (i,n) in vals):
            raise ValueError(f"Slicing require as input a list of tuples (i,n) with valid indices `i` and integers `n`")
        values = dict(vals)

        if len(vals) >= self.dim: # we do not have a sequence but an element
            return self.element(*[values[i] for i in range(self.dim)])
        
        return self._slicing(values)
    
    def _slicing(self, values: dict[int,int]):
        def to_original_input(n: list):
            result = []
            for i in range(self.dim):
                if i in values:
                    result.append(values[i])
                else:
                    result.append(n.pop(0))
            return result
        return Sequence(lambda *n: self._element(to_original_input(n)), self.universe, self.dim)

    def subsequence(self, *vals: tuple[int, Sequence]) -> Sequence:
        r'''
            Method to compute a subsequence of a given sequence.

            This allows to substitute the arguments of the current sequence by another sequence over the integers.
            This is the equivalent as the composition of two functions and, hence, we require that the 
            composed function maps integers to integers.

            This allows to obtain sparse subsequences, linear subsequences, etc.

            INPUT:

            * A list of tuples indicating the subsequence to be taken. It can have several formats:
              - `(i, f: \mathbb{N} \rightarrow \mathbb{N}): changes the index `i` by the sequence `f`.
              - `(i, (a,b))`: transforms the `i`-th index of the sequence linearly by taking the subsequence `ai+b`.

            OUTPUT:

            The corresponding subsequence after taking the consideration of the input.
        '''
        final_input = []
        for value in vals:
            i, seq = value
            if not i in ZZ or i < 0 or i >= self.dim:
                raise IndexError(f"[subsequence] The given index is not valid (got {i}). It must be a non-negative integer smaller than {self.dim}")
            if isinstance(seq, (list,tuple)):
                if len(seq) != 2:
                    raise TypeError(f"[subsequence - linear] Error in format for a linear subsequence. Expected a pair of integers")
                elif any((not el in ZZ) for el in seq):
                    raise TypeError(f"[subsequence - linear] Error in format for a linear subsequence. Expected a pair of integers")
                a, b = seq
                final_input.append([(i,),Sequence(lambda n : a*n+b, ZZ, 1)])
            else: 
                final_input.append((i,seq))
                
        for value in final_input:
            index, seq = value
            if (not index in ZZ) or index < 0 or index >= self.dim:
                raise ValueError(f"[subsequence] Indices are given wrongly: they need to be non-negative integers smaller than {self.dim}")
            elif not isinstance(seq, Sequence):
                raise TypeError(f"[subsequence] Subsequence values are given wrongly: they need to be a sequence")
            elif seq.dim != 1:
                raise TypeError(f"[subsequence] Subsequence values are given wrongly: they need to match the dimension of the indices (got: {seq.dim}, expected: {1})")
            elif pushout(seq.universe, ZZ) != ZZ:
                raise TypeError(f"[subsequence] Subsequence values are given wrongly: they need to be sequences over integers")
            
        return self._subsequence(dict(final_input))
    
    def _subsequence(self, final_input: dict[int, Sequence]):
        return Sequence(lambda *n : self._element(*[final_input[i](n[i]) if i in final_input else n[i] for i in range(self.dim)]), self.universe, self.dim)

    def interlace(self, *others : Sequence, dim_to_interlace: int = 0):
        raise NotImplementedError("Method 'interlacing' not yet implemented.")

    def linear_subsequence(self, index: int, scale: int, shift: int):
        r'''Method to compute a linear subsequence of ``self``'''
        if index < 0 or index >= self.dim: raise IndexError(f"The index given must be in the dimension ({self.dim}). Got {index}")
        if scale <= 0 or not scale in ZZ: raise ValueError(f"The scale must be a positive integer. Got {scale}")
        if shift < 0 or not shift in ZZ: raise ValueError(f"The shift given must be a non-negative integer. Got {shift}")

        return self._subsequence((index,(scale,shift)))
    
    #############################################################################
    ## Arithmetic methods
    #############################################################################
    def __mul__(self, other): # overrides multiplication as map to set it as element.
        return Element.__mul__(self, other)
    def composition(self, other): # implements the multiplication as a map (i.e., composition)
        return self._composition(other)

    def __coerce_into_common_class__(self, other: Sequence):
        r'''We assume ``other`` is in the same parent as ``self``. Hence it is a :class:`Sequence`'''
        common_class = Sequence.MinimalCommonClass(self.__class__, other.__class__)
        self_casted = self.change_class(common_class)
        other_casted = other.change_class(common_class)

        return (common_class, (self_casted, other_casted))

    ## Method that defines the generic behavior of the sequences once they are pushed to the same parent  
    def _add_(self, other: Sequence) -> Sequence:
        _, (sc, oc) = self.__coerce_into_common_class__(other)
        return sc._final_add(oc)
    
    def _sub_(self, other: Sequence) -> Sequence:
        _, (sc, oc) = self.__coerce_into_common_class__(other)
        return sc._final_sub(oc)
        
    def _mul_(self, other: Sequence) -> Sequence:
        _, (sc, oc) = self.__coerce_into_common_class__(other)
        return sc._final_mul(oc)
        
    def _div_(self, other: Sequence) -> Sequence:
        _, (sc, oc) = self.__coerce_into_common_class__(other)
        return sc._final_div(oc)
    
    def _mod_(self, other: Sequence) -> Sequence:
        _, (sc, oc) = self.__coerce_into_common_class__(other)
        return sc._final_mod(oc)
    
    def _floordiv_(self, other: Sequence) -> Sequence:
        _, (sc, oc) = self.__coerce_into_common_class__(other)
        return sc._final_floordiv(oc)

    ## This is special because is unary. This is equivalent to the other __final_op methods.
    def _neg_(self) -> Sequence:
        return Sequence(lambda *n : (-1)*self._element(*n), self.universe, dim = self.dim) 
    
    ## Final operation method for the basic sequence type
    def _final_add(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) + other._element(*n), self.universe, self.dim)
    def _final_sub(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) - other._element(*n), self.universe, self.dim)
    def _final_mul(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) * other._element(*n), self.universe, self.dim)
    def _final_div(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) / other._element(*n), self.universe, self.dim)
    def _final_mod(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) % other._element(*n), self.universe, self.dim)
    def _final_floordiv(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) // other._element(*n), self.universe, self.dim)
    
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

                sage: from pseries_basis.sequences.base import *
                sage: def fib(n): return 1 if n == 0 else (1 if n==1 else fib(n-1) + fib(n-2))
                sage: Fib = Sequence(fib, ZZ)
                sage: Fac = Sequence(lambda n : factorial(n), ZZ)
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
        domain = ZZ if dimension == 1 else cartesian_product(dimension*[ZZ])
        super().__init__(domain, codomain, category=_Sets)

    def dimension(self):
        try:
            return len(self.domain().construction()[1])
        except TypeError:
            return 1

    ## Category and coercion methods
    def _element_constructor_(self, x, check=None, **options):
        if x in self.codomain():
            return Sequence(lambda *_ : x, self.codomain(), dim=self.dimension()) ## TODO: Change for constant sequence
        elif parent(x).has_coerce_map_from(self.codomain()):
            return Sequence(lambda *_: x, pushout(self.codomain(), parent(x)), dim=self.dimension()) ## TODO: Change for constant sequence
        elif isinstance(x, Sequence) and x.dim == self.dimension():
            return x.change_universe(self.codomain())
        
        return super()._element_constructor_(x, check, **options)
    
    def construction(self):
        return SequenceFunctor(self.dimension()), self.codomain()

    def _coerce_map_from_(self, S):
        return pushout(self.codomain(), S.codomain() if isinstance(S, SequenceSet) else S) == self.codomain()

    def __repr__(self):
        return f"Set of Sequences from NN{'' if self.dimension()==1 else f'^{self.dimension()}'} to {self.codomain()}"
    
class SequenceFunctor(ConstructionFunctor):
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise TypeError("The dimension must be positive")
        self.__dim = dimension
        self.rank = 100
        
        super().__init__(_CommutativeRings, _Homsets)
    
    def _apply_functor(self, x):
        return SequenceSet(self.__dim, x)
    
    def _repr_(self):
        return f"SequenceSet(*,{self.__dim}])"
    
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__dim == other.__dim

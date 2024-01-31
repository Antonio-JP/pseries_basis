r'''
    Base module for Sequence

    This module contains the basic structures for Sequences and their interaction with the category and parent-element 
    framework of SageMath. We include the main definition for a class for sequence and a specific case for Constant
    Sequence.

    All other types of sequences will inherit from the main class :class:`Sequence` and must define a relation among other
    previously implemented sequence in order to provide appropriate conversion among different types of sequences. This is somehow
    similar to the inheritance or coercion systems from Python or SageMath, but adding an extra layer of flexibility when 
    implementing sequences within the same ring.

    More specifically, we define a Parent structure (see :class:`SequenceSet`) that will include all
    the sequences of a fixed dimension over a fixed ring in SageMath. This, together with the Construction Functor
    :class:`SequenceFunctor` will allow to interact with different objects in SageMath transforming them
    into sequences that can be then computed with in this framework.

    When implementing different types of sequences that all belong to the same parent :class:`SequenceSet`,
    we needed to provide a way to interact even if the classes are different between the sequences. For example,
    we may use the simple sequence given by a callable in Python. However, we may consider simpler types of 
    sequences given by a Polynomial, or a Sage Expression (see :mod:`.element`). We could even consider 
    sequences defined via recurrence equations. 

    That is the reason we provided a system that will allow to register different classes of sequences and 
    the interactions among them are automatically define. Essentially, when a new class is instantiated, we 
    register this class into a directed graph, where the edges define conversions that can be perform. Then, 
    when two sequences meet in an operation and need to be merged, we find the minimal common structure (always 
    existing since the "callable" base definition is always available) that can represent both structures
    and we then perform the operation at that level.

    One particular instance of a class of sequence that is the opposite to the callable sequence is the 
    Constant Sequence. This sequence can be converted to any type of sequence and it is used as a base case for
    all other sequences.

    Basic sequences from callables
    ==================================================
    
    We can easily take any callable with appropriate number of arguments and transform it into a sequence within the 
    framework of this module. We only need to use the :class:`Sequence` class::

        sage: from pseries_basis.sequences import *
        sage: Fac = Sequence(factorial, ZZ, 1)
        sage: Fib = Fibonacci()
        sage: Fac
        Sequence over [Integer Ring]: (1, 1, 2,...)
        sage: Fib
        Sequence over [Integer Ring]: (1, 1, 2,...)
        sage: (Fac + Fib)[:5]
        [2, 2, 4, 9, 29]
        sage: (Fac - Fib)[:5]
        [0, 0, 0, 3, 19]
        sage: (Fac * Fib)[:5]
        [1, 1, 4, 18, 120]
        sage: (Fac / Fib)[:5]
        [1, 1, 1, 2, 24/5]
        sage: (Fac % Fib)[:5]
        [0, 0, 0, 0, 4]
        sage: (Fac // Fib)[:5]
        [1, 1, 1, 2, 4]

    The relation between a constant sequence and a callable sequence will always go to a callable sequence. This is handled automatically::

        sage: Seq2 = ConstantSequence(2, ZZ, 1)
        sage: Seq2[:10]
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        sage: Fib + Seq2
        Sequence over [Integer Ring]: (3, 3, 4,...)
        sage: Seq2 + Fib
        Sequence over [Integer Ring]: (3, 3, 4,...)
        sage: Fib * Seq2
        Sequence over [Integer Ring]: (2, 2, 4,...)
        sage: Seq2 * Fib
        Sequence over [Integer Ring]: (2, 2, 4,...)
        sage: Fib % Seq2
        Sequence over [Integer Ring]: (1, 1, 0,...)

    The coercion will also work with more complexes types of SageMath rings::

        sage: a = NumberField(QQ["a"]("a^2 - 2"), "a").gens()[0]
        sage: SeqA = ConstantSequence(a, a.parent(), 1)
        sage: Seq2 + SeqA
        Sequence over [Number Field in a with defining polynomial a^2 - 2]: (a + 2, a + 2, a + 2,...)
        sage: SeqA * Fac
        Sequence over [Number Field in a with defining polynomial a^2 - 2]: (a, a, 2*a,...)
        sage: SeqA**2 == Seq2
        True

    We can also use any element that can be coerced into the sequence space::

        sage: a + Fac * 2
        Sequence over [Number Field in a with defining polynomial a^2 - 2]: (a + 2, a + 2, a + 4,...)
        sage: a * Fac * a == 2*Fac
        True

    Special operations of sequences
    ==================================================

    Sequences has many operations defined over them. In particular, the shift operation allows to move the sequence back and forth::

        sage: Fib.shift(5)
        Sequence over [Integer Ring]: (8, 13, 21,...)
        sage: Fib.shift(5) == Fibonacci(Fib(5), Fib(6))
        True

    When considering multivariate sequences, the shift operation can be defined to any of its coefficients::

        sage: Binomial.generic("n", "k")
        binomial(n, k)
        sage: Binomial.shift(1,3).generic("n", "k")
        binomial(n + 1, k + 3)

    We can also consider subsequences of any given sequence. A subsequence is defined as a new sequence of the same dimension but where we 
    take the elements of the original sequence in a specific order::

        sage: Binomial.subsequence((1,Fac)).generic("n", "k")
        binomial(n, factorial(k))
        sage: Fac.subsequence((0,Sequence(lambda n: n//2, ZZ, 1)))[:10] # repeated Factorial sequence
        [1, 1, 1, 1, 2, 2, 6, 6, 24, 24]
        sage: Fib.subsequence((0,Sequence(lambda n: n**2 + 2*n + 1, ZZ, 1)))[:10] # quadratic sparse Fibonacci sequence
        [1, 5, 55, 1597, 121393, 24157817, 12586269025, 17167680177565, 61305790721611591, 573147844013817084101]

    The final operation we provide for any sequence is the slicing operation. This is equivalent to the subsequence operation
    when one of the arguments is constant and, hence, removed. Namely, a slice of a sequence is another sequence with **fewer**
    dimension than the original sequence. In the case of slicing *too much* we return just the corresponding element::

        sage: Fac.slicing((0,5)) == Fac(5)
        True
        sage: Binomial.slicing((0,4)).generic("k")
        binomial(4, k)
        sage: Binomial.slicing((1, 10)).generic("n")
        1/3628800*(n - 1)*(n - 2)*(n - 3)*(n - 4)*(n - 5)*(n - 6)*(n - 7)*(n - 8)*(n - 9)*n
        sage: Binomial.slicing((0,5), (1,3)) == Binomial((5,3))
        True

'''
from __future__ import annotations

import logging

from collections.abc import Callable
from functools import cached_property
from itertools import product
from sage.all import (cached_method, cartesian_product, parent, prod, var, oo, NaN, SR, ZZ)
from sage.categories.all import CommutativeRings, Sets
from sage.categories.homset import Homset
from sage.categories.homsets import Homsets
from sage.categories.morphism import SetMorphism # pylint: disable=no-name-in-module
from sage.categories.pushout import ConstructionFunctor, pushout
from sage.graphs.digraph import DiGraph
from sage.structure.element import Element #pylint: disable=no-name-in-module
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
        
        * Override the class method :func:`register_class` with the desired classes we want to be directly below and above
        * Implement the following methods:
          - :func:`_change_dim`: changes the dimension of a sequence.
          - :func:`_change_class`: receives a class (given when registering the sequence class) and cast the current sequence to the new class.
          - :func:`_change_from_class`: class method that receives a sequence in a different class and transform into the current class. 
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
          - :func:`args_to_self` to declare how an element is built.

        This class inherits from :class:`sage.categories.morphism.SetMorphism`allowing us to use methods such as
        ``domain`` and ``codomain``, ``__call__``, etc.

        However, we have overridden the method __mul__ to fit better to the parent-element Sage framework.
    '''
    def __init__(self, sequence: Callable, universe = None, dim : int = 1, *, _extend_by_zero=False, **kwds):
        if universe is None:
            raise TypeError("The universe of a sequence must never be None")
        self.__sequence = sequence
        self.__extend_by_zero = _extend_by_zero
        parent = SequenceSet(dim, universe)
        func = (lambda n : self.element(*n)) if dim > 1 else (lambda n : self.element(n))
        self.__CACHE_ELEMENTS = {}

        super().__init__(parent, func)
        self.__class__.register_class()
        self.__extra_args = kwds

    #############################################################################
    ## Methods related to the Graph of sequences classes
    #############################################################################
    CLASSES_GRAPH: DiGraph = DiGraph(loops=False, multiedges=False, weighted=False, data_structure="sparse")

    @classmethod
    def register_class(cls):
        if len(Sequence.CLASSES_GRAPH) == 0: ## adding the base of Sequence and ConstantSequence
            Sequence.CLASSES_GRAPH.add_vertex(Sequence)
            Sequence.CLASSES_GRAPH.add_vertex(ConstantSequence)
            logger.debug(f"[Sequence - register] Adding edge ({ConstantSequence}) -> ({Sequence})")
            Sequence.CLASSES_GRAPH.add_edge(ConstantSequence, Sequence)
        cls._register_class()

    @classmethod
    def _register_class(cls, super_classes: list = None, sub_classes: list = None):
        if (super_classes == None or len(super_classes) == 0) and cls != Sequence:
            super_classes = [Sequence]
        if (sub_classes == None or len(sub_classes) == 0) and cls != ConstantSequence:
            sub_classes = [ConstantSequence]
        
        if not cls in Sequence.CLASSES_GRAPH.vertices(sort=False):
            Sequence.CLASSES_GRAPH.add_vertex(cls)
            for sup_class in super_classes:
                logger.debug(f"[Sequence - register] Checking if {sup_class} is registered...")
                if not sup_class in Sequence.CLASSES_GRAPH.vertices(sort=False):
                    sup_class.register_class()
                logger.debug(f"[Sequence - register] Adding edge ({cls}) -> ({sup_class})")
                Sequence.CLASSES_GRAPH.add_edge(cls, sup_class)
            for sub_class in sub_classes:
                logger.debug(f"[Sequence - register] Checking if {sub_class} is registered...")
                if not sub_class in Sequence.CLASSES_GRAPH.vertices(sort=False):
                    sub_class.register_class()
                logger.debug(f"[Sequence - register] Adding edge ({sub_class}) -> ({cls})")
                Sequence.CLASSES_GRAPH.add_edge(sub_class, cls)
        else:
            logger.debug(f"[Sequence - register] Trying to register repeatedly a sequence class ({cls})")

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
            for v in Sequence.CLASSES_GRAPH.neighbor_out_iterator(current):
                if not v in depth:
                    depth[v] = depth[current] + 1
                    to_check.append(v)
        
        # now we do a breadth search from cls2 until we find something like in 1
        to_check = [cls2]
        while len(to_check) > 0:
            found_intersection = [v for v in to_check if v in depth]
            if len(found_intersection) > 0: # we return the minimum intersection
                depths = [depth[v] for v in found_intersection]
                return found_intersection[depths.index(min(depths))]
            else: # not found yet. we check next layer of vertices
                to_check = sum([Sequence.CLASSES_GRAPH.neighbors_out(v) for v in to_check],[])

        logger.debug(f"No common class found for {cls1} and {cls2}. Returning default class")
        return Sequence

    def extra_info(self) -> dict:
        return {"extra_args": self.__extra_args}

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
    
    def __getattr__(self, attr: str):
        if attr in self.__extra_args:
            return self.__extra_args[attr]
        
        raise AttributeError(f"Attrbute {attr} not found")

    #############################################################################
    ## Casting methods
    #############################################################################
    def args_to_self(self):
        r'''
            Method to return the arguments necessary to reproduce this sequence

            This method returns a list and a dictionary of arguments that when used on 
            ``self.__class__``, it creates an equal sequence to ``self``. It is useful to 
            change some of the arguments or to create copies of the sequences.
        '''
        return [self._element], {"universe": self.universe, "dim": self.dim, "_extend_by_zero": self.__extend_by_zero, **self.__extra_args}

    def change_universe(self, new_universe):
        r'''
            Method that change the universe of a sequence. This can help to use the same universe in different 
            spaces or when it is required to force a specific universe.
        '''
        args, kwds = self.args_to_self()
        kwds["universe"] = new_universe
        return self.__class__(*args, **kwds)
    
    def change_dimension(self, new_dim: int, old_dims: list[int] = None, **kwds):
        r'''
            Method to create an equivalent sequence to ``self`` but with more dimensions.

            This method creates a copy of this sequences with several extra dimensions and 
            giving the old dimensions a new position.
        '''
        if not new_dim in ZZ or new_dim < self.dim:
            raise ValueError(f"[change_dim] Change dimension requires as new dimension an integer with value, at least, the old dimension ({self.dim})")
        
        if old_dims == None: old_dims = list(range(self.dim)) # old dimensions are the first new dimensions

        if not isinstance(old_dims, (list, tuple)):
            raise TypeError(f"[change_dim] The type for old dimension must be a list or tuple.")
        elif len(old_dims) != self.dim:
            raise TypeError(f"[change_dim] The number of old dimensions must be exact ({self.dim} in this case)")
        elif any(nd < 0 or nd >= new_dim for nd in old_dims):
            raise ValueError(f"[change_dim] The old dimensions must be valid new dimensions.")
        
        return self._change_dimension(new_dim, old_dims, **kwds)
    
    def _change_dimension(self, new_dim: int, old_dims: list[int], **_):
        def new_to_old(*n):
            return [n[i] for i in old_dims]
        return Sequence(lambda *n : self._element(*new_to_old(*n)), universe=self.universe, dim=new_dim, _extend_by_zero=self.__extend_by_zero, **self.__extra_args)

    def change_class(self, goal_class, **extra_info):
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
            for v in Sequence.CLASSES_GRAPH.neighbor_out_iterator(current):
                to_check.append(v); prev_class[v] = current
                if v == goal_class: break
            else: # if we do not break, we are not finished
                continue
            break # otherwise we simply get out of the loop
        else: # if we never break, it means we did not find "goal_class"
            raise TypeError(f"The class {goal_class} is not admissible for {self.__class__}")
        path_to_goal = [goal_class]; current = goal_class
        while prev_class[current] != self.__class__:
            current = prev_class[current]
            path_to_goal.insert(0, current)
        
        current = self
        for cls in path_to_goal:
            try:
                current = current._change_class(cls, **extra_info)
            except NotImplementedError:
                current = cls._change_from_class(current, **extra_info)
        return current
    
    def _change_class(self, cls, **extra_info): # pylint: disable=unused-argument
        raise NotImplementedError(f"For {self.__class__}, class {cls} not recognized.")
    
    @classmethod
    def _change_from_class(cls, sequence: Sequence, **extra_info): # pylint: disable=unused-argument
        return Sequence(lambda *n : sequence._element(*n), sequence.universe, sequence.dim, **{**sequence.__extra_args, **extra_info.get("extra_args", dict())})
        
    #############################################################################
    ## Element methods
    #############################################################################
    def element(self, *indices : int, _generic=False, _debug=False):
        if len(indices) != self.dim:
            raise TypeError(f"Invalid number of arguments for a sequence. Provided {len(indices)} but required {self.dim}")
        
        if (not _generic) and any(index < 0 for index in indices):
            return 0 if self.__extend_by_zero else NaN
        
        if not tuple(indices) in self.__CACHE_ELEMENTS:
            try:
                output = self._element(*indices)
            except ZeroDivisionError as e:
                if _debug:
                    raise e
                output = oo
            except Exception as e:
                if _debug:
                    raise e
                output = NaN

            if (not output is NaN) and (not output is oo):
                try:
                    output = self.universe(output) #pylint: disable=not-callable
                    if self.universe == SR: output = output.simplify_full()
                except Exception as e:
                    if not _generic or _debug:
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
        if result.has(NaN):
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
    ## In this section we present the main methods that are defined over sequences
    ## including, but not being limited to:
    ##   * Shifts -> shifting the arguments of the sequence
    ##   * Subsequences -> getting a subsequence indexed by other sequence, but keeping the dimension
    ##   * Slicing -> getting a part of the sequence reducing the dimension
    ##   * Interlacing -> interlacing several sequences
    ##   * Swap -> changes the order of two indices
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
        return Sequence(lambda *n : self._element(*[n[i]+shifts[i] for i in range(self.dim)]), self.universe, dim=self.dim, **self.__extra_args)

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
        if any(i < 0 or i >= self.dim or not n in ZZ for (i,n) in vals):
            raise ValueError(f"Slicing require as input a list of tuples (i,n) with valid indices `i` and integers `n`")
        values = dict(vals)

        if len(vals) >= self.dim: # we do not have a sequence but an element
            return self.element(*[values[i] for i in range(self.dim)])
        
        return self._slicing(values)
    
    def _slicing(self, values: dict[int,int]):
        def to_original_input(n: list):
            result = []; read = 0
            for i in range(self.dim):
                if i in values:
                    result.append(values[i])
                else:
                    result.append(n[read]); read += 1
            return result
        return Sequence(lambda *n: self._element(*to_original_input(n)), self.universe, self.dim - len(values), **self.__extra_args)

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
            seq = self._subsequence_input(seq)
            if not (seq is False):
                final_input.append((i, seq))

        if len(final_input) == 0: # all subsequences were trivial
            return self

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
    
    def _subsequence_input(self, input):
        if isinstance(input, (list,tuple)):
                if len(input) != 2:
                    raise TypeError(f"[subsequence - linear] Error in format for a linear subsequence. Expected a pair of integers")
                elif any((not el in ZZ) for el in input):
                    raise TypeError(f"[subsequence - linear] Error in format for a linear subsequence. Expected a pair of integers")
                a, b = input
                if a != 1 or b != 0:
                    from .element import RationalSequence
                    return RationalSequence(ZZ['n'](f"{a}*n + {b}"), universe=ZZ)
        elif input != Sequence(lambda n : n, self.universe, 1): 
            return input
        
        return False # the sequence looks like the identity
    
    def _subsequence(self, final_input: dict[int, Sequence]):
        return Sequence(
            lambda *n : self._element(*[final_input[i]._element(n[i]) if i in final_input else n[i] for i in range(self.dim)]), 
            self.universe, 
            self.dim,
            **self.__extra_args
        )

    def interlace(self, *others : Sequence, dim_to_interlace: int = 0):
        raise NotImplementedError("Method 'interlacing' not yet implemented.")

    def linear_subsequence(self, index: int, scale: int, shift: int):
        r'''Method to compute a linear subsequence of ``self``'''
        if index < 0 or index >= self.dim: raise IndexError(f"The index given must be in the dimension ({self.dim}). Got {index}")
        if scale <= 0 or not scale in ZZ: raise ValueError(f"The scale must be a positive integer. Got {scale}")

        return self.subsequence((index,(scale,shift)))
    
    def swap(self, src: int = None, dst: int = None):
        r'''
            Method that exchange the indices of a multi-dimensional sequence.
        '''
        if src == None and dst == None and self.dim == 2:
            src = 0; dst = 1
        elif src == None and dst == None:
            raise TypeError(f"If a sequence is not bi-dimensional, the arguments ``src`` and ``dst`` are mandatory")
        elif any(el < 0 or el >= self.dim for el in (src,dst)):
            raise IndexError(f"Required valid indices for swapping.")
        elif src == dst:
            return self
        elif src > dst:
            src,dst = dst,src
        return self._swap(src, dst)
    
    def _swap(self, src: int, dst: int):
        r'''
            Method that exchange the indices of a multi-dimensional sequence.

            This methods can assume that `src < dst`.
        '''
        def __swap_index(*n):
            n = list(n)
            n[src], n[dst] = n[dst], n[src]
            return tuple(n)

        return Sequence(lambda *n : self._element(*__swap_index(*n)), self.universe, self.dim, _extend_by_zero = self.__extend_by_zero, **self.__extra_args)
        
    def partial_sum(self, index: int = None) -> Sequence:
        r'''
            Method to create the partial sum sequence.

            Given a sequence `(a_n)_n`, we can always define the sequence of partial sums,
            i.e., `b_n = \sum_{i=0}^n a_i`. This method creates the corresponding 
            sequence. 

            Similarly, we can do the same for multivariate sequences. More precisely, we 
            require the index that we will compute the sum for. Iterated call of this method
            may allow summing through several dimensions of the sequence.

            *INFORMATION*: this method may be overridden for subclasses to offer a better
            structure of these sequences.

            INPUT:

            * ``index``: the index over which we will compute the partial sum sequence. ``None`` 
              is allowed for univariate sequences.

            EXAMPLES::

                sage: from pseries_basis.sequences import *
                sage: N = ExpressionSequence(var('n'), universe=QQ)
                sage: N.partial_sum() == ExpressionSequence((n*(n-1))/2, universe=QQ)
                True
                sage: N.shift().partial_sum() == ExpressionSequence((n*(n+1))/2, universe=QQ)
                True
        '''
        if self.dim == 1 and index is None: index = 0
        if not index in ZZ or index < 0 or index >= self.dim:
            raise IndexError(f"The index for partial sum must be a valid index for this sequence.")
        
        if self.dim > 1:
            def _partial_sum(*n):
                bf = list(n[:index]); lt = list(n[index+1:])
                return sum(self(bf+[i]+lt) for i in range(n[index]))
        else:
            def _partial_sum(n):
                return sum(self(i) for i in range(n))
            
        return Sequence(_partial_sum, self.universe, self.dim, _extend_by_zero=self.__extend_by_zero, **self.__extra_args)
    
    def partial_prod(self, index: int = None) -> Sequence:
        r'''
            Method to create the partial product sequence.

            Given a sequence `(a_n)_n`, we can always define the sequence of partial products,
            i.e., `b_n = \prod_{i=0}^n a_i`. This method creates the corresponding 
            sequence. 

            Similarly, we can do the same for multivariate sequences. More precisely, we 
            require the index that we will compute the product for. Iterated call of this method
            may allow multiplying through several dimensions of the sequence.

            *INFORMATION*: this method may be overridden for subclasses to offer a better
            structure of these sequences.

            INPUT:

            * ``index``: the index over which we will compute the partial product sequence. ``None`` 
              is allowed for univariate sequences.

            EXAMPLES::

                sage: from pseries_basis.sequences import *
                sage: N = ExpressionSequence(var('n'), universe=QQ)
                sage: N.shift().partial_prod() == Factorial
                True
        '''
        if self.dim == 1 and index is None: index = 0
        if not index in ZZ or index < 0 or index >= self.dim:
            raise IndexError(f"The index for partial sum must be a valid index for this sequence.")
        
        if self.dim > 1:
            def _partial_prod(*n):
                bf = list(n[:index]); lt = list(n[index+1:])
                return prod((self(bf+[i]+lt) for i in range(n[index])), z=self.universe.one())
        else:
            def _partial_prod(n):
                return prod((self(i) for i in range(n)), z=self.universe.one())
            
        return Sequence(_partial_prod, self.universe, self.dim, _extend_by_zero=self.__extend_by_zero, **self.__extra_args)

    #############################################################################
    ## Arithmetic methods
    #############################################################################
    def __mul__(self, other): # overrides multiplication as map to set it as element.
        return Element.__mul__(self, other)
    def composition(self, other): # implements the multiplication as a map (i.e., composition)
        return self._composition(other)

    def __pow__(self, power: int):
        if not power in ZZ or power < 0:
            raise ValueError("Power must be a non-negative integer for Sequences")
        if power == 0:
            return ConstantSequence(1, self.universe, self.dim)
        elif power == 1:
            return self
        else:
            p1 = power//2; p2 = p1 + power%2
            return (self**p1)*(self**p2)

    def __coerce_into_common_class__(self, other: Sequence):
        r'''We assume ``other`` is in the same parent as ``self``. Hence it is a :class:`Sequence`'''
        common_class = Sequence.MinimalCommonClass(self.__class__, other.__class__)
        all_info = self.extra_info(); all_info.update(other.extra_info())
        self_casted = self.change_class(common_class, **all_info)
        other_casted = other.change_class(common_class, **all_info)

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
        if not self.universe.is_field():
            self = self.change_universe(self.universe.fraction_field())
            other = other.change_universe(other.universe.fraction_field())
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
        return Sequence(lambda *n : (-1)*self._element(*n), self.universe, dim = self.dim, **self.__extra_args) 
    
    ## Final operation method for the basic sequence type
    def _final_add(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) + other._element(*n), self.universe, self.dim, **{**self.__extra_args, **other.__extra_args})
    def _final_sub(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) - other._element(*n), self.universe, self.dim, **{**self.__extra_args, **other.__extra_args})
    def _final_mul(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) * other._element(*n), self.universe, self.dim, **{**self.__extra_args, **other.__extra_args})
    def _final_div(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) / other._element(*n), self.universe, self.dim, **{**self.__extra_args, **other.__extra_args})
    def _final_mod(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) % other._element(*n), self.universe, self.dim, **{**self.__extra_args, **other.__extra_args})
    def _final_floordiv(self, other:Sequence) -> Sequence:
        return Sequence(lambda *n: self._element(*n) // other._element(*n), self.universe, self.dim, **{**self.__extra_args, **other.__extra_args})
    
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
        first_terms = product(range(order), repeat=self.dim) if self.dim > 1 else range(order)
        if isinstance(self.universe, SequenceSet):
            return all(self(term).almost_zero(order) for term in first_terms)
        else:
            return all(bool(self(term) == 0) for term in first_terms)

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
        first_terms = product(range(order), repeat=self.dim) if self.dim > 1 else range(order)
        if isinstance(self.universe, SequenceSet): # equality if the codomain are more sequences
            return all(self(term).almost_equals(other(term), order) for term in first_terms)
        else: # equality when elements lied in other ring
            return all(bool(self(term) == other(term)) for term in first_terms)

    def __eq__(self, other: Sequence) -> bool:
        r'''Checking for partial equality'''
        if not isinstance(other, Sequence):
            try:
                R = pushout(self.parent(), other.parent())
                self = R(self); other = R(other)
                return self == other
            except:
                return False
        return self.almost_equals(other, order=Sequence.EQUALITY_BOUND)
    
    def __ne__(self, other) -> bool:
        return not (self == other)
    
    #############################################################################
    ## Other methods
    #############################################################################
    def is_polynomial(self) -> bool:
        r'''Method to check if a sequence is polynomial'''
        try:
            expr = SR(self.generic()).simplify_full()
            return all(expr.is_polynomial(x) for x in expr.variables())
        except:
            return False
    def as_polynomial(self) -> Sequence:
        r'''Method to cast a sequence to a polynomial sequence'''
        if self.is_polynomial():
            from .element import RationalSequence
            return RationalSequence(SR(self.generic()).simplify_full(), self.generic().variables(), self.universe)
        else:
            raise ValueError(f"{self} is not a polynomial sequence.")
    def is_rational(self) -> bool:
        r'''Method to check if a sequence is rational'''
        try:
            return SR(self.generic()).simplify_full().is_rational_expression()
        except:
            return False
    def as_rational(self) -> Sequence:
        r'''Method to cast a sequence to a rational sequence'''
        if self.is_rational():
            from .element import RationalSequence
            return RationalSequence(SR(self.generic()).simplify_full(), self.generic().variables(), self.universe)
        else:
            raise ValueError(f"{self} is not a polynomial sequence.")
        
    def is_hypergeometric(self, index: int = None) -> tuple[bool, Sequence]:
        r'''Method to check if a sequence is hypergeometric or not'''
        if index == None:
            if self.dim == 1:
                index = 0
            else:
                raise TypeError("Sequence.is_hypergeometric() missing 1 required positional argument: 'index'")
        quotient = self.shift(tuple([1 if i == index else 0 for i in range(self.dim)])) / self
        if quotient.is_rational():
            return True, quotient.as_rational()
        elif quotient.is_constant():
            return True, quotient.as_constant()
        return False, None

    def is_constant(self, bound: int = None) -> bool:
        return all(self(el) == self(self.dim*[0]) for el in product(range(bound if bound != None else self.EQUALITY_BOUND), repeat=self.dim))
    
    def as_constant(self, bound = None) -> ConstantSequence:
        if self.is_constant(bound):
            return ConstantSequence(self(self.dim*[0]), self.universe, self.dim, _extend_by_zero = self.__extend_by_zero, **self.__extra_args)
        else:
            raise ValueError(f"{self} is not a constant sequence.")
        

    #############################################################################
    ## Representation methods
    #############################################################################
    def __repr__(self) -> str:
        if self.dim == 1:
            return f"Sequence over [{self.universe}]: ({self[0]}, {self[1]}, {self[2]},...)"
        else:
            return f"Sequence with {self.dim} variables over [{self.universe}]"

class ConstantSequence(Sequence):
    r'''
        Class representing constant elements in the Sequence ring.

        This class is the entry-point from other SageMath structures to the sequence setting. Namely, 
        if `x` is an element in a universe ring `R` for a sequence set (see class :class:`SequenceSet`),
        then it can represent the constant sequence `(x, x, x, \ldots)`.

        This class is, hence, at the bottom of the Sequence hierarchy. This class has no default tranfromation to any other type
        of sequences and all other sequences types **must** implement a tranformation from :class:`ConstantSequence`.
    '''
    def __init__(self, value, universe=None, dim: int = 1, *, _extend_by_zero=True, **kwds):
        super().__init__(None, universe, dim, _extend_by_zero=_extend_by_zero, **kwds)
        self.__value = self.universe(value)

    def args_to_self(self):
        _, args_sequence = super().args_to_self()
        return [self.__value], args_sequence
    
    def _change_dimension(self, new_dim: int, old_dims: list[int], **_):
        return ConstantSequence(self.__value, self.universe, new_dim, **self.extra_info()["extra_args"])
    
    def _change_class(self, cls, **extra_info): # pylint: disable=unused-argument
        raise NotImplementedError(f"Class {cls} not recognized from a ConstantSequence")
    @classmethod
    def _change_from_class(self, sequence: Sequence, **extra_info): # pylint: disable=unused-argument
        raise NotImplementedError(f"Class {sequence.__class__} not recognized from ConstantSequence")

    def _neg_(self) -> ConstantSequence:
        return ConstantSequence(-self.__value, self.universe, self.dim, **self.extra_info()["extra_args"])
    
    def _final_add(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value + other.__value, self.universe, self.dim, **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]})
    def _final_sub(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value - other.__value, self.universe, self.dim, **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]})
    def _final_mul(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value * other.__value, self.universe, self.dim, **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]})
    def _final_div(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value / other.__value, self.universe, self.dim, **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]})
    def _final_mod(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value % other.__value, self.universe, self.dim, **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]})
    def _final_floordiv(self, other: ConstantSequence) -> ConstantSequence:
        return ConstantSequence(self.__value // other.__value, self.universe, self.dim, **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]})

    def _element(self, *_: int):
        return self.__value
    def _shift(self, *_: int):
        return self
    def _subsequence(self, _: dict[int, Sequence]):
        return self
    def _slicing(self, values: dict[int, int]):
        if len(values) >= self.dim:
            return self.__value
        return ConstantSequence(self.__value, self.universe, self.dim - len(values), **self.extra_info()["extra_args"])
    def _swap(self, _: int, __: int):
        return self

from sage.categories.all import Rings
_Rings = Rings.__classcall__(Rings)
class SequenceSet(Homset,UniqueRepresentation):
    r'''
        Class for the set of sequences. We implement more coercion methods to allow more operations for sequences.
    '''
    Element = Sequence

    def __init__(self, dimension, codomain, category=None):
        domain = ZZ if dimension == 1 else cartesian_product(dimension*[ZZ])
        super().__init__(domain, codomain, category=_Sets)
        if category is None:
            if self.codomain() in _CommutativeRings:
                category = _CommutativeRings
            else:
                category = _Rings
        self._refine_category_(category)

    def dimension(self):
        try:
            return len(self.domain().construction()[1])
        except TypeError:
            return 1

    ## Category and coercion methods
    def _element_constructor_(self, x, check=None, **options):
        if x in self.codomain():
            return ConstantSequence(x, self.codomain(), dim=self.dimension()) 
        elif parent(x).has_coerce_map_from(self.codomain()):
            return ConstantSequence(x, self.codomain(), dim=self.dimension()) 
        elif isinstance(x, Sequence) and x.dim == self.dimension():
            return x.change_universe(self.codomain())
        
        return super()._element_constructor_(x, check, **options)
    
    def construction(self):
        return SequenceFunctor(self.dimension()), self.codomain()

    def _coerce_map_from_(self, S):
        return pushout(self.codomain(), S.codomain() if isinstance(S, SequenceSet) else S) == self.codomain()

    def __repr__(self):
        return f"Set of Sequences from NN{'' if self.dimension()==1 else f'^{self.dimension()}'} to {self.codomain()}"
    
    ## Ring methods
    def one(self):
        return ConstantSequence(1, self.codomain(), self.dimension())
    def zero(self):
        return ConstantSequence(0, self.codomain(), self.dimension())
    
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

__all__=["Sequence", "ConstantSequence"]
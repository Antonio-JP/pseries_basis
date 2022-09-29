r'''
    Module containing methods, strucutres and functions related with Sequences.

    A sequence is a function `f: \mathbb{N} \rightarrow \mathbb{K}` where the space `\mathbb{K}` is 
    called the *universe* of the sequence. The package :mod:`pseries_basis` is oriented towards 
    formal power series, which are equivalent to sequences. 
    
    This module will create a framework for the whole package to manage, access and manipulate sequences
    in a generic fashion.
'''
from abc import ABCMeta, abstractmethod
from functools import cache

from sage.categories.pushout import CoercionException, pushout

class Sequence(metaclass=ABCMeta):
    r'''
        Main calss for sequences. It defines a universe for its elements and a general interface to access the sequence.
    '''
    def __init__(self, universe):
        self.__universe = universe

    @property
    def universe(self): 
        r'''
            Attribute with the common parent for all the elements of this sequence
        '''
        return self.__universe

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

    def __call__(self, input):
        return self.element(input)

    @abstractmethod
    def element(self, index : int):
        pass

    def almost_zero(self, order=10):
        r'''
            Method that checks if a sequence is zero at the beginning.

            This method receives a number of elements to be checked and then it proceeds to check if all the first 
            elements of the sequence are equal to zero. This is helpful in some sequence to guarantee that the whole 
            sequence is exactly zero.
        '''
        return all(self[i] == 0 for i in range(order))

class LambdaSequence(Sequence):
    r'''
        Simplest implementation of :class:`Sequence`.

        This class computes a sequence by calling a function with some integer coefficient. It does not check whether this 
        defining function will always work. This class implements some basic arithmetic for sequences, simplifying 
        their use.

        INPUT:

        * ``func``: a callable that defines the sequence as their output with natural input.
        * ``universe``: a parent structure where all the elements of the sequence will be (a specific casting is performed)

        EXAMPLES:

            sage: from pseries_basis.sequences import LambdaSequence
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
    def __init__(self, func, universe):
        super().__init__(universe)

        self.__func = func

    @cache
    def element(self, index : int):
        if self.universe is None: # we allow None universe
            return self.__func(index)
        return self.universe(self.__func(index))

    # basic arithmethic methods
    def __add__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented

        universe = pushout(self.universe, other.universe)
        return LambdaSequence(lambda n : self[n] + other[n], universe)

    def __sub__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented

        universe = pushout(self.universe, other.universe)
        return LambdaSequence(lambda n : self[n] - other[n], universe)

    def __mul__(self, other):
        if isinstance(other, Sequence):
            universe = pushout(self.universe, other.universe)
            return LambdaSequence(lambda n : self[n] * other[n], universe)
        else:
            try:
                universe = pushout(self.universe, other.parent())
                return LambdaSequence(lambda n : self[n] * other, universe)
            except (AttributeError,CoercionException):
                return NotImplemented
        
    def __neg__(self):
        return LambdaSequence(lambda n : -self[n], self.universe)
        
    # reverse arithmethic methods
    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return self.__sub__(other)
    def __rmul__(self, other):
        return self.__mul__(other)



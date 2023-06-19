r'''
    Module containing examples of sequences that can be useful for the user.

    This module contains several common examples for sequences that can be reuse further in the future.
'''

from .base import Sequence

from functools import lru_cache
from sage.all import binomial, factorial, parent, ZZ
from sage.categories.pushout import pushout

Factorial = Sequence(factorial, ZZ, 1)
Binomial = Sequence(binomial, ZZ, 2)

def Fibonacci(a = 1, b = 1):
    r'''
        Method that defines the Fibonacci sequence for a given set of initial values.

        The universe of the sequence is defined on the fly depending on the given values.
    '''
    universe = pushout(ZZ,pushout(parent(a), parent(b)))
    @lru_cache(maxsize=None)
    def __fib(n: int): return __fib(n-1) + __fib(n-2) if n > 1 else a if n == 0 else b

    return Sequence(__fib, universe, 1)
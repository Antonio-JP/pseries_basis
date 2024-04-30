r'''
    Module containing examples of sequences that can be useful for the user.

    This module contains several common examples for sequences that can be reuse further in the future.
'''

from .base import Sequence
from .qsequences import QSequence, QPower

from functools import lru_cache, reduce
from sage.categories.pushout import pushout
from sage.functions.other import binomial, factorial
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ
from sage.structure.element import parent
from sage.symbolic.constants import NaN

###############################################################################################################
###
### CLASSICAL SEQUENCES
###
###############################################################################################################
Factorial = Sequence(factorial, ZZ, 1)
Binomial = Sequence(binomial, ZZ, 2)

def Fibonacci(a = 1, b = 1):
    r'''
        Method that defines the Fibonacci sequence for a given set of initial values.

        The universe of the sequence is defined on the fly depending on the given values.
    '''
    universe = pushout(ZZ,pushout(parent(a), parent(b)))
    
    @lru_cache(maxsize=None)
    def __fib(n: int): 
        if n not in ZZ or n < 0:
            return NaN
        return __fib(n-1) + __fib(n-2) if n > 1 else a if n == 0 else b

    return Sequence(__fib, universe, 1)

###############################################################################################################
###
### Q-SEQUENCES
###
###############################################################################################################
from sage.combinat.q_analogues import q_binomial, q_int, q_pochhammer, q_factorial

__Rq = PolynomialRing(QQ, "q").fraction_field()
__q = __Rq.gens()[0]

Qn = QPower(1, __Rq, q=__q)
Q_int = QSequence(lambda n : q_int(n, q=__q), __Rq, 1, q=__q)
Q_factorial = QSequence(lambda n : q_factorial(n, q=__q), __Rq, 1, q=__q)
@lru_cache(maxsize=256)
def Q_binomial_type(a:int = 1, b: int = 0, c: int = 0, r : int = 0, s: int = 1, t: int = 0, e: int = 1):
    r'''
        Creates the `q`-sequence ``QBinomial[a*n + b*k + c, r*n + s*k + t; q**e]
    '''
    R = reduce(lambda p, q : pushout(p,q), [__Rq] + [parent(el) for el in (a,b,c,r,s,t)])
    a,b,c,r,s,t = [R(el) for el in (a,b,c,r,s,t)]
    if e not in ZZ:
        raise TypeError(f"[q_binomial] The exponent for `q` must be an integer")
    e = ZZ(e)

    return QSequence(
        lambda n, k: q_binomial(a*n + b*k + c, r*n + s*k + t, __q**e),
        universe=R,
        dim = 2,
        q = __q
    )
Q_binomial = Q_binomial_type()

def Q_pochhammer(a=__q, q=__q):
    r'''
        Builds the sequence `(a;q)_n` as a `q`-sequence.
    '''
    # We compute the corresponding pochhammer sequence
    R = pushout(a.parent(), __Rq) if a not in __Rq else __Rq
    a, q = R(a), R(q)
    return QSequence(lambda n: q_pochhammer(n, a, q=q), R, 1, q=__q)


__all__ = [
    "Factorial", "Binomial", "Fibonacci",
    "QPower", "Qn", "Q_int", "Q_factorial", "Q_binomial", "Q_pochhammer"
]
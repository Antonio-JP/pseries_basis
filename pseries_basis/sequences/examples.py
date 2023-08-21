r'''
    Module containing examples of sequences that can be useful for the user.

    This module contains several common examples for sequences that can be reuse further in the future.
'''

from .base import Sequence
from .qsequences import QSequence

from functools import lru_cache, reduce
from sage.all import binomial, factorial, NaN, parent, PolynomialRing, QQ, ZZ # pylint: disable=no-name-in-module
from sage.categories.pushout import pushout

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
        if not n in ZZ or n < 0:
            return NaN
        return __fib(n-1) + __fib(n-2) if n > 1 else a if n == 0 else b

    return Sequence(__fib, universe, 1)


###############################################################################################################
###
### Q-SEQUENCES
###
###############################################################################################################
from sage.combinat.q_analogues import q_binomial, q_int, q_pochhammer, q_factorial
from .qsequences import logq

__Rq = PolynomialRing(QQ, "q").fraction_field(); __q = __Rq.gens()[0]

Qn = QSequence(lambda qn : qn, __Rq, 1, q=__q)
Q_int = QSequence(lambda qn : q_int(logq(qn, __q), q=__q), __Rq, 1, q=__q)
Q_factorial = QSequence(lambda qn : q_factorial(logq(qn, __q), q=__q), __Rq, 1, q=__q)
@lru_cache(maxsize=256)
def Q_binomial_type(a:int = 1, b: int = 0, c: int = 0, r : int = 0, s: int = 1, t: int = 0, e: int = 1):
    r'''
        Creates the `q`-sequence ``QBinomial[a*n + b*k + c, r*n + s*k + t; q**e]
    '''
    R = reduce(lambda p, q : pushout(p,q), [__Rq] + [parent(el) for el in (a,b,c,r,s,t)])
    a,b,c,r,s,t = [R(el) for el in (a,b,c,r,s,t)]
    if not e in ZZ:
        raise TypeError(f"[q_binomial] The exponent for `q` must be an integer")
    e = ZZ(e)

    return QSequence(
        lambda qn, qk: q_binomial(
            a*logq(qn, __q) + b*logq(qk, __q) + c, 
            r*logq(qn, __q) + s*logq(qk, __q) + t, 
            __q**e
        ),
        universe=R,
        dim = 2,
        q = __q
    )

Q_binomial = Q_binomial_type()
def Q_pochhammer(a=None):
    # If 'a' is not given, we set it to `q`
    a = __q if a is None else a
    # We compute the corresponding pochhammer sequence
    R = pushout(a.parent(), __Rq) if not a in __Rq else __Rq
    a = R(a); q = R(__q)
    return QSequence(lambda qn: q_pochhammer(logq(qn, q), a, q=q), R, 1, q=q)


__all__ = [
    "Factorial", "Binomial", "Fibonacci",
    "Qn", "Q_int", "Q_factorial", "Q_binomial", "Q_pochhammer"
]
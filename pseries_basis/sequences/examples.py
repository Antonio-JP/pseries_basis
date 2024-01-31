r'''
    Module containing examples of sequences that can be useful for the user.

    This module contains several common examples for sequences that can be reuse further in the future.
'''

from .base import Sequence
from .qsequences import QSequence, QRationalSequence

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

__Rq = PolynomialRing(QQ, "q").fraction_field(); __q = __Rq.gens()[0]

def QPower(base = __Rq, q = __q, power: int = 1, exponent: int = 1):
    if power <= 0: raise ValueError(f"Incorrect value for power of `q` ({power})")

    ## We need that exponent divides power so we can split the exponential into two exponents
    if not power%exponent == 0: raise ValueError(f"Incorrect relation between exponent and for power of `q` ({power=}, {exponent=})")

    name_qn = f"{q}_{f'{exponent}' if exponent != 1 else ''}n"
    R = PolynomialRing(base.fraction_field(), name_qn)
    qn = R.gens()[0]
    return QRationalSequence(qn**(power//exponent), variables=[name_qn], universe=base, q=q, power=exponent)

Qn = QPower()
Q_int = QSequence(lambda n : q_int(n, q=__q), __Rq, 1, q=__q)
Q_factorial = QSequence(lambda n : q_factorial(n, q=__q), __Rq, 1, q=__q)
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
    R = pushout(a.parent(), __Rq) if not a in __Rq else __Rq
    a = R(a); q = R(q)
    return QSequence(lambda n: q_pochhammer(n, a, q=q), R, 1, q=__q)


__all__ = [
    "Factorial", "Binomial", "Fibonacci",
    "QPower", "Qn", "Q_int", "Q_factorial", "Q_binomial", "Q_pochhammer"
]
r'''
    Module with the basic classes for implementing `q`-series.
'''
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

from collections.abc import Callable

from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

from ..polynomial.factorial import FactorialBasis
from ..psbasis import PSBasis, Compatibility
from ..sequences.base import Sequence
from ..sequences.examples import Qn, Q_binomial_type, q_binomial
from ..sequences.qsequences import QRationalSequence, is_QSequence, QPower

#######################################################################################################
### Classes for some basis of Q-series
#######################################################################################################
def QBasis(sequence: Callable[[int], Sequence], universe = None, *, q="q", _extend_by_zero=False) -> PSBasis:
    aux = PSBasis(sequence, universe)
    q = aux.base(q)
    return PSBasis(sequence, universe, _extend_by_zero=_extend_by_zero, q=q)

def QFactorialBasis(ak: Sequence, bk: Sequence, universe = None, *, q, e: int=1, as_2seq=None):
    ## We check the arguments for the factorial coefficients
    if not isinstance(ak, Sequence) or not is_QSequence(ak):
        raise TypeError(f"[QFactorialBasis] The argument ak must be a q-sequence")
    if not isinstance(bk, Sequence) or not is_QSequence(bk):
        raise TypeError(f"[QFactorialBasis] The argument ak must be a q-sequence")
    
    # We compare the values for q
    if ak.q != bk.q: 
        raise ValueError(f"[QFactorialBasis] The q-values do not coincide: {ak.q} != {bk.q}")
    
    if q is None: 
        q = ak.q
    elif isinstance(q, str) and str(ak.q) != q: 
        raise ValueError(f"[QFactorialBasis] The q-values do not coincide: {q} != {ak.q}")
    elif q != ak.q:
        raise ValueError(f"[QFactorialBasis] The q-values do not coincide: {q} != {ak.q}")
    
    ## Creating the sequence q^(en) which base the factorial basis
    beta = QPower(e, universe, q=ak.q)
    name_variable = f"{q}_{e}n" if e > 1 else f"{q}__{-e}" if e < -1 else f"{q}__n" if e == -1 else f"{q}_n"
    return FactorialBasis(ak, bk, universe, beta=(name_variable, beta), gamma=(f"{q}_k", QPower(1, universe, q=ak.q)), as_2seq=as_2seq, q=ak.q)

def QBinomialBasis(a: int = 1, c: int = 0, t: int = 0, e: int = 1, universe = None, *, q = "q"):
    r'''
        Factory of `q`-binomial basis of generic form.
        
        This method creates a `q`-binomial basis where the `k` element of the basis is the sequence
        
        .. MATH::
        
            \left[\begin{array}{c}an+c\\k+t\right]_{q^e}
            
        These bases are always compatible with `E: n \mapsto n+1` and the multiplication by `q^{aen}`.
        This method guarantees the input `a`, `c`, `t` and `e` are of correct form, build the `QBasis` corresponding
        to it and include both compatibilities automatically.
        
        INPUT:
        
        * ``a``: the integer value for `a`.
        * ``c``: the integer value for `c`.
        * ``t``: the integer value for `t`.
        * ``e``: the integer value for `e`.
        * ``universe``: decides the inner universe of the sequence. It must include the value of ``q``.
        * ``q``: the name or value for the `q`.
        * ``q_n``: the name of the operator that will be use for the multiplication by `q^{aen}`. 
    '''
    if a not in ZZ or a < 0:
        raise TypeError(f"[QBinomialBasis] The value for the parameter `a` must be a natural number.")
    if c not in ZZ or c < 0:
        raise TypeError(f"[QBinomialBasis] The value for the parameter `c` must be a natural number.")
    if t not in ZZ or t < 0:
        raise TypeError(f"[QBinomialBasis] The value for the parameter `t` must be a natural number.")
    if e not in ZZ or e <= 0:
        raise TypeError(f"[QBinomialBasis] The value for the parameter `b` must be a natural number.")
    a, c, t, e = ZZ(a), ZZ(c), ZZ(t), ZZ(e)

    universe = Qn.universe if universe is None else universe

    logger.debug(f"[QBinomialBasis] Creating the basis with {a=}, {c=}, {t=}, {e=}")
    q = universe(q)
    R = PolynomialRing(universe, "q_k")
    q_k = R.gens()[0]

    ak = QRationalSequence(-(q**(e*(c-t)))/(q_k**e*(1-q_k**e*q**(e*t+e))), [q_k], universe=universe, q=q)
    bk = QRationalSequence((1)/(1-q_k**e*q**(e*t+e)), [q_k], universe=universe, q=q)

    basis = QFactorialBasis(ak, bk, universe=universe, q=q, e=a*e, as_2seq=Q_binomial_type(a=a,c=c,t=t,e=e).swap(0,1))

    ## Compatibility with the `E: n -> (n+1)`
    basis.set_homomorphism(
        "E", 
        Compatibility(
            [[QRationalSequence(q_binomial(a, a-i, 1/q**e)*q_k**(e*(a-i)), variables=[q_k], universe = basis.base, q = q)
                for i in range(a, -1, -1)]],
            a, 0, 1
        ), 
        True)
    
    ## Computing the quasi_triangular sequence
    if c >= t:
        basis._PSBasis__quasi_triangular = Sequence(lambda n : a*n + c - t, ZZ)
    
    return basis

def QPowerBasis(a: int = 1, universe=None, *, q= "q", q_n = None):
    r'''
        Method to create a `q`-basis with `q`-powers as its elements.

        The elements of the basis will be `P_k(n) = q^{ank}`. These basis are 
        compatible with the multiplication by `q^{an}` and the shift in `n`.

        The name for the shift in `n` will be coded as "E". The name for the 
        multiplication operator is given by ``q_n``. If not, we will use ``f"q_{a}n"``.

        INPUT:

        * ``a``: a positive integer defining the parameter `a`. 
        * ``universe``: desired base universe for the basis. It must includes `q`.
        * ``q``: name for the `q` that will be used.
        * ``q_n`` (optional): name for the multiplication by `q^{an}`.

        TODO: add examples
    '''
    if a not in ZZ or a < 0:
        raise TypeError(f"[QPowerBasis] The value for the parameter `a` must be a natural number.")
    a = ZZ(a)

    universe = universe if universe is not None else Qn.universe
    q = universe(q)
    R = PolynomialRing(universe, "q_k")
    q_k = R.gens()[0]

    basis = QFactorialBasis(
        QRationalSequence(ZZ(1), ["q_k"], universe, q=q), #a_k
        QRationalSequence(ZZ(0), ["q_k"], universe, q=q), #b_k
        universe, q=q, e=a, 
        as_2seq = lambda k : Qn**(a*k)
    )
    
    ## Creating the compatibilities
    p = QRationalSequence(q_k**a, variables=[q_k], universe=basis.base, q = q)
    basis.set_homomorphism("E", Compatibility([[p]], 0, 0, 1), True)

    return basis

__all__ = [
   "QBasis", "QFactorialBasis", "QBinomialBasis", "QPowerBasis"
]
r'''
    Sage package for Factorial Series Basis.

    A factorial basis is a specific type of Sequences basis where the elements
    are defined using a recurrence of order 1. This can be seen also as a specific
    type of hypergeometric sequence of sequences.

    More precisely, a factorial basis is a basis of sequences `B = \{P_k(n)\}` where 
    the `k`-th element is a polynomial sequence of degree `k` such that 

    .. MATH::

        P_{k+1}(n) = (a_{k}n + b_{k})P_{k}.

    This lead to specific types of compatibility as sequences as described in :doi:`10.1016/j.jsc.2022.11.002`.
    For example, the "multiplication by `n`" operation is always comaptible with a factorial basis since:

    .. MATH::

        nP_k = \frac{1}{a_k}P_{k+1}(n) - \frac{b_k}{a_k}P_k.

    This module includes the main class for working with factorial basis, all falling down 
    into the two defining sequences `a_k` and `b_k` (see class :class:`FactorialBasis`). This 
    class will be then created specifically for other types of examples, such as the power basis 
    `\{n^k\}`, the binomial basis `\left\{\binom{n}{k}\right\}` and falling-type basis.
'''

from functools import lru_cache
from sage.all import PolynomialRing, SR
from ..sequences.base import Sequence
from ..sequences.element import ExpressionSequence, RationalSequence
from ..psbasis_new import PSBasis, Compatibility

class FactorialBasis(PSBasis):
    r'''
        Class representing a Factorial Basis.

        A factorial basis is a specific type of Sequences basis where the elements
        are defined using a recurrence of order 1. This can be seen also as a specific
        type of hypergeometric sequence of sequences.

        More precisely, a factorial basis is a basis of sequences `B = \{P_k(n)\}` where 
        the `k`-th element is a polynomial sequence of degree `k` such that 

        .. MATH::

            P_{k+1}(n) = (a_{k}n + b_{k})P_{k}.

        This type of basis have special types of compatibilities. More precisely, they are 
        **always** comaptible with the "multiplication by `n`" operation. This is a special type
        of homomorphism, and always satisfies:

        .. MATH::

            nP_k = \frac{1}{a_k}P_{k+1}(n) - \frac{b_k}{a_k}P_k.

        Besides the sequences `(a_k)_k` and `(b_k)_k` that define the elements of the basis, 
        there are two other equivalent sequences: the root sequences and the leading coefficient sequence:

        .. MATH::

            \rho_{k+1} = \frac{-b_k}{a_k},\qquad c_k = \prod_{l=1}^k a_l.

        The root sequence `\rho_k` defines for each element the new root added to the element `P_k(n)`. On the 
        other hand, the leading coefficient sequence provides the leading coefficient of the polynomial `P_k(n)`.
        Then, there are two main criteria to determine whether a Factorial sequence is compatible with an 
        homomorphism and a derivation (see :doi:`10.1016/j.jsc.2022.11.002`, Propositions 14 and 16).
    '''
    def __init__(self, ak: Sequence, bk: Sequence, universe = None, *, _extend_by_zero=False):
        ## Treating the arguments a_k and b_k
        if not isinstance(ak, Sequence):
            if universe != None:
                ak = ExpressionSequence(SR(ak), ["k"], universe)
        if not isinstance(bk, Sequence): 
            if universe != None:
                bk = ExpressionSequence(SR(bk), ["k"], universe)
        if not isinstance(ak, Sequence) or ak.dim != 1:
            raise TypeError(f"[FactorialBasis] The element a_k must be a univariate sequence or an expression in 'k'")
        if not isinstance(bk, Sequence) or bk.dim != 1:
            raise TypeError(f"[FactorialBasis] The element a_k must be a univariate sequence or an expression in 'k'")
        
        self.__ak = ak
        self.__bk = bk
        self.__rho = -bk/ak
        self.__lc = ak.partial_prod()

        self.__poly_ring = PolynomialRing(universe, "n") # this is the polynomial ring for the elements of the sequence
        self.__gen = self.__poly_ring.gens()[0]

        @lru_cache
        def __get_element(k):
            if k < 0: return 0
            elif k == 0: return 1
            else: return (self.ak(k-1)*self.__gen + self.bk(k-1))*__get_element(k-1)

        super().__init__(lambda k : RationalSequence(__get_element(k), ["n"], universe), universe, _extend_by_zero=_extend_by_zero)

        # We create now the compatibility with the multiplication by "n"
        self.set_compatibility("n", Compatibility([[self.rho, 1/self.ak]], 0, 1, 1), True, "any")

    @property
    def ak(self): return self.__ak #: Sequence a_k from definition of Factorial basis.
    @property
    def bk(self): return self.__bk #: Sequence b_k from definition of Factorial basis.
    @property
    def rho(self): return self.__rho #: Root sequence of a Factorial basis.
    @property
    def lc(self): return self.__lc #: Leading coefficient sequence of a Factorial basis.

    ##################################################################################
    ### TODO: def increasing_polynomial(self, *args, **kwds)
    ### TODO: def increasing_basis(self, shift: int) -> FactorialBasis
    ### TODO: def compatible_division(self, operator: str | OreOperator) -> Divisibility
    ### TODO: def matrix_ItP(self, src: element.Element, size: int) -> matrix_class
    ### TODO: def matrix_PtI(self, src: element.Element, size: int) -> matrix_class
    ### TODO: def equiv_DtC(self, compatibility: str | OreOperator | TypeCompatibility) -> TypeCompatibility
    ### TODO: def equiv_CtD(self, division: TypeCompatibility) -> TypeCompatibility

def RootSequence(rho: Sequence, lc: Sequence, universe = None, *, _extend_by_zero=False):
    ## Treating the arguments rho and lc
    if not isinstance(rho, Sequence):
        if universe != None:
            rho = ExpressionSequence(SR(rho), ["k"], universe)
    if not isinstance(lc, Sequence): 
        if universe != None:
            lc = ExpressionSequence(SR(lc), ["k"], universe)
    if not isinstance(rho, Sequence) or rho.dim != 1:
        raise TypeError(f"[FactorialBasis] The element rho must be a univariate sequence or an expression in 'k'")
    if not isinstance(lc, Sequence) or lc.dim != 1:
        raise TypeError(f"[FactorialBasis] The element lc must be a univariate sequence or an expression in 'k'")
    
    ak = lc.shift()/lc
    bk = -rho*ak
    return FactorialBasis(ak, bk, universe, _extend_by_zero=_extend_by_zero)
        

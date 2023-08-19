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
    For example, the "multiplication by `n`" operation is always compatible with a factorial basis since:

    .. MATH::

        nP_k = \frac{1}{a_k}P_{k+1}(n) - \frac{b_k}{a_k}P_k.

    This module includes the main class for working with factorial basis, all falling down 
    into the two defining sequences `a_k` and `b_k` (see class :class:`FactorialBasis`). This 
    class will be then created specifically for other types of examples, such as the power basis 
    `\{n^k\}`, the binomial basis `\left\{\binom{n}{k}\right\}` and falling-type basis.
'''
from __future__ import annotations

from functools import lru_cache, reduce
from sage.all import binomial, parent, PolynomialRing, QQ, SR, ZZ #pylint: disable=no-name-in-module
from sage.categories.pushout import pushout
from ..sequences.base import Sequence, ConstantSequence
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
        **always** compatible with the "multiplication by `n`" operation. This is a special type
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
        universe = universe if universe != None else pushout(ak.universe, bk.universe)
        
        self.__ak = ak.change_universe(universe)
        self.__bk = bk.change_universe(universe)
        self.__rho = -bk/ak
        self.__lc = ak.partial_prod()

        self.__poly_ring = PolynomialRing(universe, "n") # this is the polynomial ring for the elements of the sequence
        self.__gen = self.__poly_ring.gens()[0]

        @lru_cache
        def __get_element(k):
            if k < 0: return self.__poly_ring.zero()
            elif k == 0: return self.__poly_ring.one()
            else: return (self.ak(k-1)*self.__gen + self.bk(k-1))*__get_element(k-1) #pylint: disable=not-callable

        super().__init__(lambda k : RationalSequence(__get_element(k), ["n"], universe), universe, _extend_by_zero=_extend_by_zero)

        # We create now the compatibility with the multiplication by "n"
        self.set_compatibility("n", Compatibility([[self.rho, 1/self.ak]], 0, 1, 1), True, "any")

    def args_to_self(self):
        return [self.ak, self.bk], {"universe": self.base, "_extend_by_zero": self._Sequence__extend_by_zero}

    @property
    def ak(self): return self.__ak #: Sequence a_k from definition of Factorial basis.
    @property
    def bk(self): return self.__bk #: Sequence b_k from definition of Factorial basis.
    @property
    def rho(self): return self.__rho #: Root sequence of a Factorial basis.
    @property
    def lc(self): return self.__lc #: Leading coefficient sequence of a Factorial basis.

    ##################################################################################
    ### METHODS FROM PSBASIS
    ##################################################################################
    def _scalar_basis(self, factor: Sequence) -> FactorialBasis:
        r'''
            Creates the scaled version of a :class:`FactorialBasis`.

            EXAMPLES::

                sage: from pseries_basis.factorial.factorial_basis_new import *
                sage: from pseries_basis.sequences.examples import Factorial
                sage: FallingFactorial.lc / BinomialBasis.lc == Factorial
                True
                sage: FallingFactorial.rho == BinomialBasis.rho
                True
                sage: BinomialBasis.scalar(Factorial) == FallingFactorial
                True

            The compatibilities from the original basis are automatically created in the 
            scalared basis::

                sage: Fac_B = BinomialBasis.scalar(Factorial)
                sage: Fac_B.basic_compatibilities() == BinomialBasis.basic_compatibilities()
                True
                sage: Fac_B.compatibility("E")
                Compatibility condition (1, 0, 1) with following coefficient matrix:
                [1 1]
                sage: FallingFactorial.compatibility("E")
                Compatibility condition (1, 0, 1) with following coefficient matrix:
                [1 1]
                sage: Fac_B.compatibility("n")
                Compatibility condition (0, 1, 1) with following coefficient matrix:
                [                                    n (n + 1)*factorial(n)/factorial(n + 1)]
                sage: FallingFactorial.compatibility("n")
                Compatibility condition (0, 1, 1) with following coefficient matrix:
                [k 1]

            It seems that these two compatibilities differ. However, this is because while
            extending the compatibilities from ``BinomialBasis`` to ``Fac_B``, the generic information
            on the sequence is messed up. Let us check the equivalence of the two compatibilities::

                sage: Fac_B.compatibility("n").equiv(FallingFactorial.compatibility("n"))
                True
        '''
        new_universe = pushout(self.base, factor.universe)
        quotient = factor.shift() / factor
        output = FactorialBasis(self.ak*quotient, self.bk*quotient, universe=new_universe)
        if self._PSBasis__original_sequence != None:
            output._PSBasis__original_sequence = factor.change_dimension(2, [0], new_variables=["n"])*self._PSBasis__original_sequence
        return output


    ##################################################################################
    ### TODO: def increasing_polynomial(self, *args, **kwds)
    ### TODO: def increasing_basis(self, shift: int) -> FactorialBasis
    ### TODO: def compatible_division(self, operator: str | OreOperator) -> Divisibility
    ### TODO: def matrix_ItP(self, src: element.Element, size: int) -> matrix_class
    ### TODO: def matrix_PtI(self, src: element.Element, size: int) -> matrix_class
    ### TODO: def equiv_DtC(self, compatibility: str | OreOperator | TypeCompatibility) -> TypeCompatibility
    ### TODO: def equiv_CtD(self, division: TypeCompatibility) -> TypeCompatibility

def RootSequenceBasis(rho: Sequence, lc: Sequence, universe = None, *, _extend_by_zero=False):
    r'''
        Factory for creating a factorial basis from the root sequence and sequence of coefficients.

        INPUT:

        * ``rho``: the sequence of roots for the factorial basis.
        * ``cn``: the sequence of leading coefficients for the factorial basis.
        * ``universe``: the base universe where the :class:`PSBasis` will be created.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: RootSequenceBasis(0, 1)[:5]
            [Sequence over [Rational Field]: (1, 1, 1,...),
             Sequence over [Rational Field]: (0, 1, 2,...),
             Sequence over [Rational Field]: (0, 1, 4,...),
             Sequence over [Rational Field]: (0, 1, 8,...),
             Sequence over [Rational Field]: (0, 1, 16,...)]
    '''
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
        
def FallingBasis(a, b, c, universe = None, E: str = 'E'):
    r'''
        Factory for creating a factorial basis as a falling factorial-type.

        This class represent the FactorialBasis formed by the falling factorial basis
        for the power series ring `\mathbb{Q}[[x]]` with two extra parameters `a` and `b`:

        .. MATH::

            1,\quad (ax+b),\quad (ax+b)(ax+b-c),\quad (ax+b)(ax+b-c)(ax+b-2c),\dots

        In the case of `a = 1`, `b = 0` and `c = 0`, we have the usual power basis 
        and in the case of `a=1`, `b = 0` and `c = \pm 1` we have the falling (or
        raising) factorial basis.

        Following the notation in :doi:`10.1016/j.jsc.2022.11.002`, these basis
        have compatibilities with the multiplication by `n` (as any other factorial basis)
        and with the homomorphism `E: n \mapsto n+\frac{c}{a}`. All other compatible shifts (i.e., 
        maps `E_{\alpha}: n \mapsto n+\alpha)` are just powers of `E`.

        To be more precise, we always have:

        .. MATH::

            E P_k(n) = P_k(n + c/a) = P_k(n) + cP_{k-1}(n)

        INPUT:

        * ``a``: the natural number corresponding to the parameter `a`.
        * ``b``: the shift corresponding to the value `b`.
        * ``c``: the value for `c`
        * ``universe``: the universe where the :class:`PSBasis` will live.
        * ``E``: the name for the operator representing the shift of `n` by `c/a`. If not given, we will 
          consider "E" as default.

        OUTPUT:

        The corresponding :class:`FactorialBasis` with the given compatibilities.
    '''
    if universe is None: # we need to compute a base universe
        universe = reduce(lambda p,q: pushout(p,q), (parent(a), parent(b), parent(c)), QQ)
    
    ## We require everything to be in the universe
    a,b,c = [universe(el) for el in (a,b,c)]
    
    ak = ConstantSequence(a, universe, 1)
    R = PolynomialRing(universe, "k"); k = R.gens()[0]
    bk = RationalSequence(b - k*c, [k], universe=universe)

    output = FactorialBasis(ak, bk, universe)
    # This object already has the compatibility with "n". The shift remains
    comp = Compatibility([[ConstantSequence(c, universe, 1), ConstantSequence(1,universe, 1)]], 1, 0, 1)

    if c == 0: # no shift is compatible --> just identity
        E = "Id"
    elif a in ZZ and a > 0: # the shift by 1 is compatible
        E_base = E + f"_{abs(c)}_{abs(a)}"
        output.set_compatibility(E_base, comp, True, "homomorphism")
        comp = comp**ZZ(a)
    output.set_compatibility(E, comp, True, "homomorphism")

    ## We create the base sequence for generic purposes
    n, k, i = SR.var("n"), SR.var("k"), SR.var("i")
    output._PSBasis__original_sequence = ExpressionSequence((a*n + b - c*i).prod(i, 0, k), [k,n], universe)

    return output

def PowerTypeBasis(a = 1, b = 0, universe = None, Dn: str = 'Dn'):
    r'''
        Factory for creating power-type basis.

        This class represents the :class:`FactorialBasis` formed by the simplest basis
        for the power series: `1`, `(an+b)`, `(an+b)^2`, etc.

        Following the notation in :arxiv:`2202.05550`, we can find that these basis
        have compatibilities with the multiplication by `n` and with the derivation
        with respect to `n`.

        INPUT:

        * ``a``: the element of the value `a`.
        * ``b``: the shift corresponding to the value `b`.
        * ``universe``: the universe where the :class:`PSBasis` will live.
        * ``Dn``: the name for the operator representing the derivation by `n`. If not given, we will
          consider `Dn` as default.
        
        TODO: add examples
    '''
    output = FallingBasis(a, b, 0, universe, None)
    universe = output.base
    R = PolynomialRing(universe, "k"); k = R.gens()[0]
    output.set_compatibility(Dn, Compatibility([[RationalSequence(a*k, [k], universe), ConstantSequence(0, universe, 1)]], 1, 0, 1), True, "derivation")

    ## Creating the generic for this type of sequences
    n,k = SR.var("n"), SR.var("k")
    output._PSBasis__original_sequence = ExpressionSequence((a*n+b)**k, [k,n], universe)

    return output

def BinomialTypeBasis(a = 1, b = 0, universe = None, E : str = 'E'):
    r'''
        Factory for the generic binomial basis.

        This class represents a binomial basis with a shift and dilation effect on the
        top variable. Namely, a basis of the form

        .. MATH::

            \binom{an+b}{k},

        where `a` is a natural number and `b` is a rational number.

        In :arxiv:`2202.05550` this corresponds to `\mathfrak{C}_{a,b}`
        and it is compatible with the multiplication by `n` and by the shift operator
        `E: n \rightarrow n+1`.

        INPUT:

        * ``a``: the natural number corresponding to the value `a`.
        * ``b``: the shift corresponding to the value `b`.
        * ``universe``: the main ring where the basis of sequences is defined.
        * ``E``: the name for the operator representing the shift of `n` by `1`. If not given, we will
          consider "E" as default. The operator of shift by `1/a` will be named by adding a `_t` to the name.

        OUTPUT:

        A :class:`FactorialBasis` with the corresponding compatibilities and the binomial structure.
    '''
    if(not a in ZZ or a <= 0):
        raise ValueError("The value for 'a' must be a natural number")
    
    if universe is None: # we need to compute a base universe
        universe = reduce(lambda p,q: pushout(p,q), (parent(a), parent(b)), QQ)

    a = ZZ(a); b = universe(b)
    R = PolynomialRing(universe, "k"); k = R.gens()[0] # creating the polynomial ring for the sequences
    ak = RationalSequence(a/(k+1), [k], universe)
    bk = RationalSequence((b-k)/(k+1), [k], universe)

    output = FactorialBasis(ak, bk, universe) # this includes the compatibility with "n"

    E_comp = Compatibility([[ConstantSequence(binomial(a, -i), universe, 1) for i in range(-a, 1)]], a, 0, 1)
    output.set_compatibility(E, E_comp, True, "homomorphism")
    if a != 1:
        output.set_compatibility(E + "_t", Compatibility([[ConstantSequence(1, universe, 1), ConstantSequence(1, universe, 1)]], 1, 0, 1), True, "homomorphism")

    ## Creating the original sequence for this type
    n,k = SR.var("n"), SR.var("k")
    output._PSBasis__original_sequence = ExpressionSequence(binomial(a*n+b,k), [k,n], universe)

    return output

FallingFactorial = FallingBasis(1, 0, 1, QQ)
RaisingFactorial = FallingBasis(1, 0,-1, QQ)
PowerBasis = PowerTypeBasis(universe=QQ)
BinomialBasis = BinomialTypeBasis(universe=QQ)


        

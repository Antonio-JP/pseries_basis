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

import logging

from functools import lru_cache, reduce, cached_property
from sage.all import binomial, parent, PolynomialRing, vector, QQ, SR, ZZ #pylint: disable=no-name-in-module
from sage.categories.pushout import pushout
from sage.misc.cachefunc import cached_method #pylint: disable=no-name-in-module
from typing import Any

from ..sequences.base import Sequence, ConstantSequence
from ..sequences.element import ExpressionSequence, RationalSequence
from ..psbasis import PSBasis, Compatibility

logger = logging.getLogger(__name__)

###############################################################################
###
### FACTORIAL BASIS AND EXAMPLES
###
###############################################################################
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
    def __init__(self, ak: Sequence, bk: Sequence, universe = None, *, variable="n", seq_variable="k", other_seq: Sequence = None, _extend_by_zero=False, **kwds):
        ## Treating the arguments a_k and b_k
        if not isinstance(ak, Sequence):
            if universe != None:
                ak = ExpressionSequence(SR(ak), [seq_variable], universe)
        if not isinstance(bk, Sequence): 
            if universe != None:
                bk = ExpressionSequence(SR(bk), [seq_variable], universe)
        if not isinstance(ak, Sequence) or ak.dim != 1:
            raise TypeError(f"[FactorialBasis] The element a_k must be a univariate sequence or an expression in 'k'")
        if not isinstance(bk, Sequence) or bk.dim != 1:
            raise TypeError(f"[FactorialBasis] The element a_k must be a univariate sequence or an expression in 'k'")
        universe = universe if universe != None else pushout(ak.universe, bk.universe)
        
        self.__ak = ak.change_universe(universe)
        self.__bk = bk.change_universe(universe)
        self.__rho = -(bk/ak) 
        self.__lc = ak.partial_prod()

        self.__poly_ring = PolynomialRing(universe, variable) # this is the polynomial ring for the elements of the sequence
        self.__gen = self.__poly_ring.gens()[0]

        @lru_cache
        def __get_element(k):
            if k < 0: return self.__poly_ring.zero()
            elif k == 0: return self.__poly_ring.one()
            else: return (self.ak(k-1)*self.__gen + self.bk(k-1))*__get_element(k-1) #pylint: disable=not-callable

        sequence = other_seq if other_seq != None else lambda k : self._RationalSequenceBuilder(__get_element(k))

        super().__init__(sequence, universe, _extend_by_zero=_extend_by_zero, **kwds)

        # We create now the compatibility with the multiplication by the variable generator
        self.set_compatibility(variable, Compatibility([[self.rho, 1/self.ak]], 0, 1, 1), True, "any")

    def args_to_self(self) -> tuple[list, dict[str]]:
        return [self.ak, self.bk], {"universe": self.base, "variable": str(self.gen()), "seq_variable": str(self.ore_var()), "_extend_by_zero": self._Sequence__extend_by_zero}

    @property
    def ak(self): return self.__ak                  #: Sequence a_k from definition of Factorial basis.
    @property
    def bk(self): return self.__bk                  #: Sequence b_k from definition of Factorial basis.
    @property
    def rho(self): return self.__rho                #: Root sequence of a Factorial basis.
    @property
    def lc(self): return self.__lc                  #: Leading coefficient sequence of a Factorial basis.

    def gen(self): return self.__gen                #: Getter of the variable generator for the polynomial basis
    def poly_ring(self): return self.__poly_ring    #: Getter of the polynomial ring for the basis
        
    ##################################################################################
    ### METHODS FROM PSBASIS
    ##################################################################################
    def _scalar_basis(self, factor: Sequence) -> FactorialBasis:
        r'''
            Creates the scaled version of a :class:`FactorialBasis`.

            EXAMPLES::

                sage: from pseries_basis.polynomial.factorial import *
                sage: from pseries_basis.sequences.examples import Factorial
                sage: FallingFactorial.lc / BinomialBasis.lc == Factorial
                True
                sage: FallingFactorial.rho == BinomialBasis.rho
                True
                sage: BinomialBasis.scalar(Factorial) == FallingFactorial
                True

            The compatibilities from the original basis are automatically created in the 
            scalar basis::

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
        ## Getting the new universe and new quotient for changing the sequences
        new_universe = pushout(self.base, factor.universe)
        quotient = factor.shift() / factor

        ## Getting other arguments for the builder
        _, kwds = self.args_to_self()
        kwds["universe"] = new_universe

        ## Building the new basis
        output = FactorialBasis(self.ak*quotient, self.bk*quotient, **kwds)
        ## Creating (if was present) the original sequence
        if self._PSBasis__original_sequence != None:
            output._PSBasis__original_sequence = factor.change_dimension(2, [0], new_variables=[str(self.gen())])*self._PSBasis__original_sequence
        return output

    ##################################################################################
    ### TYPE GETTER METHODS
    ##################################################################################
    def _RationalSequenceBuilder(self, rational): 
        r'''
            Method that allows to build a rational sequence depending on the type of factorial basis
        '''
        return RationalSequence(rational, [str(self.gen())], self.base)

    ##################################################################################
    ### METHODS FOR FACTORIAL_BASIS
    ##################################################################################
    def increasing_basis(self, shift: int) -> FactorialBasis:
        r'''
            Method to obtain a `k`-th increasing basis.

            A factorial basis is defined by the first order recurrence on sequences:

            .. MATH::

                P_{k+1}(n) = \left(a_k n + b_k\right)P_k(n).

            This implies that the sequence `P_k(n)` is always a polynomial sequence of degree 
            exactly `k` and that `P_k(n)` divides (in terms of polynomial division) the following
            element of the basis. Let us consider the polynomials `Q^{(k)}_t(n)` defined by:

            .. MATH::

                Q^{(k)}_t(n) = \frac{P_{k+t}(n)}{P_k(n)}.

            It is clear by definition that `Q^{(k)}_t(n)` is a polynomial of degree exactly `t` and,
            moreover, it satisfies the following first order recurrence as sequences:

            .. MATH::

                Q^{(k)}_{t+1}(n) = \left(a_{k+t} n + b_{k+t}\right) Q^{(k)}_t(n).

            Hence the set of polynomials `\{Q^{(k)}_t(n)\}_t` is again a factorial basis. This method 
            returns this new factorial basis for ``shift`` taking the value of `k`.

            TODO: add tests
        '''
        _, self_args = self.args_to_self()
        return FactorialBasis(self.ak.shift(shift), self.bk.shift(shift),
                              universe=self_args["universe"], variable=self_args["variable"], seq_variable=self_args["seq_variable"]
        )

    ##################################################################################
    ### TODO: def increasing_polynomial(self, *args, **kwds)
    ### TODO: def compatible_division(self, operator: str | OreOperator) -> Divisibility
    ### TODO: def matrix_ItP(self, src: element.Element, size: int) -> matrix_class
    ### TODO: def matrix_PtI(self, src: element.Element, size: int) -> matrix_class
    ### TODO: def equiv_DtC(self, compatibility: str | OreOperator | TypeCompatibility) -> TypeCompatibility
    ### TODO: def equiv_CtD(self, division: TypeCompatibility) -> TypeCompatibility

def RootSequenceBasis(rho: Sequence, lc: Sequence, universe = None, *, variable="n", seq_variable="k", _extend_by_zero=False):
    r'''
        Factory for creating a factorial basis from the root sequence and sequence of coefficients.

        INPUT:

        * ``rho``: the sequence of roots for the factorial basis.
        * ``cn``: the sequence of leading coefficients for the factorial basis.
        * ``universe``: the base universe where the :class:`PSBasis` will be created.
        * ``variable`` and ``seq_variable``: see :class:`FactorialBasis` for further information.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: RootSequenceBasis(0, 1, QQ)[:5]
            [Sequence over [Rational Field]: (1, 1, 1,...),
             Sequence over [Rational Field]: (0, 1, 2,...),
             Sequence over [Rational Field]: (0, 1, 4,...),
             Sequence over [Rational Field]: (0, 1, 8,...),
             Sequence over [Rational Field]: (0, 1, 16,...)]
    '''
    ## Treating the arguments rho and lc
    if not isinstance(rho, Sequence):
        if universe != None:
            rho = ExpressionSequence(SR(rho), [seq_variable], universe)
    if not isinstance(lc, Sequence): 
        if universe != None:
            lc = ExpressionSequence(SR(lc), [seq_variable], universe)
    if not isinstance(rho, Sequence) or rho.dim != 1:
        raise TypeError(f"[FactorialBasis] The element rho must be a univariate sequence or an expression in 'k'")
    if not isinstance(lc, Sequence) or lc.dim != 1:
        raise TypeError(f"[FactorialBasis] The element lc must be a univariate sequence or an expression in 'k'")
    
    ak = lc.shift()/lc
    bk = -(rho*ak)
    return FactorialBasis(ak, bk, universe, variable=variable, seq_variable=seq_variable, _extend_by_zero=_extend_by_zero)
        
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
        output.set_homomorphism(E_base, comp, True)
        comp = comp**ZZ(a)
    output.set_homomorphism(E, comp, True)

    ## We create the base sequence for generic purposes
    n, k, i = SR(output.gen()), SR.var(output.ore_var()), SR.var("i")
    output._PSBasis__original_sequence = ExpressionSequence((a*n + b - c*i).prod(i, 0, k-1), [k,n], universe)

    ## We check is the basis is quasi-triangular
    ## roots are (nc-b)/a then we need n = (ka + b)/c
    if (c != 0) and (a%c == 0 and b%c == 0 and a//c in ZZ and b//c in ZZ) and (a//c > 0 and b//c >= 0):
        output._PSBasis__quasi_triangular = Sequence(lambda k : (k*a + b)//c, ZZ)

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
    output.set_derivation(Dn, Compatibility([[RationalSequence(a*k, [k], universe), ConstantSequence(0, universe, 1)]], 1, 0, 1), True)

    ## Creating the generic for this type of sequences
    n, k = SR(output.gen()), SR.var(output.ore_var())
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
    output.set_homomorphism(E, E_comp, True)
    if a != 1:
        output.set_homomorphism(E + "_t", Compatibility([[ConstantSequence(1, universe, 1), ConstantSequence(1, universe, 1)]], 1, 0, 1), True)

    ## Creating the original sequence for this type
    n, k = SR(output.gen()), SR.var(output.ore_var())
    output._PSBasis__original_sequence = ExpressionSequence(binomial(a*n+b,k), [k,n], universe)
    
    ## We check is the basis is quasi-triangular
    ## roots are (n-b)/a then we need n = ak + b
    if b in ZZ and b >= 0: # conditions on a are satisfied above
        output._PSBasis__quasi_triangular = Sequence(lambda k : (k*a + b), ZZ)

    return output

###############################################################################
### Examples
FallingFactorial = FallingBasis(1, 0, 1, QQ)
RaisingFactorial = FallingBasis(1, 0,-1, QQ)
PowerBasis = PowerTypeBasis(universe=QQ)
BinomialBasis = BinomialTypeBasis(universe=QQ)

###############################################################################
### 
### SIEVED BASIS AND PRODUCT BASIS
### 
###############################################################################
class SievedBasis(FactorialBasis):
    r'''
        Class for a Sieved Basis.

        A sieved basis is a factorial basis built from a finite set
        of source basis `B_i = \left(P_k^{(i)}(n)\right)_k` for `i=0,\ldots,F-1`. This is built 
        in `m` sections using a *deciding cycle*:

        .. MATH::

            (\sigma_0,\ldots,\sigma_{m-1})

        where `\sigma_i \in \{0,\ldots,F-1\}`. We can then define the `n`-th element
        of the basis with the following formula:

        .. MATH::

            Q_k(n) = \prod_{i=0}^F P_{e_i(k)}^{(i)}(n)

        where the following formula stands:
        
        * `k = lm+r`, 
        * `S_i = \# \{ j \in \{0,\ldots,m-1\}\ :\ \sigma_j = i\}`,
        * `e_i(k) = S_i l + \#\{j \in \{0,\ldots,r\}\ :\ \sigma_j = i\}`.

        If we look recursively, we can see that each element is built from the previous
        element by increasing one of the factors one degree in the corresponding basis:

        .. MATH::

            Q_k(n) = Q_{k-1}(n)\frac{P_{e_{\sigma_i}(k)}^{(\sigma_i)}(n)}{P_{e_{\sigma_i}(k)-1}^{(\sigma_i)}(n)}

        INPUT:

        * ``factors``: the basis that build the sieved basis.
        * ``cycle``: a tuple of length `m` indicating which factor use in each step.
        ??* ``init``: value for the constant element of the basis.
        ??* ``var_name``: name of the operator representing the multiplication by `x`.

        EXAMPLES::

            sage: from pseries_basis.polynomial.factorial import *
            sage: B = BinomialBasis; P = PowerBasis
            sage: B2 = SievedBasis([B,P], [0,1,1,0])
            sage: B2[:4]
            [Sequence over [Rational Field]: (1, 1, 1,...),
             Sequence over [Rational Field]: (0, 1, 2,...),
             Sequence over [Rational Field]: (0, 1, 4,...),
             Sequence over [Rational Field]: (0, 1, 8,...)]
            sage: [el.generic() for el in B2[:6]]
            [1, n, n^2, n^3, 1/2*n^4 - 1/2*n^3, 1/6*n^5 - 1/2*n^4 + 1/3*n^3]

        With this system, we can build the same basis changing the order and the values in the cycle::

            sage: B3 = SievedBasis([P,B], [1,0,0,1])
            sage: B3.almost_equals(B2, 30) # checking equality for 30 elements 
            True

        The length of the cycle is the number of associated sections::

            sage: B2.nsections
            4
            sage: SievedBasis([B,B,P],[0,0,1,2,1,2]).nsections
            6

        This basis can be use to deduce some nice recurrences for the Apery's `\zeta(2)` sequence::

            sage: #TODO b1 = FallingBasis(1,0,1); b2 = FallingBasis(1,1,-1); n = b1.n()
            sage: #TODO B = SievedBasis([b1,b2],[0,1]).scalar(1/factorial(n))

        This basis ``B`` contains the elements 

        .. MATH::

            \begin{matrix}
            \binom{n + k}{2k}\ \text{if }k\equiv 0\ (mod\ 2)\\
            \binom{n + k}{2k+1}\ \text{if }k\equiv 1\ (mod\ 2)
            \end{matrix}

        We first extend the compatibility with `E: x\mapsto x+1` by guessing and then we compute the sieved basis
        with the binomial basis with the cycle `(1,0,1)`::

            sage: #TODO B.set_endomorphism('E', guess_compatibility_E(B, sections=2))
            sage: #TODO B2 = SievedBasis([BinomialBasis(), B], [1,0,1])

        Now the basis ``B2`` is formed in 3 sections by the following elements:

        .. MATH::

            \begin{matrix}
                \binom{n}{k}\binom{n+k}{2k}\ \text{if }k\equiv 0\ (mod\ 3)\\
                \binom{n}{k}\binom{n+k}{2k+1}\ \text{if }k\equiv 1\ (mod\ 3)\\
                \binom{n}{k+1}\binom{n+k}{2k+1}\ \text{if }k\equiv 2\ (mod\ 3)
            \end{matrix}

        We can check that `B2` is compatible with the multiplication by `x` and with 
        the endomorphism `E`::

            sage: #TODO a,b,m,alpha = B2.compatibility('x')
            sage: #TODO Matrix([[alpha(i,j,B2.n()) for j in range(-a,b+1)] for i in range(m)]) ## output: [      n 2*n + 1], [      n   n + 1], [ -n - 1 2*n + 2]
            sage: #TODO B2.recurrence('x') ## output: [        n         0 (2*n)*Sni], [(2*n + 1)         n         0], [        0   (n + 1)  (-n - 1)]
            sage: #TODO a,b,m,alpha = B2.compatibility('E')
            sage: #TODO Matrix([[alpha(i,j,B2.n()) for j in range(-a,b+1)] for i in range(m)]) ## output: [                      1           (4*n - 3/2)/n                     3/2                       1], [    (n - 1/2)/(n + 1/2)         1/2*n/(n + 1/2) (3/2*n + 1/2)/(n + 1/2)                       1], [                      0                       1       (3*n + 2)/(n + 1)                       1]
            sage: #TODO B2.recurrence('E') ## output: [                    Sn + 1        (3*n + 1)/(2*n + 1)                          1], [    (8*n + 5)/(2*n + 2)*Sn (2*n + 1)/(2*n + 3)*Sn + 1          (3*n + 2)/(n + 1)], [                    3/2*Sn       (n + 1)/(2*n + 3)*Sn                          1]

        Now consider the following difference operator:

        .. MATH::

            L = (n+2)^2 E^2 - (11n^2+33n+25)E - (n+1)^3

        This operator `L` is compatible with the basis ``B2``. We can get then
        the associated recurrence matrix. Taking the first column and the GCRD
        of its elements, we can see that if a sequence `y(n)`
        that can be written in the form `y(n) = \sum_{k\geq 0}c_k\binom{n}{k}\binom{n+k}{2k}` satisfies
        that

        .. MATH::

            (k+1)^2c_{k+1} - 2(2k+1)c_k = 0.

        Doing that with the code::

            sage: #TODO from ore_algebra import OreAlgebra
            sage: #TODO R.<x> = QQ[]; OE.<E> = OreAlgebra(R, ('E', lambda p : p(x=x+1), lambda p : 0))   
            sage: #TODO L = (x+2)^2*E^2 - (11*x^2 + 33*x+25)*E - (x+1)^2 
            sage: #TODO M = B2.recurrence(L)
            sage: #TODO column = [B2.remove_Sni(M.coefficient((j,0))) for j in range(M.nrows())]
            sage: #TODO column[0].gcrd(*column[1:]) ## output: (n + 1)*Sn - 4*n - 2
    '''
    def __init__(self, 
        factors : list[FactorialBasis] | tuple[FactorialBasis], 
        cycle: list[int] |tuple[int], 
        variable="n", seq_variable="k", _extend_by_zero=False, **kwds
    ):
        ## Checking the input
        if(not type(factors) in (list,tuple)):
            raise TypeError("The factors must be either a list or a tuple")
        if(any(not isinstance(el, FactorialBasis) for el in factors)):
            raise TypeError("All the factors has to be factorial basis")

        if(not type(cycle) in (list,tuple)):
            raise TypeError("The deciding cycle must be a list or a tuple")
        cycle = [ZZ(el) for el in cycle]
        if(any(el < 0 or el > len(factors) for el in cycle)):
            raise ValueError("The deciding cycle must be composed of integers indexing the factors basis")

        ## Storing the main elements
        self.__factors = tuple(factors)
        self.__cycle = tuple(cycle)
        universe = reduce(lambda p,q: pushout(p,q), [f.base for f in self.factors])

        new_ak = Sequence(lambda k : (self.factors[self.cycle[k%self.nsections]]).ak[self.indices[k][self.cycle[k%self.nsections]]], universe)
        new_bk = Sequence(lambda k : (self.factors[self.cycle[k%self.nsections]]).bk[self.indices[k][self.cycle[k%self.nsections]]], universe)
        FactorialBasis.__init__(self, new_ak, new_bk, universe, variable=variable, seq_variable=seq_variable, _extend_by_zero=_extend_by_zero, **kwds)

        ## We reset the compatibility wrt the variable name
        try:
            self.set_compatibility(variable, self._extend_compatibility_X(), True, "any")
        except ValueError:
            logger.warning(f"[SievedBasis] Compatibility with {variable=} was not extended")

        ## We try to extend other compatibilities
        for E in self.factors[0].compatible_endomorphisms():
            if all(E in factor.compatible_endomorphisms() for factor in self.factors[1:]):
                try:
                    self.set_homomorphism(E, self._extend_compatibility_E(E), True)
                except (ValueError, NotImplementedError):
                    logger.info(f"[SievedBasis] Compatibility with endomorphism {E=} could not be extended")
        for D in self.factors[0].compatible_derivations():
            if all(D in factor.compatible_derivations() for factor in self.factors[1:]):
                try:
                    self.set_derivation(D, self._extend_compatibility_D(D), True)
                except (ValueError, NotImplementedError):
                    logger.info(f"[SievedBasis] Compatibility with endomorphism {D=} could not be extended")

        ## TODO: Fill from here
        ## 1. (DONE) Call the super method of Factorial Basis with the necessary information.
        ##    1.1. Test the creation of elements (fix tests of __init__)
        ## 2. Create the compatibility w.r.t. "n". In particular, extend "any" operators.
        ## 3. Extend "homomorphisms"
        ## 4. Extend "derivations"
        ## 5. Fix all tests of __init__
        # raise NotImplementedError(f"[SievedBasis] Initialization not yet implemented")
    
        quasi_triangular_sequences = [factor.is_quasi_triangular() for factor in self.factors]
        if all(el != None for el in quasi_triangular_sequences):
            self._PSBasis__quasi_triangular = _SievedQuasiTriangular(quasi_triangular_sequences, self.cycle)

    @cached_property
    def indices(self) -> Sequence:
        r'''
            Computes the indices of each factor in the given element.
        '''
        def _element(k):
            counts = [self.cycle.count(i) for i in range(self.nfactors)]
            K,r = k//len(self.cycle), k%len(self.cycle)
            extras = [self.cycle[:r].count(i) for i in range(self.nfactors)]
            return vector([K*c+e for c,e in zip(counts, extras)])
        return Sequence(_element, ZZ**self.nfactors, 1)
    
    @cached_method
    def division_decomposition(self, indices: tuple[int]) -> tuple[tuple, int, Any]:
        r'''
            Any combination of elements of the factors can be written as a maximal element
            in the sieved basis times a polynomial. This method computes this decomposition.

            It returns the tuple of indices, the index and the polynomial that remains
        '''
        if len(indices) != self.nfactors:
            raise TypeError("Indices must coincide with number of factors")
        if any(el < 0 for el in indices): 
            raise ValueError("Indices must be all non-negative")
        k = self.nsections*min([ind//self.cycle.count(i) for (i,ind) in enumerate(indices)])
        indices = vector(indices) # guaranteeing the vector structure
        while all(el >=0 for el in indices-self.indices[k]):
            k += 1
        k -= 1

        remaining = indices - self.indices[k]
        poly = reduce(
            lambda p, q: p*q, 
            [
                factor.increasing_basis(self.indices[k][i])[remaining[i]].generic(str(self.gen())) 
                for (i,factor) in enumerate(self.factors)
            ]
        )
        return self.indices[k], k, poly

    @property
    def factors(self) -> tuple[FactorialBasis]:
        r'''Property to get the factors of the :class:`SievedBasis`'''
        return self.__factors
    @property
    def cycle(self) -> tuple[int]:
        r'''Property to get the deciding cycle of the :class:`SievedBasis`'''
        return self.__cycle
    @property
    def nfactors(self) -> int:
        r'''
            Method to get the number of factors of the sieved basis.

            This method returns the number of factors which compose
            this :class:`SievedBasis`.

            OUTPUT:

            Number of factors of this :class:`SievedBasis`.
            
            TODO: add examples
        '''
        return len(self.factors)
    F = nfactors #: alias for the number of factors
    @property
    def nsections(self) -> int:
        r'''
            Method to get the number of sections of the sieved basis.

            This method returns the number of elements in the deciding cycle which 
            is the number of sections in which the :class:`SievedBasis` is divided.

            OUTPUT:

            Number of sections of this :class:`SievedBasis`.
            
            TODO: add examples
        '''
        return len(self.cycle)
    m = nsections #: alias for the number of sections in the cycle
    
    ###############################################################################
    ## Methods for extending compatibilities (protected)
    ###############################################################################
    def _extend_compatibility_X(self) -> Compatibility:
        r'''
            Method that extend the compatibility of multiplication by the sequence variable.

            This method uses the information in the factor basis to extend 
            the compatibility behavior of the multiplication by `x` to the 
            :class:`SievedBasis`.
        '''
        m = self.nsections; F = self.nfactors
        comps = [factor.compatibility(str(factor.gen())) for factor in self.factors]
        t = [comp.t for comp in comps]
        S = [self.cycle.count(i) for i in range(F)]
        s = [self.cycle[:i].count(self.cycle[i]) for i in range(m)]
        
        ## Computing the optimal value for the sections
        T = 1
        while(any([not T*S[i]%t[i] == 0 for i in range(F)])): T += 1 # this always terminate at most with T = lcm(t_i)
        
        a = [T*S[i]//t[i] for i in range(F)]

        new_coeffs = list()
        for i in range(m*T): # section i
            section_i = list()
            i1 = i%m; i0 = (i-i1)//m
            next = self.cycle[i1]
            t = comps[next].t
            i3 = (S[next]*i0 + s[i1])%t; i2 = (S[next]*i0+s[i1]-i3)//t
            section_i.append(comps[next][(i3,0)].linear_subsequence(0, a[next], i2))
            section_i.append(comps[next][(i3,1)].linear_subsequence(0, a[next], i2))
            new_coeffs.append(section_i)
        print(T, new_coeffs)
        return Compatibility(new_coeffs, 0, 1, m*T)
    
    def _extend_compatibility_E(self, E: str) -> Compatibility:
        r'''
            Method that extend the compatibility of an endomorphism `E`.

            This method uses the information in the factor basis to extend 
            the compatibility behavior of an endomorphism `E` to the 
            :class:`SievedBasis`.

            This method can be extended in subclasses for a different behavior.

            INPUT:

            * ``E``: name of the endomorphism to extend.

            OUTPUT:

            A tuple `(A,B,m,\alpha_{i,k,j})` representing the compatibility of ``E``
            with ``self``.
        '''
        # A, m, D = self._compatible_division_E(E)
        # n = self.n()
        # B = D(0,0,n).degree()-A

        # alphas = []
        # for i in range(m):
        #     alphas += [self.matrix_PtI(m*n-A+i,A+B+1)*vector([D(i,0,n)[j] for j in range(A+B+1)])]

        # return (A, B, m, lambda i,j,k : alphas[i][j+A](n=k))
        raise NotImplementedError(f"[SievedBasis] Extension of homomorphisms not yet implemented")

    def _extend_compatibility_D(self, D: str) -> Compatibility:
        r'''
            Method that extend the compatibility of a derivation `D`.

            This method uses the information in the factor basis to extend 
            the compatibility behavior of a derivation `D` to the 
            :class:`SievedBasis`.

            This method can be extended in subclasses for a different behavior.

            INPUT:

            * ``D``: name of the derivation to extend.

            OUTPUT:

            A tuple `(A,B,m,\alpha_{i,k,j})` representing the compatibility of ``D``
            with ``self``.
        '''
        # A, m, Q = self._compatible_division_D(D)
        # n = self.n()
        # B = max(Q(0,0,n).degree()-A,0)

        # alphas = []
        # for i in range(m):
        #     alphas += [self.matrix_PtI(m*n-A+i,A+B+1)*vector([Q(i,0,n)[j] for j in range(A+B+1)])]
        
        # return (A, B, m, lambda i,j,k : alphas[i][j+A](n=k))
        raise NotImplementedError(f"[SievedBasis] Extension of derivations not yet implemented")

    ###############################################################################
    ### Representation methods
    ###############################################################################
    def __repr__(self) -> str:
        return f"Sieved Basis {self.cycle} of the basis:" + "".join([f"\n\t- {f}" for f in self.factors])

    def _latex_(self) -> str:
        return (r"\prod_{%s}" %self.cycle)  + "".join([f._latex_() for f in self.factors])

    # def extend_compatibility_X(self) -> TypeCompatibility:
    #     r'''
    #         Method to extend the compatibility of the multiplication by `x`.

    #         This method computes the compatibility of a he multiplication by `x` over
    #         the ring `\mathbb{K}[x]`. This operator is always compatible with all 
    #         :class:`FactorialBasis`.

    #         If this method was already called (or the compatibility was found in another way)
    #         this method only returns the compatibility

    #         OUTPUT:

    #         The compatibility for the multiplication by `x` computed during this process.

    #         TODO: add examples
    #     '''
    #     X = str(self.universe.gens()[0])
    #     if(not self.has_compatibility(X)):
    #         self.set_compatibility(X, self._extend_compatibility_X())

    #     return self.compatibility(X)

    # def extend_compatibility_E(self, name: str) -> TypeCompatibility:
    #     r'''
    #         Method to extend the compatibility of an endomorphism.

    #         This method computes the compatibility of an endomorphism `L` over
    #         the ring `\mathbb{K}[x]`. Such derivation must be compatible with all the
    #         factors on the basis.

    #         If the operator `L` was already compatible with ``self``, this method does
    #         nothing.

    #         INPUT:

    #         * ``name``: name of the derivation or a generator of a *ore_algebra*
    #           ring of operators.

    #         OUTPUT:

    #         The compatibility for `L` computed during this process.

    #         WARNING:

    #         This method do not check whether the operator given is an endomorphism
    #         or not. That remains as a user responsibility.

    #         TODO: add examples
    #     '''
    #     if(not (type(name) is str)):
    #         name = str(name)

    #     if(not self.has_compatibility(name)):
    #         self.set_endomorphism(name, self._extend_compatibility_E(name))

    #     return self.compatibility(name)

    # def extend_compatibility_D(self, name: str) -> TypeCompatibility:
    #     r'''
    #         Method to extend the compatibility of a derivation.

    #         This method computes the compatibility of a derivation `L` over
    #         the ring `\mathbb{K}[x]`. Such derivation must be compatible with all the
    #         factors on the basis.

    #         If the operator `L` was already compatible with ``self``, this method does
    #         nothing.

    #         INPUT:

    #         * ``name``: name of the derivation or a generator of a *ore_algebra*
    #           ring of operators.

    #         OUTPUT:

    #         The compatibility for `L` computed during this process.

    #         WARNING:

    #         This method do not check whether the operator given is a derivation
    #         or not. That remains as a user responsibility.

    #         TODO: add examples
    #     '''
    #     if(not (type(name) is str)):
    #         name = str(name)

    #     if(not self.has_compatibility(name)):
    #         self.set_derivation(name, self._extend_compatibility_D(name))

    #     return self.compatibility(name)

    # def increasing_polynomial(self, src: element.Element, diff : int = None, dst: int = None) -> element.Element:
    #     r'''
    #         Returns the increasing factorial for the factorial basis.

    #         This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial.factorial_basis.FactorialBasis`.
    #         See method :func:`~pseries_basis.factorial.factorial_basis.FactorialBasis.increasing_polynomial` for further information 
    #         in the description or the output.

    #         As a :class:`SievedBasis` is composed with several factors, we compute the difference between each element
    #         in the factors and compute the corresponding product of the increasing polynomials. 

    #         In this case, we consider the input given by `n = kF + r` where `F` is the number of sections of the 
    #         :class:`SievedBasis` (see method :func:`nsections`).

    #         INPUT:

    #         * ``src``: either the value of `n` or a tuple with the values `(k,r)`
    #         * ``diff``: difference between the index `n` and the largest index, `m`. Must be a positive integer.
    #         * ``dst``: value for `m`. It could be either its value or a tuple `(t,s)` where `m = tF + s`.
    #     '''
    #     ## Checking the input "src"
    #     if(not type(src) in (tuple, list)):
    #         k,r = self.extended_quo_rem(src, self.nsections())
    #         if(not r in ZZ):
    #             raise ValueError("The value for the starting point must be an object where we can deduce the section")
    #     else:
    #         k, r = src

    #     ## If no diff, we use dst instead to build diff
    #     if(diff == None):
    #         if(type(dst) in (tuple, list)):
    #             dst = dst[0]*self.nsections() + dst[1]
    #         diff = dst - src
        
    #     ## Now we check the value for 'diff'
    #     if(not diff in ZZ):
    #         raise TypeError("The value of 'diff' must be an integer")
    #     diff = ZZ(diff)
    #     if(diff < 0):
    #         raise ValueError("The value for 'diff' must be a non-negative integer")
    #     if(diff == 0):
    #         return self.universe.one()

    #     if(not (k,r,diff) in self.__cached_increasing):
    #         original_index = [self.index((k,r), i) for i in range(self.nfactors())]
    #         t, s = self.extended_quo_rem(diff+r, self.nsections())
    #         end_index = [self.index((k+t, s), i) for i in range(self.nfactors())]
    #         self.__cached_increasing[(k,r,diff)] = prod(
    #             [self.factors[i].increasing_polynomial(original_index[i],dst=end_index[i]) for i in range(self.nfactors())]
    #         )
    #     return self.__cached_increasing[(k,r,diff)]

    # @cached_method
    # def increasing_basis(self, shift: int) -> SievedBasis:
    #     r'''
    #         Method to get the structure for the `n`-th increasing basis.

    #         This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial.factorial_basis.FactorialBasis`.
    #         See method :func:`~pseries_basis.factorial.factorial_basis.FactorialBasis.increasing_basis` for further information.

    #         For a :class:`SievedBasis`, the increasing basis is again a :class:`SievedBasis` of the increasing basis
    #         of its factors. Depending on the actual shift, the increasing basis may differ. Namely, if the shift is 
    #         `N = kF+j` where `F` is the number of sections of ``self`` and `B_i` are those factors, then the we can express 
    #         the increasing basis as a :class:`SievedBasis` again.

    #         INPUT:

    #         * ``shift``: value for the starting point of the increasing basis. It can be either
    #           the value for `N` or the tuple `(k,j)`.
              
    #         OUTPUT:

    #         A :class:`SievedBasis` representing the increasing basis starting at `N`.

    #         WARING: currently the compatibilities aer not extended to the increasing basis.

    #         TODO: add examples
    #     '''
    #     ## Checking the input "src"
    #     if(type(shift) in (tuple, list)):
    #         N = shift[0]*self.nsections() + shift[1]
    #     else:
    #         N = shift
    #         shift = self.extended_quo_rem(N,self.nsections())
        
    #     if((shift[1] < 0) or (shift[1] > self.nsections())):
    #         raise ValueError("The input for the shift is not correct")

    #     new_cycle = self.cycle[shift[1]:] + self.cycle[:shift[1]]
    #     indices = [self.index(shift, i) for i in range(self.nfactors())]
    #     new_basis = [self.factors[i].increasing_basis(indices[i]) for i in range(self.nfactors())]
    #     return SievedBasis(new_basis, new_cycle, var_name=str(self.universe.gens()[0]))
     
    # def compatible_division(self, operator: str | OreOperator) -> Divisibility:
    #     r'''
    #         Method to get the division of a polynomial by other element of the basis after an operator.

    #         This method *overrides* the implementation from class :class:`FactorialBasis`. See :func:`FactorialBasis.compatible_division`
    #         for a description on the output.

    #         For a :class:`SievedBasis`, since its elements are products of elements of other basis, we can compute this 
    #         division using the information in the factors of ``self``. However, we need to know how this operator
    #         acts on products distinguishing between three classes:

    #         * **Multiplication operators**: `L(f(x)) = g(x)f(x)`.
    #         * **Endomorphisms**: `L(f(x)g(x)) = L(f(x))L(g(x))`.
    #         * **Derivations**: `L(f(x)g(x)) = L(f(x))g(x) + f(x)L(g(x))`.

    #         In order to know if an operator is an *endomorphism* or a *derivation*, we check if we have extended already those 
    #         compatibilities. If we do not found them, we assume they are multiplication operators.

    #         TODO: add examples
    #     '''
    #     comp_type = self.compatibility_type(operator)
    #     if(comp_type == "der"):
    #         return self._compatible_division_D(operator)
    #     elif(comp_type == "endo"):
    #         return self._compatible_division_E(operator)
    #     else:
    #         return self._compatible_division_X(operator)

    # def _compatible_division_X(self, operator: str | OreOperator) -> Divisibility:
    #     r'''
    #         Method o compute the compatible division for multiplication operators.
    #     '''
    #     raise NotImplementedError("_compatible_division_X not implemented for Sieved Basis")

    # def _compatible_division_D(self, operator: str | OreOperator) -> Divisibility:
    #     r'''
    #         Method o compute the compatible division for derivations.
    #     '''
    #     F = self.nfactors(); m = self.nsections()
    #     comp_divisions = [self.factors[i].compatible_division(operator) for i in range(F)] # list with (A_i, t_i, D)
    #     As = [comp_divisions[i][0] for i in range(F)]; t = [comp_divisions[i][1] for i in range(F)]
    #     D = [comp_divisions[i][2] for i in range(F)]
    #     I = [self.factors[i].increasing_polynomial for i in range(F)]
    #     S = [self.appear(i) for i in range(F)]; s = lambda i,r : self.cycle[:r].count(i)

    #     ## Computing the optimal value for the sections
    #     T = 1
    #     while(any([not T*S[i]%t[i] == 0 for i in range(F)])): T += 1 # this always terminate at most with T = lcm(t_i)
    #     a = [T*S[i]//t[i] for i in range(F)]

    #     ## Computing the lower bound for the final compatibility
    #     S = [self.appear(i) for i in range(F)]; A = max(int(ceil(As[i]/S[i])) for i in range(F))
    #     b = [A*S[i] - As[i] for i in range(F)]

    #     def new_D(r,j,n):
    #         if(j != 0): raise IndexError("Division not computed for more than compatibility")
    #         r0, r1 = self.extended_quo_rem(r, m)
    #         r2, r3 = list(zip(*[self.extended_quo_rem(S[i]*r0+s(i,r1), t[i]) for i in range(F)]))

    #         return sum(
    #             D[i](r3[i], b[i], a[i]*n+r2[i]) * 
    #             prod(
    #                 I[j]((a[j]*n+r2[j])*t[j] + r3[j] - As[j] - b[j], As[j]+b[j])
    #                 for j in range(F) if i != j
    #             )
    #             for i in range(F)
    #         )
            
    #     return (m*A, m*T,new_D)

    # def _compatible_division_E(self, operator: str | OreOperator) -> Divisibility:
    #     r'''
    #         Method o compute the compatible division for endomorphisms.
    #     '''
    #     F = self.nfactors(); m = self.nsections()
    #     comp_divisions = [self.factors[i].compatible_division(operator) for i in range(F)] # list with (A_i, t_i, D)
    #     As = [comp_divisions[i][0] for i in range(F)]; t = [comp_divisions[i][1] for i in range(F)]
    #     D = [comp_divisions[i][2] for i in range(F)]
    #     S = [self.appear(i) for i in range(F)]; s = lambda i,r : self.cycle[:r].count(i)      

    #     ## Computing the optimal value for the sections
    #     T = 1
    #     while(any([not T*S[i]%t[i] == 0 for i in range(F)])): T += 1 # this always terminate at most with T = lcm(t_i)
    #     a = [T*S[i]//t[i] for i in range(F)]

    #     ## Computing the lower bound for the final compatibility
    #     S = [self.appear(i) for i in range(F)]; A = max(int(ceil(As[i]/S[i])) for i in range(F))
    #     b = [A*S[i] - As[i] for i in range(F)]

    #     def new_D(r,j,n):
    #         if(j != 0): raise IndexError("Division not computed for more than compatibility")
    #         r0, r1 = self.extended_quo_rem(r, m)
    #         r2, r3 = list(zip(*[self.extended_quo_rem(S[i]*r0+s(i,r1), t[i]) for i in range(F)]))

    #         return prod(D[i](r3[i],b[i],a[i]*n+r2[i]) for i in range(F))
            
    #     return (m*A, m*T,new_D)

    # def is_quasi_func_triangular(self) -> bool:
    #     return all(basis.is_quasi_func_triangular() for basis in self.factors)
    # def is_quasi_eval_triangular(self) -> bool:
    #     return all(basis.is_quasi_eval_triangular() for basis in self.factors)

def ProductBasis(factors: list[FactorialBasis] | tuple[FactorialBasis]) -> SievedBasis:
    r'''
        Factory for creating a special type of :class:`SievedBasis`: Product Basis.

        Namely, the `k=lm+j` element of the product of `m` basis, is the product of

        .. MATH::

            Q_k(n) = \prod_{i=1}^{j}P_{l+1}^{(j)}(n)\prod_{i=j+1}^{m}P_{l}^{(j)}(n).

        See the paper :doi:`10.1016/j.jsc.2022.11.002` for further information.

        INPUT:

        * ``factors``: list of :class:`FactorialBasis`.
        ??* ``init``: value for the constant element of the basis.
        ??* ``var_name``: name of the operator representing the multiplication by `x`.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: B1 = BinomialBasis; B2 = PowerBasis; B3 = FallingBasis(1,0,1)
            sage: #TODO ProductBasis([B1,B2]).factors == (B1, B2) ## output: True
            sage: #TODO ProductBasis([B1,B2]).nfactors() ## output: 2
            sage: #TODO ProductBasis([B1,B3,B2]).factors == (B1,B3,B2) ## output: True
            sage: #TODO ProductBasis([B1,B3,B2]).nfactors() ## output: 3

        This method is a simplification of a call to :class:`SievedBasis`. The following example
        illustrates how this can be used to understand better the recurrence for the Apery's `\zeta(3)`-recurrence::

            sage: #TODO b1 = FallingBasis(1,0,1); b2 = FallingBasis(1,1,-1); n = b1.n()
            sage: #TODO B = ProductBasis([b1,b2]).scalar(1/factorial(n))

        This basis ``B`` contains the elements 

        .. MATH::

            \begin{matrix}
            \binom{n + k}{2k}\ \text{if }k\equiv 0\ (mod\ 2)\\
            \binom{n + k}{2k+1}\ \text{if }k\equiv 1\ (mod\ 2)
            \end{matrix}

        We first extend the compatibility with `E: n\mapsto n+1` by guessing and then we compute the product basis
        with itself::

            sage: #TODO B.set_endomorphism('E', guess_compatibility_E(B, sections=2))
            sage: #TODO B2 = ProductBasis([B,B])

        Now the basis ``B2`` is formed in 4 sections by the following elements:

        .. MATH::

            \begin{matrix}
                \binom{n+k}{2k}^2\ \text{if }k\equiv 0\ (mod\ 4)\\
                \binom{n+k}{2k}\binom{n+k}{2k+1}\ \text{if }k\equiv 1\ (mod\ 4)\\
                \binom{n+k}{2k+1}^2\ \text{if }k\equiv 2\ (mod\ 4)\\
                \binom{n+k+1}{2k+2}\binom{n+k}{2k+1}\ \text{if }k\equiv 3\ (mod\ 4)
            \end{matrix}

        We can check that ``B2`` is compatible with the multiplication by `n` and with 
        the endomorphism `E`::

            sage: #TODO a,b,m,alpha = B2.compatibility('x')
            sage: #TODO Matrix([[alpha(i,j,B2.n()) for j in range(-a,b+1)] for i in range(m)]) ## output: [      n 2*n + 1], [      n 2*n + 1], [ -n - 1 2*n + 2], [ -n - 1 2*n + 2]
            sage: #TODO B2.recurrence('x') ## output: [        n         0         0 (2*n)*Sni], [(2*n + 1)         n         0         0], [        0 (2*n + 1)  (-n - 1)         0], [        0         0 (2*n + 2)  (-n - 1)]

        Now consider the following difference operator:

        .. MATH::

            L = (n+2)^3 E^2 - (2*n + 3)(17*n^2+51*n+39)E + (n+1)^3

        This operator `L` is compatible with the basis ``B2``. We can get then
        the associated recurrence matrix. Taking the first column and the GCRD
        of its elements, we can see that if a sequence `y(n)`
        that can be written in the form `y(n) = \sum_{k\geq 0}c_k\binom{n+k}{2k}^2` satisfies
        that

        .. MATH::

            (k+1)^2c_{k+1} - 4(2k+1)^2c_k = 0.

        Doing that with the code::

            sage: #TODO from ore_algebra import OreAlgebra
            sage: #TODO R.<x> = QQ[]; OE.<E> = OreAlgebra(R, ('E', lambda p : p(x=x+1), lambda p : 0))   
            sage: #TODO L = (x+2)^3 *E^2 - (2*x+3)*(17*x^2+51*x+39)*E+(x+1)^3
            sage: #TODO M = B2.recurrence(L)
            sage: #TODO column = [B2.remove_Sni(M.coefficient((j,0))) for j in range(4)]
            sage: #TODO column[0].gcrd(*column[1:]) ## output: (n^2 + 2*n + 1)*Sn - 16*n^2 - 16*n - 4
    '''
    return SievedBasis(factors, list(range(len(factors))))

class _SievedQuasiTriangular:
    def __init__(self, quasi_triangular_sequences : list[Sequence], cycle : list[int]):
        self.__qt_seq = quasi_triangular_sequences
        if any(el < 0 or el >= self.F for el in cycle):
            raise ValueError("Incorrect cycle for the number of factors") 
        self.__cycle = cycle
        self.__computed : dict[int, int] = dict()

    @property
    def L(self): return len(self.__cycle)
    @property
    def F(self): return len(self.__qt_seq)

    @cached_property
    def generator(self):
        n = 0
        current = self.F * [0]
        m = 0 # `m` says the current position on the basis

        while True:
            ## We start the vector of goals for the current n
            I = self.I(n)
            while all(i-c >= 0 for (c,i) in zip(current, I)):
                current[self.__cycle[m%self.L]] += 1
                m += 1
            self.__computed[n] = m-1
            yield m-1
            n += 1

    def I(self, n: int) -> list[int]:
        return [ZZ(seq(n)) for seq in self.__qt_seq]
    
    def __call__(self, n: int):
        while n not in self.__computed:
            next(self.generator)
        return self.__computed[n]

##################################################################################################################
###
### DEFINITE SUM SOLUTIONS METHOD (see article)
###
##################################################################################################################
def DefiniteSumSolutions(operator, *input: int | list[int]):
    r'''
        Petkovšek's algorithm for transforming operators into recurrence equations.
        
        This method is the complete execution for the algorithm **DefiniteSumSolutions** described in
        :doi:`10.1016/j.jsc.2022.11.002`. This methods takes an operator `L` and convert the problem
        of being solution `L \cdot y(n) = 0` to a recurrence equation assuming some hypergeometric
        terms in the expansion.
        
        The operator must be a difference operator of `\mathbb{Q}[x]<E>` where `E: n \mapsto n+1`.
        
        This function does not check the nature of the generator, so using this algorithm with different 
        types of operators may lead to some inconsistent results.
        
        INPUT:

        * ``operator``: difference operator to be transformed.
        * ``input``: the coefficients of the binomial coefficients we assume appear in the expansion
          of the solutions. This input can be given with the following formats:
          - ``a_1,a_2,...,a_m,b_1,b_2,...,b_m``: an unrolled list of `2m` elements.
          - ``[a_1,a_2,...,a_m,b_1,b_2,...,b_m]``: a compress list of `2m` elements.
          - ``[a_1,...,a_m],[b_1,...,b_m]``: two lists of `m` elements.

        OUTPUT:

        An operator `\tilde{L}` such that if a sequence `(c_k)_k` satisfies `L \cdot (c_k)_k = 0` then 
        the sequence

        .. MATH::

            y(n) = \sum_{k \geq 0}c_k\prod{i=1}^m \binom{a_in+b_i}{k}

        satisfies `L \cdot y(n) = 0`.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: from ore_algebra import OreAlgebra
            sage: R.<n> = QQ[]; OE.<E> = OreAlgebra(R, ('E', lambda p : p(n=n+1), lambda p : 0))
            sage: #TODO DefiniteSumSolutions((n+1)*E - 2*(2*n+1), 1,1,0,0) ## output: Sk - 1
            sage: example_2 = 4*(2*n+3)^2*(4*n+3)*E^2 - 2*(4*n+5)*(20*n^2+50*n+27)*E + 9*(4*n+7)*(n+1)^2
            sage: #TODO DefiniteSumSolutions(example_2, 1,1,0,0) ## output: (k + 1/2)*Sk - 1/4*k - 1/4
    '''
    ## Checking the input
    if(len(input) == 1 and type(input) in (tuple, list)):
        input = input[0]

    if(len(input)%2 != 0):
        raise TypeError("The input must be a even number of elements")
    elif(len(input) !=  2 or any(type(el) not in (list,tuple) for el in input)):
        m = len(input)//2
        a = input[:m]; b = input[m:]
    else:
        a,b = input; m = len(a)
    
    if(len(a) != len(b)):
        raise TypeError("The length of the two arguments must be exactly the same")
        
    if(any(el not in ZZ or el <= 0 for el in a)):
        raise ValueError("The values for `a` must be all positive integers")
    if(any(el not in ZZ for el in b)):
        raise ValueError("The values for `a` must be all integers")
        
    ## Getting the name of the difference operator
    E = str(operator.parent().gens()[0])
    
    if(m == 1): # Non-product case
        return BinomialTypeBasis(a[0],b[0],E=E).recurrence(operator)
    
    ## Building the appropriate ProductBasis
    B = ProductBasis([BinomialTypeBasis(a[i],b[i],E=E) for i in range(m)])
    
    ## Getting the compatibility matrix R(operator)
    matrix_recurrences = B.recurrence(operator, output="ore")
        
    ## Extracting the gcrd for the first column
    result = matrix_recurrences[0][0].gcrd(*[matrix_recurrences[j][0] for j in range(m)])
    
    return result

##################################################################################################################
###
### Generic Binomial Basis Methods (see file "gen_binomial_basis")
###
##################################################################################################################
## TODO def GeneralizedBinomial(a: int, b: int, c: int, m: int, r: int) -> FactorialBasis (cached)
## TODO def multiset_inclusion(l1 : list|tuple, l2 : list|tuple) -> bool
## TODO def guess_compatibility_E(basis: FactorialBasis, shift: element.Element = 1, sections: int = None, A: int = None, bound_roots: int = 50, bound_data: int = 50) -> OreOperator
## TODO def guess_rational_function(data: Collection, algebra: OreAlgebra_generic) -> element.Element
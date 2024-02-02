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

from collections.abc import Callable
from functools import lru_cache, reduce, cached_property
from itertools import chain, product
from sage.all import binomial, latex, Matrix, parent, PolynomialRing, prod, vector, QQ, SR, ZZ #pylint: disable=no-name-in-module
from sage.categories.pushout import pushout
from sage.misc.cachefunc import cached_method #pylint: disable=no-name-in-module
from typing import Any, Collection

from ..sequences.base import Sequence, SequenceSet, ConstantSequence, IdentitySequence
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

        A `beta(n)`-factorial basis is a specific type of Sequences basis where the elements
        are defined using a recurrence of order 1. This can be seen also as a specific
        type of hypergeometric sequence of sequences.

        More precisely, a `\beta(n)`-factorial basis is a basis of sequences `B = \{P_k(n)\}` where 
        the `k`-th element is a polynomial w.r.t. `\beta(n)` of degree `k` such that 

        .. MATH::

            P_{k+1}(n) = (a_{k}\beta(n) + b_{k})P_{k}(n).

        This type of basis have special types of compatibilities. More precisely, they are 
        **always** compatible with the "multiplication by `\beta(n)`" operation. This is a special type
        of homomorphism, and always satisfies:

        .. MATH::

            \beta(n)P_k = \frac{1}{a_k}P_{k+1}(n) - \frac{b_k}{a_k}P_k(n).

        Besides the sequences `(a_k)_k` and `(b_k)_k` that define the elements of the basis, 
        there are two other equivalent sequences: the root sequences and the leading coefficient sequence:

        .. MATH::

            \rho_{k+1} = \frac{-b_k}{a_k},\qquad c_k = \prod_{l=1}^k a_l.

        The root sequence `\rho_k` defines for each element the new root added to the element `P_k(n)`. On the 
        other hand, the leading coefficient sequence provides the leading coefficient of the polynomial `P_k(n)`.
        Then, there are two main criteria to determine whether a `\beta(n)`-factorial sequence is compatible with an 
        homomorphism and a derivation (see :doi:`10.1016/j.jsc.2022.11.002`, Propositions 14 and 16).

        INPUT:

            * ``ak``: a sequence to be used for `a_k`. It can be a rational expression in some variable (see argument ``gamma``)
            * ``bk``: a sequence to be used for `b_k`. See argument ``ak``.
            * ``universe`` (optional): universe for the elements of the basis.
            * ``beta``: either ``None`` or a tuple ``(name, seq)``. This defines the sequence `beta(n)` and a name for it. If
              not given, it takes as default the values ``(`n`, n -> n)``.
            * ``gamma``: either ``None`` or a tuple ``(name, seq)``. This defines a sequence `\gamma(k)` such that `a_k` and `b_k`
              are built (if necessary) as :class:`RationalSequence` w.r.t. `gamma(k)`. By default, it takes the value ``(`k`, n -> n)``.
            * ``as_2seq`` (optional): sequence in 2 variables that will be use for generic purposes in :class:`PSBasis`.
    '''
    def __init__(self, ak: Sequence, bk: Sequence, universe = None, *, 
                 beta: tuple[str, Sequence]=None, 
                 gamma: tuple[str, Sequence]=None, 
                 as_2seq: Sequence = None, _extend_by_zero=False, 
                 **kwds):
        ## Treating the beta/gamma arguments
        beta = beta if beta != None else ('n', IdentitySequence(ZZ, **kwds))
        gamma = gamma if gamma != None else ('k', IdentitySequence(ZZ, **kwds))

        ## Treating the arguments a_k and b_k
        if not isinstance(ak, Sequence):
            if universe != None:
                ak = ExpressionSequence(SR(ak), [gamma[0]], universe, meanings=gamma[1], **kwds)
        if not isinstance(bk, Sequence): 
            if universe != None:
                bk = ExpressionSequence(SR(bk), [gamma[0]], universe, meanings=gamma[1], **kwds)
        if not isinstance(ak, Sequence) or ak.dim != 1:
            raise TypeError(f"[FactorialBasis] The element a_k must be a univariate sequence or an expression in 'k'")
        if not isinstance(bk, Sequence) or bk.dim != 1:
            raise TypeError(f"[FactorialBasis] The element a_k must be a univariate sequence or an expression in 'k'")
        universe = universe if universe != None else pushout(ak.universe, bk.universe)
        
        self.__ak = ak.change_universe(universe)
        self.__bk = bk.change_universe(universe)
        self.__rho = -(bk/ak) 
        self.__lc = ak.partial_prod()

        self.__poly_ring = PolynomialRing(universe, beta[0]) # this is the polynomial ring for the elements of the sequence
        self.__beta = beta; self.__gamma = gamma
        self.__gen = self.__poly_ring.gens()[0]

        @lru_cache
        def __get_element(k):
            if k < 0: return self.__poly_ring.zero()
            elif k == 0: return self.__poly_ring.one()
            else: return (self.ak(k-1)*self.__gen + self.bk(k-1))*__get_element(k-1) #pylint: disable=not-callable

        sequence = as_2seq if as_2seq != None else lambda k : self._RationalSequenceBuilder(__get_element(k))

        super().__init__(sequence, universe, _extend_by_zero=_extend_by_zero, **kwds)

        # We create now the compatibility with the multiplication by the variable generator
        self.set_compatibility(beta[0], Compatibility([[self.rho, 1/self.ak]], 0, 1, 1), True, "any")

    def args_to_self(self) -> tuple[list, dict[str]]:
        return (
            [self.ak, self.bk], 
            {"universe": self.base, 
             "beta": self.__beta, 
             "gamma": self.__gamma, 
             "as_2seq": self.as_2dim(),
             "_extend_by_zero": self._Sequence__extend_by_zero,
             **self.extra_info()["extra_args"]
             }
        )

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
        return RationalSequence(rational, [self.__beta[0]], self.base, meanings=self.__beta[1])

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
        return FactorialBasis(self.ak.shift(shift), self.bk.shift(shift), **self_args)

    def compatible_division(self, operator) -> DivisionCondition:
        r'''
            Method to compute the division condition for a given operator.

            The division condition for a compatible operator comes from the equivalence in :doi:`10.1016/j.jsc.2022.11.002`, Proposition 11. 
            For further information, check the class :class:`DivisionCondition`
        '''
        return DivisionCondition.from_compatibility(self.compatibility(operator), self)
    
    @cached_method
    def matrix_ItP(self, size: int, section: int = 0) -> tuple[tuple[Sequence]]:
        r'''
            Computes the matrix that transforms the polynomial basis given by ``self`` into the associated power basis.

            Let `\{P_k(n)\}` be a `\beta(n)`-factorial basis. Then, the element `P_k(n)` is a polynomial in `\beta(n)` 
            of degree exactly `k` and, moreover, the polynomial `P_k(n)` divides `P_K(n)` for all `k \leq K`. This implies
            that for any fixed `k \in \mathbb{N}` the sequence `\left(\frac{P_{k+i}(n)}{P_k(i)}\right)_i` is a sequence
            of polynomials of degree `i` and, hence, a basis of the polynomial ring.

            This method computes the matrix of fixed ``size`` that write the polynomials `\frac{P_{k+i}(n)}{P_k(i)}` 
            into the power basis `\beta(n)^i`.

            For doing so, the only information we have is the `(1,0)`-compatibility of the :class:`FactorialBasis` with 
            the multiplication by `\beta(n)` (see ``self.compatibility(self.gen())``). Let us assume that `\beta(n)` 
            is compatible in `t`sections and let `k = mt+r` for `r \in \{0,\ldots, t-1\}`. Then:

            .. MATH::

                \beta(n)^i P_{mt+r}(n) = \sum_{j=0}^i \alpha_{r,j}^{(i)}(m)P_{mt+r+j},

            where the `\alpha_r,j}^{(i)}(m)` are the compatibility coefficients of the multiplication by `\beta(n)^i`. Then,
            we can write these identities in a matri-vector multiplication format:

            .. MATH::

                P_{mt+r}(n) \begin{pmatrix} 1\\ \beta(n) \\ \beta(n)^2 \\ \vdots \\ \beta(n)^{S-1} \end{pmatrix} = 
                \begin{pmatrix}
                    1 & 0 & 0 & \ldots & 0 \\
                    \alpha_{r,0}^{(1)}(m) & \alpha_{r,1}^{(1)}(m) & 0 & \ldots & 0 \\
                    \alpha_{r,0}^{(2)}(m) & \alpha_{r,1}^{(2)}(m) & \alpha_{r,2}^{(2)}(m) & \ldots & 0 \\
                    \vdots & \vdots & \vdots & \ddots & \vdots \\
                    \alpha_{r,0}^{(S-1)}(m) & \alpha_{r,1}^{(S-1)}(m) & \alpha_{r,2}^{(S-1)}(m) & \ldots & \alpha_{r,S-1}^{(S-1)}(m)
                \end{pmatrix}
                \begin{pmatrix} P_{mt+r}(n)\\ P_{mt+r+1}(n) \\ P_{mt+r+2}(n) \\ \vdots \\ P_{mt+r+S-1}(n)\end{pmatrix}

            This matrix is lower-triangular and has a non-zero determinant (since we know the degrees of the polynomials are exact). Hence,
            this matrix is always invertible. Multiplying by its inverse from the left and dividing the whole equation by `P_{mt+r}` we
            obtain formulas for `\frac{P_{mt+r+i}(n)}{P_{mt+r}(n)}` with respect to `\beta(n)` (i.e., transforming the increasing basis 
            (see method :func:`increasin_basis`) into the power basis).

            REMARK: in order to get information about the matrix for `k = mt+r`, we need to know the exact `r` that we are considering. That
            is why we ask for an input ``section`` in the method, that will be automatically handled when having simply one section.

            INPUT:

            * ``size``: size of the matrix ti be considered. (May be used in the future to improve performance in this method)
            * ``section``: value for `r`. This value will always be tuned down depending on the sections of the multiplication 
              by `\beta(n)`.

            OUTPUT:

                A matrix (in format of tuple of tuples) such that the multiplication from the left with the coordinates of a polynomial
                of degree at most ``size-1`` w.r.t. the increasing basis provides the coordinates of that polynomial w.r.t. the power basis 
                induced by `\beta(n)`.
        '''
        PtI = self.matrix_PtI(size, section) # this checks the arguments
        ## Now we need to invert PtI. We can do it with the adjoint matrix (full of determinants) or by iterated substitution (Gaussian elimination)
        ItP = [[ZZ(0) for _ in range(size)] for _ in range(size)]
        for i in range(size): ItP[i][i] = ZZ(1)

        ## PtI | ItP will do the gaussian elimination on PtI and obtain at the end ItP
        for i in range(size): # the pivot is always in the diagonal
            # making a one in the pivot
            for j in range(i+1): ItP[i][j] /= PtI[i][i]
            # now we make zeros below the pivot
            for k in range(i+1, size):
                for j in range(i+1):
                    ItP[k][j] -= PtI[k][i] * ItP[i][j]
        
        return tuple(tuple(row) for row in ItP)

    @cached_method
    def matrix_PtI(self, size: int, section: int = 0) -> tuple[tuple[Sequence]]:
        r'''
            Computes the matrix that transforms the power basis into the polynomial basis given by ``self``.

            Let `\{P_k(n)\}` be a `\beta(n)`-factorial basis. Then, the element `P_k(n)` is a polynomial in `\beta(n)` 
            of degree exactly `k` and, moreover, the polynomial `P_k(n)` divides `P_K(n)` for all `k \leq K`. This implies
            that for any fixed `k \in \mathbb{N}` the sequence `\left(\frac{P_{k+i}(n)}{P_k(i)}\right)_i` is a sequence
            of polynomials of degree `i` and, hence, a basis of the polynomial ring.

            This method computes the matrix of fixed ``size`` that write the polynomials `\frac{P_{k+i}(n)}{P_k(i)}` 
            into the power basis `\beta(n)^i`.

            For doing so, the only information we have is the `(1,0)`-compatibility of the :class:`FactorialBasis` with 
            the multiplication by `\beta(n)` (see ``self.compatibility(self.gen())``). Let us assume that `\beta(n)` 
            is compatible in `t`sections and let `k = mt+r` for `r \in \{0,\ldots, t-1\}`. Then:

            .. MATH::

                \beta(n)^i P_{mt+r}(n) = \sum_{j=0}^i \alpha_{r,j}^{(i)}(m)P_{mt+r+j},

            where the `\alpha_r,j}^{(i)}(m)` are the compatibility coefficients of the multiplication by `\beta(n)^i`. Then,
            we can write these identities in a matri-vector multiplication format:

            .. MATH::

                P_{mt+r}(n) \begin{pmatrix} 1\\ \beta(n) \\ \beta(n)^2 \\ \vdots \\ \beta(n)^{S-1} \end{pmatrix} = 
                \begin{pmatrix}
                    1 & 0 & 0 & \ldots & 0 \\
                    \alpha_{r,0}^{(1)}(m) & \alpha_{r,1}^{(1)}(m) & 0 & \ldots & 0 \\
                    \alpha_{r,0}^{(2)}(m) & \alpha_{r,1}^{(2)}(m) & \alpha_{r,2}^{(2)}(m) & \ldots & 0 \\
                    \vdots & \vdots & \vdots & \ddots & \vdots \\
                    \alpha_{r,0}^{(S-1)}(m) & \alpha_{r,1}^{(S-1)}(m) & \alpha_{r,2}^{(S-1)}(m) & \ldots & \alpha_{r,S-1}^{(S-1)}(m)
                \end{pmatrix}
                \begin{pmatrix} P_{mt+r}(n)\\ P_{mt+r+1}(n) \\ P_{mt+r+2}(n) \\ \vdots \\ P_{mt+r+S-1}(n)\end{pmatrix}

            REMARK: in order to get information about the matrix for `k = mt+r`, we need to know the exact `r` that we are considering. That
            is why we ask for an input ``section`` in the method, that will be automatically handled when having simply one section.

            INPUT:

            * ``size``: size of the matrix ti be considered. (May be used in the future to improve performance in this method)
            * ``section``: value for `r`. This value will always be tuned down depending on the sections of the multiplication 
              by `\beta(n)`.

            OUTPUT:

                A matrix (in format of tuple of tuples) such that the multiplication from the left with the coordinates of a polynomial
                of degree at most ``size-1`` w.r.t. the power basis induced by `\beta(n)` provides the coordinates of that polynomial 
                w.r.t. the increasing basis.
        '''
        beta = self.gen()
        ## Checking the arguments
        size = ZZ(size)
        if size <= 0: raise ValueError(f"[matrix_PtI] The value for size (got {size}) must be a positive integer")

        section = ZZ(section)
        section %= self.compatibility(beta).t

        result = []
        for i in range(size): # the rows of the matrix
            comp_beta_i = self.compatibility(beta**i)
            result.append(tuple([comp_beta_i[section, j] for j in range(size)])) # get zeros when j > i

        return tuple(result)

    ##################################################################################
    ### TODO: def increasing_polynomial(self, *args, **kwds)
    ### TODO: def equiv_DtC(self, compatibility: str | OreOperator | TypeCompatibility) -> TypeCompatibility
    ### TODO: def equiv_CtD(self, division: TypeCompatibility) -> TypeCompatibility

def RootSequenceBasis(rho: Sequence, lc: Sequence, 
                        universe = None, *, 
                        beta: tuple[str, Sequence]=None, 
                        gamma: tuple[str, Sequence]=None, 
                        as_2seq: Sequence = None, _extend_by_zero=False, 
                        **kwds):
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
            rho = ExpressionSequence(SR(rho), [gamma[0]], universe, meanings=gamma[1], **kwds)
    if not isinstance(lc, Sequence): 
        if universe != None:
            lc = ExpressionSequence(SR(lc), [gamma[0]], universe, meanings=gamma[1], **kwds)
    if not isinstance(rho, Sequence) or rho.dim != 1:
        raise TypeError(f"[FactorialBasis] The element rho must be a univariate sequence or an expression in 'k'")
    if not isinstance(lc, Sequence) or lc.dim != 1:
        raise TypeError(f"[FactorialBasis] The element lc must be a univariate sequence or an expression in 'k'")
    
    ak = lc.shift()/lc
    bk = -(rho*ak)
    return FactorialBasis(ak, bk, universe, beta=beta, gamma=gamma, as_2seq=as_2seq, _extend_by_zero=_extend_by_zero, **kwds)
        
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
### DIVISION CONDITIONS FOR COMPATIBILITY
###
###############################################################################
class DivisionCondition:
    r'''
        This class represents a division condition for a linear operator over a :class:`FactorialBasis`.

        In :doi:`10.1016/j.jsc.2022.11.002`, Proposition 11, the authors showed that the `(A,B)`-compatibility
        of a linear operator `L` with a :class:`FactorialBasis` is equivalent to the following two conditions:

        1. For every `k \geq 0`, `\deg(L P_k(n)) \leq k+B`.
        2. For every `k \geq A`, `P_{k-A}(n)` divides `LP_k(n)`.

        Recall that a factorial basis is a basis `P_k(n)` where the `k`-th element is a polynomial in some sequence
        `\beta(n)` of degree `k`. Hence, the degree in condition 1. and the division in condition 2. is seeing as
        a polynomial in `\beta(n)`.

        The equivalence between the condition 1. and 2., and the `(A,B)`-compatibility is a key computational 
        process for factorial bases. This class allows to represent the conditions 1. and 2. fully and provide 
        methods to transform these conditions into :class:`Compatibility` for a given factorial basis.

        To represent this divisibility condition, we need to stablish the value for `A` and how the division
        of `P_{k-A}(n)` and `LP_k(n)` actually happens. Namely, since the degrees of `P_{k-A}(n)` and 
        `LP_k(n)` are fixed, then we can represent:

        .. MATH::

            \frac{LP_k(n)}{P_{k-A}(n)} = \sum_{i=0}^{A+B} \delta_{i}(k) \beta(n)^i.

        Where `\delta_i(k)` are sequences in `k` that defines how the division condition changes for each `k \geq A`.
        Hidden in this notation is the case when the compatibility is in `t` sections. It is then clear that the 
        sequences `\delta_{i}(k)` may be defined in sections too, leading to the same structure as in a compatibility
        by sections. Namely, let `k = mt+r` for `m\geq 0` and `r\in \{0,\ldots,t-1\}`:

        .. MATH::

            \frac{LP_{mt+r}}{P_{k-A}(n)} = \sum_{i=0}^{A+B} \delta_{r,i}(m) \beta(n)^i`.


        INPUT:

        * ``delta``: a list/tuple or a list/tuple of list/tuples with sequences representing the sequences `\delta`.
        * ``A``: integer representing the lower bound for the division. Used to know exactly where the divisibility condition starts.
        * ``base`` (optional): the universe of the sequeces in ``delta``. Used optionally to enforce this universe on the sequences.
          If not given, the universe will be deduced from the arguments in ``delta`` automatically.
        * ``t`` (optional): number of sections to be used. If not given it is deduced from the input ``delta``. Otherwise, we 
          enforce that ``delta`` is a list of at least ``t`` lists.
    '''
    def __init__(self, delta: Collection[Sequence] | Collection[Collection[Sequence]], A: int, base = None, sections: int = None):
        ## Checking all the arguments
        if not isinstance(delta, Collection):
            raise TypeError(f"[DivisionCondition] The division coefficients must be given as a list (got: {delta.__class__})") 
        if sections != None: # we force the input delta to have richer structure
            if sections <= 0: raise ValueError(f"The number of sections must be strictly positive (given {sections})")

            if any(not isinstance(sec, Collection) for sec in delta):
                raise TypeError(f"[DivisionCondition] When given number of sections, the division coefficients must be given as a list of lists.")
            if len(delta) < sections:
                raise ValueError(f"[DivisionCondition] Not enough sections provided (required {sections}, given {len(delta)})")
        else:
            if all(isinstance(sec, Collection) for sec in delta):
                sections = len(delta)
            else:
                sections = 1
                delta = [delta] # now is a list of lists with just one element.
        
        if any(len(delta[0]) != len(seq) for seq in delta[1:sections]):
            raise ValueError(f"[DivisionCondition] Inconsistent number of elements in the ``delta`` coefficients")

        ## Computing the common parent
        parent = None
        for sec in delta:
            for seq in sec:
                if parent is None: parent = seq.parent()
                else: parent = pushout(parent, seq.parent())
                
        ## We transform elements in delta to a specific sequence ring
        if not isinstance(parent, SequenceSet): # we consider parent as the constant ring for sequences
            delta = [[ConstantSequence(seq, universe=parent, dim=1) for sec in seq] for seq in delta]
        else:
            delta = [[parent(seq) for seq in sec] for sec in delta]
        
        ## We make sure ``base`` is used (if given)
        if base != None: delta = [[seq.change_universe(base) for seq in sec] for sec in delta]

        self.__delta = delta
        self.__A = A
        self.__sections = sections
        self.__base = base if base != None else (parent  if parent is not isinstance(parent, SequenceSet) else parent.codomain())

    @property
    def lower_bound(self) -> int: return self.__A
    A = lower_bound #: alias for ``lower_bound``
    @property
    def sections(self) -> int: return self.__sections
    t = sections #: alias for ``sections``

    def coefficient(self, index: int, section:int = None) -> Sequence:
        if section == None and self.sections > 1: raise ValueError(f"Sections required when having more than one section.")
        elif section == None: section = 0

        return self.__delta[section][index]
    def __getitem__(self, input: int | tuple[int,int]) -> Sequence: 
        if isinstance(input, int):
            return self.coefficient(input, 0)
        else:
            return self.coefficient(input[1], input[0])
    def size(self) -> int: return len(self.__delta[0])

    def base(self): return self.__base
    def change_base(self, new_base) -> DivisionCondition:
        return DivisionCondition(self.__delta, self.A, new_base, self.sections)
    
    def in_sections(self, new_sections: int) -> DivisionCondition:
        r'''
            Represent the division condition for a given number of sections (if possible).

            Let us assume that for some operator and basis we have the following division condition.

            .. MATH::

                \frac{L P_{mt+r}(n)}{P_{mt+r-A}(n)} = \sum_{i=0}^{A+B} \delta_{r,i}(m)\beta(n)^i.

            This is a divisibility condition in `t` sections. Then, for any `T` multiple of `t`, we can 
            also write a division condition. Namely, let `k = MT+R` for `M \geq 0`, `R \in \{0,\ldots, T\}`.
            Then, `k = (Mq_T + q_R) + r_R` where `q_Tt = T` and `q_R t + r_R = R`. Hence,

            .. MATH::

                \frac{L P_{MT+R}(n)}{P_{mt+r-A}(n)} = \frac{L P_{(Mq_T + q_R) + r_R}(n)}{P_{(Mq_T + q_R) + r_R-A}(n)} = 
                \sum_{i=0}^{A+B} \delta_{r_R, i}(Mq_T+q_R) \beta(n)^i.

            Hence, we have that the new `\tilde{\delta}_{R,i}(M) = \delta_{r_R, i}(Mq_T+q_R)`.

            INPUT:

            * ``new_sections``: number of new sections. It must be a multiple of ``self.sections``.
        '''
        q_T, r_T = ZZ(new_sections).quo_rem(self.t)

        if r_T != 0: raise ValueError(f"[DivisionCondition] The new number of sections must be a multiple of the current sections (Got: {new_sections}; Current: {self.t})")

        new_delta = []
        for R in range(new_sections):
            q_R, r_R = ZZ(R).quo_rem(self.t)
            new_section = []
            for i in range(self.size()): # i = 0, ..., A+B
                new_section.append(self[r_R, i].linear_subsequence(0, q_T, q_R))
            new_delta.append(new_section)

        return DivisionCondition(new_delta, self.A, self.base(), sections=new_sections)
    
    def to_compatibility(self, basis: FactorialBasis) -> Compatibility:
        r'''
            Method tha implements the equivalence of :doi:`10.1016/j.jsc.2022.11.002`, Proposition 11.

            Let us assume that for some operator and basis we have the following division condition.

            .. MATH::

                \frac{L P_{mt+r}(n)}{P_{mt+r-A}(n)} = \sum_{i=0}^{A+B} \delta_{r,i}(m)\beta(n)^i.

            Then, we can simply multiply both sides by `P_{mt+r-A}(n)`, and check how the different multiplications with `\beta(n)^i`
            affects the element `P_{mt+r-A}(n)`. Let us assume that

            .. MATH::

                \beta(n)^i P_{mt+r}(n) = \sum_{j=0}^i \alpha_{r,j}^{(i)}(m)P_{mt+r+j}(n),

            (note that `j \geq 0` since `P_k(n)` is a `\beta(n)`-factorial basis). Let `(r - A) = Qt + R`, where `R \in \{0,\ldots,t-1\}`.
            Then we have that

            .. MATH::

                \beta(n)^i P_{mt+r-A}(n) = \beta(n)^i P_{(m+Q)t + R}(n) = \sum_{j=0}^i \alpha_{R,j}^{(i)}(m+Q)P_{mt+r-A+j}(n).

            Hence, putting everything together, we obtain the following identity:

            .. MATH::

                L P_{mt+r}(n) = \sum_{i=0}^{A+B} \delta_{r,i}(m)\beta(n)^i P_{mt+r-A}(n)
                              = \sum_{i=0}^{A+B} \delta_{r,i}(m) \sum_{j=0}^i \alpha_{R,j}^{(i)}(m+Q)P_{mt+r-A+j}(n)
                              = \sum_{j=0}^{A+B} \sum_{i=j}^{A+B} \delta_{r,i}(m) \alpha_{R,j}^{(i)}(m+Q)P_{mt+r-A+j}(n)
                              = \sum_{j=-A}^{B}  \sum_{i=j+A}^{A+B} \delta_{r,i}(m) \alpha_{R,j+A}^{(i)}(m+Q)P_{mt+r+j}(n)
                              = \sum_{j=-A}^{B}  \left(\sum_{i=j+A}^{A+B} \delta_{r,i}(m) \alpha_{R,j+A}^{(i)}(m+Q)\right) P_{mt+r+j}(n),
            
            Hence we get that the new compatibility coefficients for exactly `t` sections (i.e., same sections as the division condition) are the following:

            .. MATH::

                \tilde{\alpha}_{r,i}(m) = \sum_{j=i+A}^{A+B} \delta_{r,j}(m) \alpha_{R,i+A}^{(j)}(m+Q)

            INPUT:

            * ``basis``: a :class:`FactorialBasis` to be used for the compatibility with the _variable_ `\beta(n)`.
        '''
        if not isinstance(basis, FactorialBasis): raise TypeError(f"[DivisionCondition] The basis must be factorial w.r.t. some variable.")

        comp_powers = [basis.compatibility(basis.gen()**j).in_sections(self.t) for j in range(self.size())] # compatibility of beta(n)^j in `t` sections.
        A = self.A
        B = self.size() - A - 1

        new_alpha = []
        for r in range(self.t): # we iterate in each of the sections
            new_section = []
            for i in range(-A, B+1):
                Q, R = ZZ(r - A).quo_rem(self.t)
                new_section.append(sum(self[r,j] * comp_powers[j][R,i+A].shift(Q) for j in range(i+A, A+B+1)))
            new_alpha.append(new_section)
        
        return Compatibility(new_alpha, A, B, self.t)
    
    @staticmethod
    def from_compatibility(compatibility: Compatibility, basis: FactorialBasis) -> DivisionCondition:
        r'''
            Method tha implements the converse equivalence of :doi:`10.1016/j.jsc.2022.11.002`, Proposition 11.

            Let us assume now that we have an `(A,B)`-compatible operator `L` in `t` sections with a basis `P_k(n)`. Then
            it follows that

            .. MATH::

                L P_{mt+r}(n) = \sum_{i=-A}^{B} \alpha_{r,i}(m) P_{mt+r+i}(n).

            By definition of a factorial basis, the polynomials `P_k(n)` always divide the polynomials `P_K(n)` for every `K \geq k`. 
            In particular, we can divide the whole identity by `P_{mt+r-A}(n)` obtaining the following:

            .. MATH::

                \frac{L P_{mt+r}(n)}{P_{mt+r-A}(n)} = \sum_{i=-A}^{B} \alpha_{r,i}(m) \frac{P_{mt+r+i}(n)}{P_{mt+r-A}(n)}.

            Clearly, the quotients `\frac{P_{mt+r+i}(n)}{P_{mt+r-A}(n)}` are polynomials of degree `A+i` in the variable of the 
            factorial basis `\beta(n)`. Hence, we can see that each `P_{mt+r+i}(n)` is a linear combination of `\beta(n)^i P_{mt+r-A}(n)`.

            Let `(r - A) = Qt + R`, where `R \in \{0,\ldots,t-1\}`. Then we have that, following the compatibility with `\beta(n)` (which 
            comes from the fact of being a factorial basis):

            .. MATH::

                \beta(n)^i P_{mt+r-A}(n) = \beta(n)^i P_{(m+Q)t + R}(n) = \sum_{j=0}^i \alpha_{R,j}^{(i)}(m+Q)P_{mt+r-A+j}(n).

            Let us consider the matrix generated by the sequences `\alpha_{R,j}^{(i)}(m+Q)` and denote it by `\Lambda`. Then

            .. MATH::

                P_{mt+r-A}(n) \begin{pmatrix}1\\\beta(n)\\\beta(n)^2\\\vdots\\\beta(n)^{A+B}\end{pmatrix} = 
                \Lambda \begin{pmatrix} P_{mt+r-A}(n) \\ P_{mt+r-A+1}(n) \\ \vdots \\ P_{mt+r+B}(n)\end{pmatrix}

            The matrix `\Lambda` is triangular with the diagonal full of 1s. Hence, it is invertible (even in the sequence ring). Multiplying from
            the left by this inverse, we get the identity

            .. MATH::

                \begin{pmatrix} P_{mt+r-A}(n) \\ P_{mt+r-A+1}(n) \\ \vdots \\ P_{mt+r+B}(n)\end{pmatrix} = 
                \Lambda^{-1}\begin{pmatrix}1\\\beta(n)\\\beta(n)^2\\\vdots\\\beta(n)^{A+B}\end{pmatrix}P_{mt+r-A}(n).

            Let assume that we obtain something like

            .. MATH::

                P_{mt+r-A+i}(n) = \sum_{j=0}^{i} \lambda_{r,j}^{(i)} \beta(n)^jP_{mt+r-A}(n),

            Then we can plug this into the first equation, obtaining:

            .. MATH::

                \frac{L P_{mt+r}(n)}{P_{mt+r-A}(n)} = \sum_{i=-A}^{B} \alpha_{r,i}(m) \frac{P_{mt+r+i}(n)}{P_{mt+r-A}(n)} 
                                                    = \sum_{i=0}^{A+B} \alpha_{r,i-A}(m) \frac{P_{mt+r-A+i}(n)}{P_{mt+r-A}(n)}
                                                    = \sum_{i=0}^{A+B} \alpha_{r,i-A}(m) \left(\sum_{j=0}^{i} \lambda_{r,j}^{(i)} \beta(n)^j\right)
                                                    = \sum_{i=0}^{A+B} \sum_{j=0}^{i} \alpha_{r,i-A}(m) \lambda_{r,j}^{(i)} \beta(n)^j
                                                    = \sum_{j=0}^{A+B} \left(\sum_{i=j}^{A+B} \alpha_{r,i-A}(m) \lambda_{r,j}^{(i)}\right) \beta(n)^j

            Hence we obtain that the division coefficients that need to be computed are:

            .. MATH::

                \delta_{r,i}(m) = \sum_{i=j}^{A+B} \alpha_{r,i-A}(m) \lambda_{r,j}^{(i)}

            
            INPUT:

            * ``compatibility``: a :class:`Compatibility` condition for a linear operator.
            * ``basis``: a :class:`FactorialBasis` for which the compatibility condition holds.

            OUTPUT:

            The :class:`DivisionCondition` associated to the given ``basis`` and the ``compatibility`` described.
        '''
        A = compatibility.A
        B = compatibility.B
        t = compatibility.t

        new_delta = []
        for r in range(t):
            Q,R = ZZ(r - A).quo_rem(t)
            ## The method matrix_ItP do the job of computing Lambda^{-1}. We need to take the appropriate section and shift
            Lambda_1 = [[el.shift(Q) if isinstance(el, Sequence) else el for el in row] for row in basis.matrix_ItP(A+B+1, section=R)]
            
            ## We collect the new delta coefficients following the formula in documentation
            new_delta.append([sum(compatibility[r,i-A]*Lambda_1[i][j] for i in range(j, A+B+1)) for j in range(A+B+1)])

        return DivisionCondition(new_delta, A, base = basis.base, sections=t)
        
    def __repr__(self) -> str:
        start = f"Divisibility condition (A={self.A}, t={self.sections})"
        try:
            M = Matrix([[self[t,i].generic() for i in range(self.size())] for t in range(self.t)]) 
            start += f" with following coefficient matrix:\n{M}"
        except:
            pass
        return start
    def _latex_(self) -> str:
        code = r"\text{Divisibility condition with shape " + f"(A={self.A}, t={self.sections})" + r":}\\"
        if self.t > 1:
            code += r"\left\{\begin{array}{rl}"
        
        for r in range(self.sections):
            if self.sections > 1:
                code += r"L \cdot \frac{L P_{" + latex(self.t) + r"k + " + latex(r) + r"}}{P_{" + latex(self.t) + r"k + " + latex(r) + r" - " + latex(self.A) + r"}} & = "
            else:
                code += r"L \cdot P_{k-" + latex(self.A) + r"} = "

            monomials = []
            for i in range(self.size()): 
                ## Creating the coefficient
                try:
                    c = self[r,i].generic('k')
                    if c == 0: continue
                    new_mon = r"\left(" + latex(c) + r"\right)"
                except:
                    if self.t > 1:
                        new_mon = r"c_{" + latex(r) + r"," + latex(i) + r"}(k)"
                    else:
                        new_mon = r"c_{" + latex(i) + r"}(k)"
                ## Creating the P_{k+i}
                new_mon += r"X^" + latex(i)
                monomials.append(new_mon)
            code += " + ".join(monomials)
            if self.t > 1: code += r"\\"
        if self.t > 1:
            code += r"\end{array}\right."
        return code 
    
def check_division_condition(basis: FactorialBasis, division : DivisionCondition, action: Callable, bound: int = 100, *, _full=False):
    r'''
        Method that checks whether a basis has a particular division condition for a given action.

        This method takes a :class:`FactorialBasis` (i.e., a sequence of sequences), a given
        operator compatible with it (or simply the :class:`DivisionCondition` object representing
        such compatibility) and check whether the action that is defined for the operator/division condition
        (which is provided by the argument ``action``) has precisely this division condition.

        More precisely, if an operator `L` has a division condition (see :class:`DivisionCondition`) in `t`
        sections given by the formula:
        
        .. MATH::

            \frac{LP_{mt+r}(n)}{P_{mt+r-A}(n)} = \sum_{i=0}^{A+B} \delta_{r,i}(m) \beta(n)^i.

        then this method checks this identity for the ``action`` defining `L`, and the division condition
        defined in ``division``. To avoid problems with the division by zero, we will compute this identity
        multiplying everything by `P_{k-A}(n)`, namely:

        .. MATH::

            LP_{mt+r}(n) = \sum_{i=0}^{A+B} \delta_{r,i}(m) \beta(n)^iP_{mt+r-A}(n).

        This checking is perform until a given `m` and `n` bounded by the input ``bound``.

        INPUT:

        * ``basis``: a :class:`PSBasis` that defines the basis `P=(P_n)_n`.
        * ``division``: a division condition. If an operator is given, then division condition
          for ``basis`` is computed (check method :func:`FactorialBasis.compatible_division`)
        * ``action``: a callable that actually computes the element `L P_n` so it can be compared.
        * ``bound``: a bound for the limit this equality will be checked. Since `L P_n` is a sequence
          this bound is used both for checking equality at each level `n` and until which level the 
          identity is checked.

        OUTPUT:

        ``True`` if all the checking provide equality, and ``False`` otherwise. Be cautious when reading
        this output: ``False`` guarantees that the compatibility is **not** for the action, however, ``True``
        provides a nice hint the result should be True, but it is not a complete proof.

        TODO: add examples 
    '''
    if not isinstance(basis, FactorialBasis):
        raise TypeError(f"[check_division] The given basis must be factorial")
    if not isinstance(division, DivisionCondition):
        division = basis.compatible_division(division)

    A = division.A
    t = division.t
    beta = basis._RationalSequenceBuilder(basis.gen())

    for k in range(A, bound):
        lhs:Sequence = action(basis[k]) # sequence obtained from L P_n
        m,r = ZZ(k).quo_rem(t)
        # \sum_{i=0}^{A+B} \delta_{r,i}(m) \beta(n)^iP_{mt+r-A}(n).
        rhs = sum(division[r,i][m] * beta**i * basis[k-A] for i in range(division.size())) # sequence for the rhs

        if not lhs.almost_equals(rhs, bound):
            return ((k,lhs, rhs), False if _full else False)
    return True

###############################################################################
### 
### SHUFFLED BASIS AND PRODUCT BASIS
### 
###############################################################################
class ShuffledBasis(FactorialBasis):
    r'''
        Class for a Shuffled Basis.

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
            sage: B2 = ShuffledBasis([B,P], [0,1,1,0])
            sage: B2[:4]
            [Sequence over [Rational Field]: (1, 1, 1,...),
             Sequence over [Rational Field]: (0, 1, 2,...),
             Sequence over [Rational Field]: (0, 1, 4,...),
             Sequence over [Rational Field]: (0, 1, 8,...)]
            sage: [el.generic() for el in B2[:6]]
            [1, n, n^2, n^3, 1/2*n^4 - 1/2*n^3, 1/6*n^5 - 1/2*n^4 + 1/3*n^3]

        With this system, we can build the same basis changing the order and the values in the cycle::

            sage: B3 = ShuffledBasis([P,B], [1,0,0,1])
            sage: B3.almost_equals(B2, 30) # checking equality for 30 elements 
            True

        The length of the cycle is the number of associated sections::

            sage: B2.nsections
            4
            sage: ShuffledBasis([B,B,P],[0,0,1,2,1,2]).nsections
            6

        This basis can be use to deduce some nice recurrences for the Apery's `\zeta(2)` sequence::

            sage: #TODO b1 = FallingBasis(1,0,1); b2 = FallingBasis(1,1,-1); n = b1.n()
            sage: #TODO B = ShuffledBasis([b1,b2],[0,1]).scalar(1/factorial(n))

        This basis ``B`` contains the elements 

        .. MATH::

            \begin{matrix}
            \binom{n + k}{2k}\ \text{if }k\equiv 0\ (mod\ 2)\\
            \binom{n + k}{2k+1}\ \text{if }k\equiv 1\ (mod\ 2)
            \end{matrix}

        We first extend the compatibility with `E: x\mapsto x+1` by guessing and then we compute the sieved basis
        with the binomial basis with the cycle `(1,0,1)`::

            sage: #TODO B.set_endomorphism('E', guess_compatibility_E(B, sections=2))
            sage: #TODO B2 = ShuffledBasis([BinomialBasis(), B], [1,0,1])

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
        _extend_by_zero=False, **kwds
    ):
        ## Checking the input
        if(not type(factors) in (list,tuple)):
            raise TypeError("The factors must be either a list or a tuple")
        if(any(not isinstance(el, FactorialBasis) for el in factors)):
            raise TypeError("All the factors has to be factorial basis")
        if any(factor.gen() != factors[0].gen() for factor in factors[1:]):
            raise TypeError("All the factors has to be w.r.t. the same variable")
        variable = str(factors[0].gen())

        if(not type(cycle) in (list,tuple)):
            raise TypeError("The deciding cycle must be a list or a tuple")
        cycle = [ZZ(el) for el in cycle]
        if(any(el < 0 or el > len(factors) for el in cycle)):
            raise ValueError("The deciding cycle must be composed of integers indexing the factors basis")
        
        ## Checking the factorial bases are valid
        beta = factors[0]._FactorialBasis__beta
        if any(beta[1] != factor._FactorialBasis__beta[1] for factor in factors[1:]):
            raise TypeError(f"[ShuffledBasis] Incompatible bases: the b(n)-factorial basis have different values for `b(n)`.")

        ## Storing the main elements
        self.__factors = tuple(factors)
        self.__cycle = tuple(cycle)
        universe = reduce(lambda p,q: pushout(p,q), [f.base for f in self.factors])

        new_ak = Sequence(lambda k : (self.factors[self.cycle[k%self.nsections]]).ak[self.indices[k][self.cycle[k%self.nsections]]], universe)
        new_bk = Sequence(lambda k : (self.factors[self.cycle[k%self.nsections]]).bk[self.indices[k][self.cycle[k%self.nsections]]], universe)

        ## Mixing all the "kwds" and "_extend_by_zero" arguments
        for factor in factors: kwds.update(factor.extra_info()["extra_args"])
        _extend_by_zero = _extend_by_zero and all(factor._Sequence__extend_by_zero for factor in factors)
        ## We call the constructor for a factorial basis
        FactorialBasis.__init__(self, new_ak, new_bk, universe, beta=beta, _extend_by_zero=_extend_by_zero, **kwds)

        ## We reset the compatibility wrt the variable name
        try:
            self.set_compatibility(variable, self._extend_compatibility_X(), True, "any")
        except ValueError:
            logger.warning(f"[ShuffledBasis] Compatibility with {variable=} was not extended")

        ## We try to extend other compatibilities
        for E in self.factors[0].compatible_endomorphisms():
            if all(E in factor.compatible_endomorphisms() for factor in self.factors[1:]):
                try:
                    self.set_homomorphism(E, self._extend_compatibility_E(E), True)
                except (ValueError, NotImplementedError) as e:
                    logger.info(f"[ShuffledBasis] Compatibility with endomorphism {E=} could not be extended (error extending)")
                    raise e
            else:
                logger.info(f"[ShuffledBasis] Compatibility with endomorphism {E=} could not be extended (not in all bases)")
        for D in self.factors[0].compatible_derivations():
            if all(D in factor.compatible_derivations() for factor in self.factors[1:]):
                try:
                    self.set_derivation(D, self._extend_compatibility_D(D), True)
                except (ValueError, NotImplementedError):
                    logger.info(f"[ShuffledBasis] Compatibility with derivation {D=} could not be extended")
            else:
                logger.info(f"[ShuffledBasis] Compatibility with derivation {E=} could not be extended (not in all bases)")

        ## TODO: Fill from here
        ## 1. (DONE) Call the super method of Factorial Basis with the necessary information.
        ##    1.1. Test the creation of elements (fix tests of __init__)
        ## 2. Create the compatibility w.r.t. "n". In particular, extend "any" operators.
        ## 3. Extend "homomorphisms"
        ## 4. Extend "derivations"
        ## 5. Fix all tests of __init__
        # raise NotImplementedError(f"[ShuffledBasis] Initialization not yet implemented")
    
        quasi_triangular_sequences = [factor.is_quasi_triangular() for factor in self.factors]
        if all(el != None for el in quasi_triangular_sequences):
            self._PSBasis__quasi_triangular = _ShuffledQuasiTriangular(quasi_triangular_sequences, self.cycle)

    @cached_property
    def counts(self) -> tuple:
        r'''
            Counting of appearances of each factor during a cycle
        '''
        return tuple([self.cycle.count(i) for i in range(self.nfactors)])
    
    @cached_method
    def extra(self, factor: int, remainder: int) -> int:
        r'''
            Counting function that computes how many times a factor has appear until the given remainder
        '''
        if remainder < 0 or remainder >= self.nsections:
            raise ValueError("Incompatible `remainder` for method `extra`.")
        if factor < 0 or factor >= self.nfactors:
            raise ValueError("Incompatible `factor` for method `extra`.")
        
        return self.cycle[:remainder].count(factor)

    @cached_property
    def indices(self) -> Sequence:
        r'''
            Computes the indices of each factor in the given element.
        '''
        def _element(k):
            K,r = k//len(self.cycle), k%len(self.cycle)
            extras = [self.extra(i, r) for i in range(self.nfactors)]
            return vector([K*c+e for c,e in zip(self.counts, extras)])
        return Sequence(_element, ZZ**self.nfactors, 1)
    
    def _element(self, *indices):
        index = indices[0] # we know the dimension is 1
        return prod(factor[i] for (factor, i) in zip(self.factors, self.indices(index)))

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
        r'''Property to get the factors of the :class:`ShuffledBasis`'''
        return self.__factors
    @property
    def cycle(self) -> tuple[int]:
        r'''Property to get the deciding cycle of the :class:`ShuffledBasis`'''
        return self.__cycle
    @property
    def nfactors(self) -> int:
        r'''
            Method to get the number of factors of the sieved basis.

            This method returns the number of factors which compose
            this :class:`ShuffledBasis`.

            OUTPUT:

            Number of factors of this :class:`ShuffledBasis`.
            
            TODO: add examples
        '''
        return len(self.factors)
    F = nfactors #: alias for the number of factors
    @property
    def nsections(self) -> int:
        r'''
            Method to get the number of sections of the sieved basis.

            This method returns the number of elements in the deciding cycle which 
            is the number of sections in which the :class:`ShuffledBasis` is divided.

            OUTPUT:

            Number of sections of this :class:`ShuffledBasis`.
            
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

            `\beta(n)`-factorial bases are always compatible with the multiplication by `\beta(n)`.
            This compatibility may come in `t_i` sections for the `i`-th factor of this basis. Then,
            for `T` such that `t_i` divides `TS(i)` for all `i = 0, \ldots, F-1`, the multiplication
            by `\beta(n)` is compatible with the :class:`ShuffledBasis` in `TC` sections.
        '''
        C = self.nsections; F = self.nfactors
        comps = [factor.compatibility(str(factor.gen())) for factor in self.factors]
        t = [comp.t for comp in comps]
        S = self.counts; s = self.extra

        ## Computing the number of sections
        T = 1
        while(any([not T*S[i]%t[i] == 0 for i in range(F)])): T += 1 # this always terminate at most with T = lcm(t_i)
        ## Computing the a's
        a = [T*S[i]//t[i] for i in range(F)]

        ## Computing the new compatibility coefficients
        new_coeffs = []
        for r in range(T*C):
            r_0, r_1 = ZZ(r).quo_rem(C); c_r1 = self.cycle[r_1]
            r_2, r_3 = ZZ(r_0*S[c_r1] + s(c_r1, r_1)).quo_rem(t[c_r1])
            new_coeffs.append([comps[c_r1][r_3,0].linear_subsequence(0, a[c_r1], r_2), comps[c_r1][r_3,1].linear_subsequence(0, a[c_r1], r_2)])

        return Compatibility(new_coeffs, 0, 1, T*C)
    
    def _extend_compatibility_E(self, E: str) -> Compatibility:
        r'''
            Method that extend the compatibility of an endomorphism `E`.
        '''
        C = self.nsections; F = self.nfactors
        divisions = [factor.compatible_division(E) for factor in self.factors]
        A = [comp.A for comp in divisions]
        t = [comp.t for comp in divisions]
        t_x = [factor.compatibility(str(factor.gen())).t for factor in self.factors]
        S = self.counts; s = self.extra

        ## Computing the number of sections
        T = 1
        while(any(chain((not T*S[i]%t[i] == 0 for i in range(F)),(not T*S[i]%t_x[i] == 0 for i in range(F))))): T += 1 # this always terminate at most with T = lcm(t_i, t_x_i)
        ## Computing the a's
        a = [T*S[i]//t[i] for i in range(F)]
        b = [T*S[i]//t_x[i] for i in range(F)]

        ## Computing the final `A`
        fA = 1
        while(any(fA*S[i] < A[i] for i in range(F))): fA+= 1
        D = [fA*S[i] - A[i] for i in range(F)]
        IB_D: list[list[tuple[Sequence]]] = [[factor.matrix_ItP(D[i]+1,r)[D[i]] for r in range(t_x[i])] for i,factor in enumerate(self.factors)]
        
        div_coeffs = []
        def poly_mult_list(list1: list, list2: list) -> list:
            output = [ZZ(0) for _ in range(len(list1)+len(list2)-1)]
            for (i,j) in product(range(len(list1)), range(len(list2))):
                output[i+j] += list1[i]*list2[j]
            return output
        for r in range(T*C):
            r_0, r_1 = ZZ(r).quo_rem(C)
            poly_coeffs = []
            for i in range(F): # we iterate through factors
                r_2, r_3 = ZZ(r_0*S[i] + s(i, r_1)).quo_rem(t[i])
                r_4, r_5 = ZZ(r_0*S[i] + s(i, r_1) - A[i]- D[i]).quo_rem(t_x[i])
                div_part = [divisions[i][r_3,j].linear_subsequence(0,a[i],r_2) for j in range(A[i]+1)]
                quot_part = [IB_D[i][r_5][k].linear_subsequence(0, b[i], r_4) for k in range(D[i]+1)]
                poly_coeffs.append(poly_mult_list(div_part, quot_part))
            div_coeffs.append(reduce(lambda l1,l2 : poly_mult_list(l1,l2), poly_coeffs, [ZZ(1)]))
            if len(div_coeffs[-1]) < sum(A) + sum(D) + 1: # the degree is smaller than expected
                div_coeffs[-1] += (sum(A) + sum(D) + 1 - len(div_coeffs[-1]))*[ZZ(0)]
        
        divisibility = DivisionCondition(div_coeffs, C*fA, base=self.base, sections=C*T)
        return divisibility.to_compatibility(self)
        
    def _extend_compatibility_D(self, D: str) -> Compatibility:
        r'''
            Method that extend the compatibility of a derivation `D`.

            This method uses the information in the factor basis to extend 
            the compatibility behavior of a derivation `D` to the 
            :class:`ShuffledBasis`.

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
        raise NotImplementedError(f"[ShuffledBasis] Extension of derivations not yet implemented")

    ###############################################################################
    ### Representation methods
    ###############################################################################
    def __repr__(self) -> str:
        return f"Shuffled Basis {self.cycle} of the basis:" + "".join([f"\n\t- {f}" for f in self.factors])

    def _latex_(self) -> str:
        return (r"\prod_{%s}" %self.cycle)  + "".join([f._latex_() for f in self.factors])

    # def increasing_polynomial(self, src: element.Element, diff : int = None, dst: int = None) -> element.Element:
    #     r'''
    #         Returns the increasing factorial for the factorial basis.

    #         This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial.factorial_basis.FactorialBasis`.
    #         See method :func:`~pseries_basis.factorial.factorial_basis.FactorialBasis.increasing_polynomial` for further information 
    #         in the description or the output.

    #         As a :class:`ShuffledBasis` is composed with several factors, we compute the difference between each element
    #         in the factors and compute the corresponding product of the increasing polynomials. 

    #         In this case, we consider the input given by `n = kF + r` where `F` is the number of sections of the 
    #         :class:`ShuffledBasis` (see method :func:`nsections`).

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
    # def increasing_basis(self, shift: int) -> ShuffledBasis:
    #     r'''
    #         Method to get the structure for the `n`-th increasing basis.

    #         This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial.factorial_basis.FactorialBasis`.
    #         See method :func:`~pseries_basis.factorial.factorial_basis.FactorialBasis.increasing_basis` for further information.

    #         For a :class:`ShuffledBasis`, the increasing basis is again a :class:`ShuffledBasis` of the increasing basis
    #         of its factors. Depending on the actual shift, the increasing basis may differ. Namely, if the shift is 
    #         `N = kF+j` where `F` is the number of sections of ``self`` and `B_i` are those factors, then the we can express 
    #         the increasing basis as a :class:`ShuffledBasis` again.

    #         INPUT:

    #         * ``shift``: value for the starting point of the increasing basis. It can be either
    #           the value for `N` or the tuple `(k,j)`.
              
    #         OUTPUT:

    #         A :class:`ShuffledBasis` representing the increasing basis starting at `N`.

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
    #     return ShuffledBasis(new_basis, new_cycle, var_name=str(self.universe.gens()[0]))
     
    # def compatible_division(self, operator: str | OreOperator) -> Divisibility:
    #     r'''
    #         Method to get the division of a polynomial by other element of the basis after an operator.

    #         This method *overrides* the implementation from class :class:`FactorialBasis`. See :func:`FactorialBasis.compatible_division`
    #         for a description on the output.

    #         For a :class:`ShuffledBasis`, since its elements are products of elements of other basis, we can compute this 
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
    #     raise NotImplementedError("_compatible_division_X not implemented for Shuffled Basis")

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

    # def is_quasi_eval_triangular(self) -> bool:
    #     return all(basis.is_quasi_eval_triangular() for basis in self.factors)

def ProductBasis(*factors: FactorialBasis) -> ShuffledBasis:
    r'''
        Factory for creating a special type of :class:`ShuffledBasis`: Product Basis.

        Namely, the `k=lm+j` element of the product of `m` basis, is the product of

        .. MATH::

            Q_k(n) = \prod_{i=1}^{j}P_{l+1}^{(j)}(n)\prod_{i=j+1}^{m}P_{l}^{(j)}(n).

        See the paper :doi:`10.1016/j.jsc.2022.11.002` for further information.

        INPUT:

        * ``factors``: list of :class:`FactorialBasis`.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: B1 = BinomialBasis; B2 = PowerBasis; B3 = FallingBasis(1,0,1)
            sage: #TODO ProductBasis(B1,B2).factors == (B1, B2) ## output: True
            sage: #TODO ProductBasis(B1,B2).nfactors() ## output: 2
            sage: #TODO ProductBasis([B1,B3,B2]).factors == (B1,B3,B2) ## output: True
            sage: #TODO ProductBasis([B1,B3,B2]).nfactors() ## output: 3

        This method is a simplification of a call to :class:`ShuffledBasis`. The following example
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
    if len(factors) == 1 and isinstance(factors[0], Collection): # we allow ProductBasis(B1,B2) and ProductBasis([B1,B2])
        factors = factors[0]
    return ShuffledBasis(factors, list(range(len(factors))))

class _ShuffledQuasiTriangular:
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
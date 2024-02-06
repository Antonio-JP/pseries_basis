r'''
    Sage package for Orthogonal Series Basis.

    TODO: review this file to check the compatibility with the derivative in general.
'''
from __future__ import annotations

from functools import reduce, lru_cache
from sage.all import cached_method, QQ, ZZ, SR #pylint: disable=no-name-in-module
from sage.categories.pushout import pushout
from sage.functions.orthogonal_polys import chebyshev_T, chebyshev_U, gegenbauer, hermite, jacobi_P, laguerre
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from typing import Any

# Local imports
from ..psbasis import PSBasis, Compatibility
from ..sequences import ConstantSequence, Sequence, ExpressionSequence, IdentitySequence, RationalSequence

###############################################################################
###
### FACTORIAL BASIS AND EXAMPLES
###
###############################################################################
class OrthogonalBasis(PSBasis):
    r'''
        Class representing an Orthogonal Polynomial Basis.

        A `beta(n)`-orthogonal polynomial basis is a specific type of Sequences basis where the elements
        are defined using a recurrence of order 2. 

        More precisely, a `\beta(n)`-factorial basis is a basis of sequences `B = \{P_k(n)\}` where 
        the `k`-th element is a polynomial w.r.t. `\beta(n)` of degree `k` such that 

        .. MATH::

            P_{k+1}(n) = (a_{k}\beta(n) + b_{k})P_{k}(n) - c_kP_{k-1}(n).

        This type of basis have special types of compatibilities. More precisely, they are 
        **always** compatible with the "multiplication by `\beta(n)`" operation. This is a special type
        of homomorphism, and always satisfies:

        .. MATH::

            \beta(n)P_k = \frac{1}{a_k}P_{k+1}(n) - \frac{b_k}{a_k}P_k(n) + \frac{c_k}{a_k}P_{k-1}(n).

        INPUT:

            * ``ak``: a sequence to be used for `a_k`. It can be a rational expression in some variable (see argument ``gamma``)
            * ``bk``: a sequence to be used for `b_k`. See argument ``ak``.
            * ``ck``: a sequence to be used for `c_k`. See argument ``ak``.
            * ``universe`` (optional): universe for the elements of the basis.
            * ``beta``: either ``None`` or a tuple ``(name, seq)``. This defines the sequence `beta(n)` and a name for it. If
              not given, it takes as default the values ``(`n`, n -> n)``.
            * ``gamma``: either ``None`` or a tuple ``(name, seq)``. This defines a sequence `\gamma(k)` such that `a_k`, `b_k` and `c_k`
              are built (if necessary) as :class:`RationalSequence` w.r.t. `gamma(k)`. By default, it takes the value ``(`k`, n -> n)``.
            * ``as_2seq`` (optional): sequence in 2 variables that will be use for generic purposes in :class:`PSBasis`.
    '''
    def __init__(self, ak: Sequence, bk: Sequence, ck: Sequence, universe = None, *, 
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
                try:
                    ak = RationalSequence(ak, [gamma[0]], universe, meanings=gamma[1], **kwds)
                except:
                    ak = ExpressionSequence(SR(ak), [gamma[0]], universe, meanings=gamma[1], **kwds)
        if not isinstance(bk, Sequence): 
            if universe != None:
                try:
                    bk = RationalSequence(bk, [gamma[0]], universe, meanings=gamma[1], **kwds)
                except:
                    bk = ExpressionSequence(SR(bk), [gamma[0]], universe, meanings=gamma[1], **kwds)
        if not isinstance(ck, Sequence): 
            if universe != None:
                try:
                    ck = RationalSequence(ck, [gamma[0]], universe, meanings=gamma[1], **kwds)
                except:
                    ck = ExpressionSequence(SR(ck), [gamma[0]], universe, meanings=gamma[1], **kwds)
        if not isinstance(ak, Sequence) or ak.dim != 1:
            raise TypeError(f"[FactorialBasis] The element a_k must be a univariate sequence or an expression in 'k'")
        if not isinstance(bk, Sequence) or bk.dim != 1:
            raise TypeError(f"[FactorialBasis] The element a_k must be a univariate sequence or an expression in 'k'")
        if not isinstance(ck, Sequence) or ck.dim != 1:
            raise TypeError(f"[FactorialBasis] The element a_k must be a univariate sequence or an expression in 'k'")
        universe = universe if universe != None else reduce(pushout, (ak.universe, bk.universe, ck.universe))
        
        self.__ak = ak.change_universe(universe)
        self.__bk = bk.change_universe(universe)
        self.__ck = ck.change_universe(universe)

        self.__poly_ring = PolynomialRing(universe, beta[0]) # this is the polynomial ring for the elements of the sequence
        self.__beta = beta; self.__gamma = gamma
        self.__gen = self.__poly_ring.gens()[0]

        @lru_cache
        def __get_element(k):
            if k < 0: return self.__poly_ring.zero()
            elif k == 0: return self.__poly_ring.one()
            else: return (self.ak(k-1)*self.__gen + self.bk(k-1))*__get_element(k-1) - self.ck(k-1)*__get_element(k-2) #pylint: disable=not-callable

        sequence = as_2seq if as_2seq != None else lambda k : self._RationalSequenceBuilder(__get_element(k))

        super().__init__(sequence, universe, _extend_by_zero=_extend_by_zero, **kwds)

        # We create now the compatibility with the multiplication by the variable generator
        self.set_compatibility(beta[0], Compatibility([[self.ck / self.ak, -self.bk / self.ak, 1/self.ak]], 1, 1, 1), True, "any")
        try:
            self.__Q, comp = self.derivation_compatibility()
            self.set_derivation("QDn", comp, True)
        except NotImplementedError:
            self.__Q = None

    def args_to_self(self) -> tuple[list, dict[str]]:
        return (
            [self.ak, self.bk, self.ck], 
            {"universe": self.base, 
             "beta": self.__beta, 
             "gamma": self.__gamma, 
             "as_2seq": self.as_2dim(),
             "_extend_by_zero": self._Sequence__extend_by_zero,
             **self.extra_info()["extra_args"]
             }
        )

    @property
    def ak(self): return self.__ak                  #: Sequence a_k from definition of Orthogonal basis.
    @property
    def bk(self): return self.__bk                  #: Sequence b_k from definition of Orthogonal basis.
    @property
    def ck(self): return self.__ck                  #: Sequence c_k from definition of Orthogonal basis.

    @property
    def derivation_factor(self): return self.__Q

    def gen(self): return self.__gen                #: Getter of the variable generator for the polynomial basis
    def poly_ring(self): return self.__poly_ring    #: Getter of the polynomial ring for the basis
        
    ##################################################################################
    ### METHODS FROM PSBASIS
    ##################################################################################
    def _scalar_basis(self, factor: Sequence) -> OrthogonalBasis:
        r'''
            Creates the scaled version of a :class:`OrthogonalBasis`.
        '''
        ## Getting the new universe and new quotient for changing the sequences
        new_universe = pushout(self.base, factor.universe)
        quotient_1 = factor.shift() / factor; quotient_2 = factor.shift(2) / factor

        ## Getting other arguments for the builder
        _, kwds = self.args_to_self()
        kwds["universe"] = new_universe

        ## Building the new basis
        output = OrthogonalBasis(self.ak*quotient_1, self.bk*quotient_1, self.ck * quotient_2.shift(-1), **kwds)
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
    ### METHODS FOR ORTHOGONAL POLYNOMIALS
    ##################################################################################
    @cached_method
    def differential_equation(self) -> tuple[Any,Any,Any]:
        r'''
            Method to get the second order differential equation for a Orthogonal basis.

            By definition, a set of Orthogonal polynomials satisfies a three term recurrence
            of the form  

            .. MATH::

                P_{n+1}(x) = (a_n x + b_n)P_n(x) - c_nP_{n-1}(x).

            This implies that the set also satisfies a second order differential equation. In fact,
            both representation are equivalent. This method computes the second order differential
            equation for the current Orthogonal basis.

            OUTPUT: 

            A triplet `(A(n),B(n),C(n)) \in \mathbb{Q}(n)[x]` such that, for any element `P_n(x)` of the basis, we have

            .. MATH::

                A(n)P_n(x)'' + B(n)P_n(x)' + C(n)P_n(x) = 0.

            TODO: add examples
        '''
        raise NotImplementedError(f"Method get_differential_equation not yet implemented")

    @cached_method
    def mixed_equation(self) -> tuple[Any,Any,Any,Any]:
        r'''
            Method to get a mixed relation between the shift and differential operators.

            By definition, a set of Orthogonal polynomials satisfies a three term recurrence
            of the form  

            .. MATH::

                P_{n+1}(x) = (a_n x + b_n)P_n(x) - c_nP_{n-1}(x).

            This implies that the set also satisfies a mixed difference-differential equation. In fact,
            both representation are equivalent. This method computes the mixed relation for the current 
            Orthogonal basis.

            OUTPUT: 

            A tuple `(A(n),B(n),C(n),D(n)) \in \mathbb{Q}(n)` such that, for any element `P_n(x)` of the basis, we have

            .. MATH::

                A(n)P_{n}(x)' = (B(n)x+C(n))P_n(x) + D(n)P_{n-1}(x).

            TODO: add examples

            WARNING: **this method is currently not implemented.**
        '''
        raise NotImplementedError("The mixed relation is not (yet) implemented in general")

    @cached_method
    def derivation_compatibility(self) -> tuple[Any,Compatibility]:
        r'''
            Method to get compatibility with the associated derivation.

            This method returns the compatibility of the Orthogonal basis with the 
            associated derivation with this basis. By definition, a set of Orthogonal 
            polynomials satisfies a three term recurrence of the form  

            .. MATH::

                P_{n+1}(x) = (a_n x + b_n)P_n(x) - c_nP_{n-1}(x).

            That leads to a second order differential equation (see method :func:`get_differential_equation`)
            of the form

            .. MATH::

                Q(x)P_n''(x) + R(x)P_n'(x) + S(n)P_n(x) = 0.

            This means that the operator `Q(x)\partial_x` is directly compatible with this basis. This method
            computes the compatibility with this operator.

            This method is abstract and may be implemented in all subclasses. If not 
            provided, the compatibility with the derivation will not be set, but no
            error will be raised. See also :func:`get_mixed_relation`.

            OUTPUT:

            The element `Q(x)` and a :class:`Compatibility` condition for the operator `Q(x)D_x`.
        '''
        raise NotImplementedError("The general first compatibility with derivation is not implemented")

def JacobiBasis(alpha, beta, universe=QQ):
    if universe is None: raise ValueError(f"[Jacobi] The universe can not be ``None``")
    if alpha not in universe or alpha <= -1:
        raise ValueError(f"[Jacobi] The value {alpha=} must be greater than -1.")
    if beta not in universe or beta <= -1:
        raise ValueError(f"[Jacobi] The value {beta=} must be greater than -1.")

    ak = f"(2*k + {alpha + beta} + 1)*(2*k + {alpha + beta} + 2)/(2*(k+1)*(k+{alpha+beta}+1))"
    bk = f"({alpha^2 - beta^2})*(2*k + {alpha + beta} + 1)/(2*(k+1)*(k+{alpha+beta}+1)*(2*k+{alpha+beta}))"
    ck = f"(k+{alpha})*(k+{beta})*(2*k + {alpha+beta}+2)/((k+1)*(k+{alpha+beta}+1)*(2*k+{alpha+beta}))"
        
    return OrthogonalBasis(ak,bk,ck,universe,as_2seq=ExpressionSequence(jacobi_P(SR('k'), alpha, beta, SR('n')), variables=['k','n'], universe=universe))
def GegenbauerBasis(_lambda, universe=QQ):
    if universe is None: raise ValueError(f"[Gegenbauer] The universe can not be ``None``")
    if _lambda not in universe or _lambda == 0 or _lambda <= -1/QQ(2):
        raise ValueError(f"[Gegenbauer] The value {_lambda=} must be a rational greater than -1/2 different from 0")
    
    ak = f"2*(k+{_lambda})/(k+1)"
    bk = 0
    ck = f"(k+2*{_lambda}-1)/(k+1)"

    return OrthogonalBasis(ak,bk,ck,universe,as_2seq=ExpressionSequence(gegenbauer(SR('k'), _lambda, SR('n')), variables=['k','n'], universe=universe))  
def LegendreBasis(universe=QQ):
    return JacobiBasis(0, 0, universe)
def TChebyshevBasis(universe=QQ):
    if universe is None: raise ValueError(f"[Chebyshev-T] The universe can not be ``None``")
    ak = Sequence(lambda k : 2 if k > 0 else 1, QQ)
    bk = 0
    ck = 1
    return OrthogonalBasis(ak,bk,ck,universe,as_2seq=ExpressionSequence(chebyshev_T(SR('k'), SR('n')), variables=['k','n'], universe=universe))  
def UChebyshevBasis(universe=QQ):
    if universe is None: raise ValueError(f"[Chebyshev-U] The universe can not be ``None``")
    ak = 2; bk = 0; ck = 1
    return OrthogonalBasis(ak,bk,ck,universe,as_2seq=ExpressionSequence(chebyshev_U(SR('k'), SR('n')), variables=['k','n'], universe=universe))  
def LaguerreBasis(alpha, universe=QQ):
    if universe is None: raise ValueError(f"[Laguerre] The universe can not be ``None``")
    if alpha not in QQ or alpha <= -1:
        raise ValueError(f"[Laguerre] The value {alpha=} must be a rational number greater than -1")
    
    ak = f"-1/(k+1)"
    bk = f"(2*k+{alpha+1})/(k+1)"
    ck = f"(k+{alpha})/(k+1)"
    
    return OrthogonalBasis(ak,bk,ck,universe,as_2seq=ExpressionSequence(laguerre(SR('k'), SR('n')), variables=['k','n'], universe=universe))  
def HermiteBasis(universe=QQ):
    if universe is None: raise ValueError(f"[Hermite] The universe can not be ``None``")
    ak = 2; bk = 0; ck = f"2*k"
    output = OrthogonalBasis(ak,bk,ck,universe,as_2seq=ExpressionSequence(hermite(SR('k'), SR('n')), variables=['k','n'], universe=universe))
    ## We set compatibility with derivative (special case)
    output.set_derivation("Dn", Compatibility([[RationalSequence(2*output.ore_var(), universe=universe), ConstantSequence(0, universe=universe)]], 1, 0, 1), True)
    return output
def HermitePBasis(universe=QQ):
    ak = 1; bk = 0; ck = f"k"
    output = OrthogonalBasis(ak,bk,ck,universe)  
    ## We set compatibility with derivative (special case)
    output.set_derivation("Dn", Compatibility([[RationalSequence(output.ore_var(), universe=universe), ConstantSequence(0, universe=universe)]], 1, 0, 1), True)
    return output

__all__ = ["OrthogonalBasis", "JacobiBasis", "GegenbauerBasis", "LegendreBasis", "TChebyshevBasis", "UChebyshevBasis", "LaguerreBasis", "HermiteBasis", "HermitePBasis"]
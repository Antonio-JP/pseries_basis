r'''
    Module to unify the interaction with the module :mod:`ore_algebra`.

    The module :mod:`ore_algebra` provide a good implementation of different Ore Operators, 
    that allow us to represent linear operators over the formal power series ring or over the 
    sequence ring.

    This module will provide several methods and classes to unify the access to the :mod:`ore_algebra`
    module within the package :mod:`pseries_basis`.
'''

from functools import lru_cache
from .sequences import Sequence, LambdaSequence

from ore_algebra import OreAlgebra
from ore_algebra.ore_operator import OreOperator

from sage.all import QQ, ZZ, prod, PolynomialRing
from sage.rings.polynomial.polynomial_element import is_MPolynomialRing, is_PolynomialRing

#############################################################################################
###
### METHODS TO CATEGORIZE ORE ALGEBRAS
###
#############################################################################################
def is_recurrence_algebra(_: OreAlgebra) -> bool:
    return False

__CACHE_REC_ALGEBRAS = {}
def get_recurrence_algebra(name_x : str = "x", name_shift : str = "E") -> OreAlgebra:
    r'''
        Method to get always the same ore algebra

        This method unifies the access for OreAlgebra to get the recurrence shift operator
        `x \mapsto x+1`. The method allows to provide the names for the inner variable `x` 
        and the shift operator.

        INPUT:

        * ``name_x``: string with the name of the inner variable.
        * ``name_shift``: string with the name of the shift operator for the algebra.

        A recurrence algebra for which the two generators can be extract by 
        ``x = *.base().gens()[0]`` and ``E = *.gens()[0]``.
    '''
    if not (name_x, name_shift) in __CACHE_REC_ALGEBRAS:
        PR = PolynomialRing(QQ, name_x).fraction_field(); x = PR.gens()[0]
        __CACHE_REC_ALGEBRAS[(name_x, name_shift)] = OreAlgebra(PR, (name_shift, lambda p : p(x=x+1), lambda p : 0))
    
    return __CACHE_REC_ALGEBRAS[(name_x, name_shift)]

#############################################################################################
###
### METHODS INVOLVING ORE ALGEBRAS AND OTHER STRUCTURES
###
#############################################################################################
def apply_operator_to_seq(operator : OreOperator, sequence : Sequence) -> Sequence:
    r'''
        Method to apply an operator to a sequence.
        
        This method will be similar to the usual __call__ method from ``ore_algebra``, but with sequences that
        are given as functions.
        
        INPUT:
        
        * ``operator``: and operator with 1 generator (the shift) over the polynomial ring `\mathbb{R}[x]`.
        * ``seq``: a sequence in functional format. This means that we can call it with integer values and we 
          obtain the values of the sequence at each point.
          
        OUTPUT:
        
        A sequence in function format.
    '''
    if len(operator.parent().gens()) > 1:
        raise TypeError("Only ore operators with 1 generator are allowed: we assume is the natural shift")
    coefficients = operator.coefficients(sparse=False)
    # univariate polynomial base
    if is_PolynomialRing(operator.parent().base()):
        v = operator.parent().base().gens()[0]
        R = operator.parent().base().base()
        gen = lambda i : sum(coefficients[j](**{str(v):i})*sequence[i+j] for j in range(len(coefficients)))
    else: # we assume they are all constants
        R = operator.parent().base()
        gen = lambda i : sum(coefficients[j]*sequence[i+j] for j in range(len(coefficients)))
    
    return LambdaSequence(gen, R)

def required_init(operator) -> int:
    r'''
        Method to compute the number of required initial values for a sequence.

        When given a recurrence operator, we need some initial conditions to completely define 
        a solution to the recurrence operator. This method computes the maximal index we need to compute
        in order to have a fully defined sequence.
    '''
    return max(-min([0]+[el[0]-1 for el in operator.polynomial().lc().roots() if el[0] in ZZ]), operator.order())

def eval_ore_operator(operator, ring=None,**values):
    r'''
        Method to evaluate ore operators
        
        This method evaluate operators from ``ore_algebra`` as they are polynomials. This allows to change the name 
        of the generators to try a iterative approach.
    '''
    gens = [str(el) for el in operator.parent().gens()]
    outer_vals = {el : values.get(el, 0) for el in gens}
    inner_vals = {el : values[el] for el in values if (not el in outer_vals)}
    monomials,coefficients = poly_decomp(operator.polynomial())
    coefficients = [el(**inner_vals) for el in coefficients]
    monomials = [prod(
        outer_vals[str(g)]**(m.degree(g)) for g in operator.polynomial().parent().gens()
    ) for m in monomials]
    result = sum(coefficients[i]*monomials[i] for i in range(len(monomials)))
    if ring != None:
        return ring(result)
    return result

#############################################################################################
###
### SOME CLASSES RELATING WITH ORE ALGEBRAS
###
#############################################################################################
def solution(operator, init, check_init=True) -> Sequence:
    r'''
        Method to generate a :class:`Sequence` solution to a recurrence operator

        This method receives a recurrence operator and a list of initial conditions and creates a :class:`Sequence`
        object that represent this exact solution. If not enough initial values are given, we raise an exception.

        INPUT:

        * ``operator``: a recurrence operator. We assume it only has 1 generator and it is the natural shift.
        * ``init``: list or tuple of initial values for the sequence.
        * ``check_init``: flag indicating whether to check or not the consistency of ``init`` with the operator given
          (mainly used if the data provided is more than necessary, see method :func:`required_init`)

        OUTPUT:

        A :class:`Sequence` with the solution to ``operator`` and initial values given by ``init``.
    '''
    d = operator.order()
    required = required_init(operator)
    if len(init) < required:
        raise ValueError(f"More data ({required}) is needed")
        
    from_init = required if check_init else len(init)
    @lru_cache
    def __aux_sol(n):
        if n < 0:
            return 0
        elif n < from_init:
            return init[n]
        else:
            coeffs = operator.polynomial().coefficients(False)
            lc = coeffs.pop()
            return -sum(__aux_sol(n-d+i)*coeffs[i](n-d) for i in range(operator.order()))/lc(n-d)
    return __aux_sol

class OreSequence(Sequence):
    r'''
        Class to represent a sequence defined by a linear operator. This will include
        C-finite, D-finite, Q-finite and similar type of sequences.

        These sequences are always defined with a linear operator acting (somehow) on the 
        sequences and some initial onditions. 

        TODO: Implement or use other class for this idea
    '''
    pass

####################################################################################################
###
### AUXILIARY METHODS
###
####################################################################################################
def poly_decomp(polynomial):
    r'''
        Method that splits a polynomial into a lists of monomials and a list of coefficients indexed in 
        such a way that summing thorugh both lists gives the original polynomial.
        
        This method unifies the behavior or Univariate and Multivariate polynomials.
    '''
    if is_PolynomialRing(polynomial.parent()):
        g = polynomial.parent().gens()[0]
        d = polynomial.degree()
        monomials = [g**i for i in range(d+1)]
        coefficients = polynomial.coefficients(False)
        # we clean the zeros to have a sparse representation
        monomials = [monomials[i] for i in range(d+1) if coefficients[i] != 0]
        coefficients = [c for c in coefficients if c != 0]
    elif is_MPolynomialRing(polynomial.parent()):
        monomials = polynomial.monomials()
        coefficients = polynomial.coefficients()
    else:
        raise TypeError("The input must be a polynomial")
    return monomials, coefficients
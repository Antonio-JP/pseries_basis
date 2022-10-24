r'''
    Module to define the sequences that are defined using some operator.

    A linear operator `L : \mathbb{K}[[x]] \riaghtarrow \mathbb{K}[[x]]` can define an 
    element of the power series ring by setting `L \cdot y(x) = 0` and providing some 
    initial conditions (namely, `y(0), y'(0)`, etc.).

    In SageMath these operators can be easilily represented using the module :mod:`ore_algebra`.
    However, these operators have a critical limitation: all the generators for these Ore Algebras
    must commute. For those cases where this is not true, we can use the module 
    :mod:`~pseries_basis.misc.noncom_rings`.

    This module offers a unified access to these two types of operators and will allow other classes
    in :mod:`pseries_basis`.
'''

try: # python 3.9 or higher
    from functools import cache
except ImportError: #python 3.8 or lower
    from functools import lru_cache as cache
from typing import Any, Tuple, Union # pylint: disable=unused-import

from ore_algebra.ore_algebra import OreAlgebra, OreAlgebra_generic
from ore_algebra.ore_operator import OreOperator

from sage.all import QQ, ZZ, prod, lcm
from sage.categories.fields import Fields
from sage.rings.polynomial.polynomial_ring import PolynomialRing_field, is_PolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.fraction_field import FractionField_1poly_field
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing

from .noncom_rings import OperatorAlgebra, OperatorAlgebra_element, OperatorAlgebra_generic
from .sequences import Sequence, LambdaSequence

_Fields = Fields.__classcall__(Fields)

#############################################################################################
###
### METHODS TO CATEGORIZE ALGEBRAS OF OPERATORS
###
#############################################################################################
def is_recurrence_algebra(algebra: Union[OreAlgebra_generic, OperatorAlgebra_generic]) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` has the first operator a shift operator.
    '''
    return (isinstance(algebra, OreAlgebra_generic) and algebra.is_S()) or isinstance(algebra, OperatorAlgebra_generic)

def is_double_recurrence_algebra(algebra: Union[OreAlgebra_generic, OperatorAlgebra_generic]) -> bool:
    r'''
        Method to check whether if an :class:`OreAlbegra_generic` is a double directional operator ring.

        This method will take a :class:`OreAlgebra_generic` and will check whether it is a recurrence algebra with 
        two difference operators that are the inverse of each other. For doing so, we only need to consider its 
        effect over the generator of the inner ring.
    '''
    if isinstance(algebra, OreAlgebra_generic):
        # we have exactly 2 generators
        if algebra.ngens() != 2:
            return False
        # at least one is the recurrence operator
        if all(not algebra.is_S(i) for i in [0,1]):
            return False
        
        # they are the inverse of each other
        O, OI = algebra.gens()
        return all((OI*O*v).coefficients() == [v] for v in algebra.base().base().gens())
    elif isinstance(algebra, OperatorAlgebra_generic):
        gens = algebra.gens()
        return any((gens[j]*gens[i]).canonical() == 1 for (i,j) in sum([[(i,j) for j in range(i+1,len(gens))] for i in range(len(gens))],[]))
    return False

def is_differential_algebra(algebra: Union[OreAlgebra_generic, OperatorAlgebra_generic]) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` has the first operator a differential operator.
    '''
    return isinstance(algebra, OreAlgebra_generic) and algebra.is_D()

def is_based_polynomial(algebra: Union[OreAlgebra_generic, OperatorAlgebra_generic]) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` has the base ring as a polynomial ring.
    '''
    return any(is_poly(algebra.base()) for is_poly in (is_PolynomialRing, is_MPolynomialRing))

def is_based_field(algebra: Union[OreAlgebra_generic, OperatorAlgebra_generic]) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` has the base ring as a polynomial ring.
    '''
    return algebra.base().is_field() and (algebra.base().base_ring() != algebra.base())

def has_variable(algebra, name):
    r'''
        Method to check whether an algebra has a generator or not.
    '''
    current = algebra
    while (not 1 in current.gens()) and len(current.gens()) > 0:
        gnames = [str(el) for el in current.gens()]
        if name in gnames: return True, current(name)
        current = current.base()
    
    return False, None

#############################################################################################
###
### METHODS TO CREATE ORE ALGEBRAS
###
#############################################################################################
__CACHE_POLY_ALGEBRAS = {}
def get_polynomial_algebra(name_x: str = "x", base : _Fields.parent_class = QQ) -> Tuple[PolynomialRing_field, Any]:
    r'''
        Method to get always the same Polynomial Ring

        This methods unifies the creation of polynomial rings for their use in the package
        :mod:`ore_algebra`, so there is less problems when comparing variables generated 
        in these polynomial rings.

        INPUT:

        * ``name_x``: string with the name of the variable for the polynomial ring.
        * ``base``: the base field for the polynomial ring.

        OUTPUT:

        A tuple `(R, x)` where `R = \mathbb{Q}[x]`.
    '''
    if not (name_x, base) in __CACHE_POLY_ALGEBRAS:
        Px = PolynomialRing(base, name_x); x = Px(name_x)
        __CACHE_POLY_ALGEBRAS[(name_x, base)] = (Px, x)
    
    return __CACHE_POLY_ALGEBRAS[(name_x, base)]

def get_rational_algebra(name_x: str = "x", base : _Fields.parent_class = QQ) -> Tuple[FractionField_1poly_field, Any]:
    r'''
        Method to get always the same fraction field of a Polynomial Ring

        This methods unifies the creation of field of rational functions for their use in the package
        :mod:`ore_algebra`, so there is less problems when comparing variables generated 
        in these polynomial rings.

        INPUT:

        * ``name_x``: string with the name of the variable for the polynomial ring.
        * ``base``: the base field for the rational function field.

        OUTPUT:

        A tuple `(R, x)` where `R = \mathbb{Q}(x)`.
    '''
    R, x = get_polynomial_algebra(name_x, base)
    return (R.fraction_field(), R.fraction_field()(x))

__CACHE_REC_ALGEBRAS = {}
def get_recurrence_algebra(name_x : str = "x", name_shift : str = "E", rational : bool = True, base : _Fields.parent_class = QQ) -> Tuple[OreAlgebra_generic, Tuple[Any, Any]]:
    r'''
        Method to get always the same ore algebra

        This method unifies the access for OreAlgebra to get the recurrence shift operator
        `x \mapsto x+1`. The method allows to provide the names for the inner variable `x` 
        and the shift operator.

        INPUT:

        * ``name_x``: string with the name of the inner variable.
        * ``name_shift``: string with the name of the shift operator for the algebra.
        * ``rational``: boolean (``True`` by default) deciding whether the base ring is 
          the field of rational functions or a polynomial ring.
        * ``base``: the base field for the polynomial ring w.r.t. ``name_x``.

        OUTPUT:

        A tuple `(A, (x, E))` where `A` is the corresponding recurrence algebra, `x` is the 
        variable of the inner variable and `E` the recurrence operator with `E(x) = x+1`.
    '''
    if not (name_x, name_shift, rational, base) in __CACHE_REC_ALGEBRAS:
        PR, x = get_rational_algebra(name_x, base) if rational else get_polynomial_algebra(name_x, base)
        OE = OreAlgebra(PR, (name_shift, lambda p : p(**{str(x) : x+1}), lambda _ : 0)); E = OE.gens()[0]
        __CACHE_REC_ALGEBRAS[(name_x, name_shift, rational, base)] = (OE, (x,E)) 
    
    return __CACHE_REC_ALGEBRAS[(name_x, name_shift, rational, base)]

__CACHE_DREC_ALGEBRAS = {}
def get_double_recurrence_algebra(name_x : str = "x", name_shift : str = "E", rational : bool = True, base : _Fields.parent_class = QQ) -> Tuple[OreAlgebra_generic, Tuple[Any, Any, Any]]:
    r'''
        Method to get always the same ore algebra

        This method unifies the access for OreAlgebra to get the recurrence shift operator
        `x \mapsto x+1` and its inverse in an :class:`OreAlgebra_generic`. The method allows to provide the names for the inner variable `x` 
        and the shift operator. The inverse shift will be named by adding an `i` to the name of the shift.

        INPUT:

        * ``name_x``: string with the name of the inner variable.
        * ``name_shift``: string with the name of the shift operator for the algebra.
        * ``rational``: boolean (``True`` by default) deciding whether the base ring is 
          the field of rational functions or a polynomial ring.
        * ``base``: the base field for the polynomial ring w.r.t. ``name_x``.

        OUTPUT:

        A tuple `(A, (x, E, Ei))` where `A` is the corresponding recurrence algebra, `x` is the 
        variable of the inner variable and `E` the recurrence operator with `E(x) = x+1` and 
        `Ei` is the inverse recurrence operator, i.e., `Ei(x) = x-1`.
    '''
    if not (name_x, name_shift, rational, base) in __CACHE_DREC_ALGEBRAS:
        PR, x = get_rational_algebra(name_x, base) if rational else get_polynomial_algebra(name_x, base)
        OE = OreAlgebra(PR, (name_shift, lambda p : p(**{str(x) : x+1}), lambda _ : 0), (name_shift+"i", lambda p : p(**{str(x) : x-1}), lambda _ : 0)); E, Ei = OE.gens()
        __CACHE_DREC_ALGEBRAS[(name_x, name_shift, rational, base)] = (OE, (x,E,Ei)) 
    
    return __CACHE_DREC_ALGEBRAS[(name_x, name_shift, rational, base)]

__CACHE_DER_ALGEBRAS = {}
def get_differential_algebra(name_x : str = "x", name_der : str = "Dx", rational : bool = True, base : _Fields.parent_class = QQ) -> Tuple[OreAlgebra_generic, Tuple[Any, Any]]:
    r'''
        Method to get always the same ore algebra

        This method unifies the access for OreAlgebra to get the derivation operator
        `D(f(x)) = f'(x)`. The method allows to provide the names for the inner variable `x` 
        and the given derivation.

        INPUT:

        * ``name_x``: string with the name of the inner variable.
        * ``name_der``: string with the name of the derivation for the algebra.
        * ``rational``: boolean (``True`` by default) deciding whether the base ring is 
          the field of rational functions or a polynomial ring.
        * ``base``: the base field for the polynomial ring w.r.t. ``name_x``.
        
        OUTPUT:

        A tuple `(A, (x, D))` where `A` is the corresponding recurrence algebra, `x` is the 
        variable of the inner variable and `D` the recurrence operator with `D(x) = 1`.
    '''
    if not (name_x, name_der, rational, base) in __CACHE_DER_ALGEBRAS:
        PR, x = get_rational_algebra(name_x, base) if rational else get_polynomial_algebra(name_x, base)
        OD = OreAlgebra(PR, (name_der, lambda p : p, lambda p : p.derivative(x))); D = OD.gens()[0]
        __CACHE_DER_ALGEBRAS[(name_x, name_der, rational, base)] = (OD, (x, D))
    
    return __CACHE_DER_ALGEBRAS[(name_x, name_der, rational, base)]

__CACHE_QSHIFT_ALGEBRA = {}
def get_qshift_algebra(name_shift : str = "S", name_q : str = "q", name_Q : str = "Q", base : _Fields.parent_class = QQ) -> Tuple[OperatorAlgebra_generic, Tuple[Any,Any,Any,Any]]:
    r'''
        Method to get always the same algebra for recurrence `q`-equations.

        This method allows to alwyas get the same structure for representing recurrence `q`-operators. These operators are generated by two main operators:

        .. MATH::

            S : x \maptsto x+1,\qquad Q: x \mapsto qx

        These operators applied to formal power `q`-series, i.e., elements in `\mathbb{K}(q)[[x]]`.

        INPUT:

        * ``name_shift``: ("S" by default) name for the natural shift operator `S`.
        * ``name_q``: ("q" by default) name for the `q`-parameter of the `q`-operators.
        * ``name_Q``: ("Q" by default) name for the `q`-shift operator.
        * ``base``: (`\mathbb{Q}` by default) base field to define the formal power series ring.

        OUTPUT:

        The output will be a tuple `(R, (q, Q, S))` where `R` is the generated operators algebra, `q` is the variable for `q` in `R`,
        `S` is the representation of the shift in `R` and `Q` is the `q`-shift representation in `Q`.
    '''
    if not (name_shift, name_q, name_Q, base) in __CACHE_QSHIFT_ALGEBRA:
        relations = [(name_shift, name_Q, f"{name_q}*{name_Q}*{name_shift}")]
        with_q, q = has_variable(base, name_q)
        if not with_q:
            base = PolynomialRing(base, name_q).fraction_field()

        OA = OperatorAlgebra(base, names = (name_Q, name_shift), relations = relations)
        Q,S = OA.gens()

        __CACHE_QSHIFT_ALGEBRA[(name_shift, name_q, name_Q, base)] = (OA, (q, Q, S))

    return __CACHE_QSHIFT_ALGEBRA[(name_shift, name_q, name_Q, base)]

__CACHE_DQSHIFT_ALGEBRA = {}
def get_double_qshift_algebra(name_shift : str = "S", name_q : str = "q", name_Q : str = "Q", base : _Fields.parent_class = QQ) -> Tuple[OperatorAlgebra_generic, Tuple[Any,Any,Any,Any]]:
    r'''
        Method to get always the same algebra for recurrence `q`-equations.

        This method allows to alwyas get the same structure for representing recurrence `q`-operators. These operators are generated by two main operators:

        .. MATH::

            S : x \maptsto x+1,\qquad Q: x \mapsto qx

        These operators applied to formal power `q`-series, i.e., elements in `\mathbb{K}(q)[[x]]`.

        INPUT:

        * ``name_shift``: ("S" by default) name for the natural shift operator `S`.
        * ``name_q``: ("q" by default) name for the `q`-parameter of the `q`-operators.
        * ``name_Q``: ("Q" by default) name for the `q`-shift operator.
        * ``base``: (`\mathbb{Q}` by default) base field to define the formal power series ring.

        OUTPUT:

        The output will be a tuple `(R, (q, Q, S, Si))` where `R` is the generated operators algebra, `q` is the variable for `q` in `R`,
        `S` is the representation of the shift in `R`, `Si` is the representation of the inverse of `S` and `Q` is the `q`-shift representation in `Q`.
    '''
    if not (name_shift, name_q, name_Q, base) in __CACHE_DQSHIFT_ALGEBRA:
        name_iS = f"{name_shift}i"
        relations = [
            (name_shift, name_Q, f"{name_q}*{name_Q}*{name_shift}"),
            (name_iS, name_Q, f"(1/{name_q})*{name_Q}*{name_iS}"),
            (name_iS, name_shift, f"1")
        ]
        with_q, q = has_variable(base, name_q)
        if not with_q:
            base = PolynomialRing(base, name_q).fraction_field()

        OA = OperatorAlgebra(base, names = (name_Q, name_shift, name_iS), relations = relations)
        Q,S,Si = OA.gens()

        __CACHE_DQSHIFT_ALGEBRA[(name_shift, name_q, name_Q, base)] = (OA, (q, Q, S, Si))

    return __CACHE_DQSHIFT_ALGEBRA[(name_shift, name_q, name_Q, base)]

def is_q_operator_algebra(algebra, name_q : str = "q"):
    r'''
        Method to check whether an algebra behaves like one with Q-shifts operators.
    '''
    if isinstance(algebra, OperatorAlgebra_generic):
        with_q, q = has_variable(algebra.base(), name_q)
        if not algebra.is_complete_commutation() or not with_q:
            print("Error in format")
            return None, None
        gens = algebra.gens()
        if len(gens) < 2:
            print("Too few generators")
            return None, None
        if len(gens) >= 2:
            #we need the first to be `Q` and the second to be `S`
            if (gens[1]*gens[0]).canonical() != q*gens[0]*gens[1]:
                print("Not valid commutation rule for 0 and 1")
                return None, None
        if len(gens) >= 3:
            # now the third has to be S^-1
            if (gens[2]*gens[1]).canonical() != algebra.one() or (gens[2]*gens[0]).canonical() != (1/q)*gens[0]*gens[2]:
                print("Not valid commutation rule for 2")
                return None, None
        if len(gens) > 3:
            print("Too many generators")
            return None, None

        return len(gens) == 3, (algebra, tuple([q, *gens]))
    
    return None, None
#############################################################################################
###
### METHODS INVOLVING ORE ALGEBRAS AND OTHER STRUCTURES
###
#############################################################################################
def apply_operator_to_seq(operator : Union[OreOperator,OperatorAlgebra_element], sequence : Sequence) -> Sequence:
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
    if isinstance(operator, OreOperator):
        if len(operator.parent().gens()) > 1:
            raise TypeError("Only ore operators with 1 generator are allowed: we assume is the natural shift")
        coefficients = operator.coefficients(sparse=False)
        
        E = operator.parent().gens()[0]
        v = None

        for el in operator.parent().base().gens(): # looking for the variable where the shift acts
            if E*el != el*E:
                v = el
                break

        # found the shift variable
        if v != None:
            v = operator.parent().base().gens()[0]
            R = operator.parent().base().base()
            gen = lambda i : sum(coefficients[j](**{str(v):i})*sequence[i+j] for j in range(len(coefficients)))
        else: # all the base ring are constants
            R = operator.parent().base()
            gen = lambda i : sum(coefficients[j]*sequence[i+j] for j in range(len(coefficients)))
        
        return LambdaSequence(gen, R)
    elif isinstance(operator, OperatorAlgebra_element):
        is_double, algebra = is_q_operator_algebra(operator.algebra())
        if is_double is None:
            raise ValueError(f"No valid q-algebra found for {operator}")
        elif is_double:
            _,(q,Q,S,Si) = algebra
            QPower = LambdaSequence(lambda n : q**n, operator.parent().base(), allow_sym=True)
            actions = {
                str(S) : lambda an : an.shift(1), 
                str(Si) : lambda an: an.shift(-1), 
                str(Q) : lambda an : QPower * an
            }
        else:
            _,(q,Q,S) = algebra
            QPower = LambdaSequence(lambda n : q**n, operator.parent().base(), allow_sym=True)
            actions = {str(S) : lambda an : an.shift(1), str(Q) : lambda an : QPower * an}
                    
        return operator.apply(sequence, actions)
    else:
        raise TypeError(f"Type {operator.__class__} not valid for method 'apply_operator_to_seq'")

def required_init(operator : Union[OreOperator,OperatorAlgebra_element]) -> int:
    r'''
        Method to compute the number of required initial values for a sequence.

        When given a recurrence operator, we need some initial conditions to completely define 
        a solution to the recurrence operator. This method computes the maximal index we need to compute
        in order to have a fully defined sequence.
    '''
    if isinstance(operator, OreOperator):
        if is_based_field(operator.parent()): # rational function case
            _, coeffs = poly_decomp(operator.polynomial())
            to_check = lcm([el.denominator() for el in coeffs] + [operator.polynomial().lc().numerator()])
        elif is_based_polynomial(operator.parent()):
            to_check = operator.polynomial().lc()
        return max(-min([0]+[el[0]-1 for el in to_check.roots() if el[0] in ZZ]), operator.order())
    elif isinstance(operator, OperatorAlgebra_element):
        is_double, _ = is_q_operator_algebra(operator.parent())
        if is_double is None:
            raise TypeError(f"No valid q-algebra found for {operator}")
        elif is_double:
            _, dS, dSi = operator.degrees()
            return dS + dSi
        else:
            _, dS = operator.degrees()
            return dS
    else:
        raise TypeError(f"Type {operator.__class__} not valid for method 'required_init'")

def eval_ore_operator(operator : OreOperator, ring=None,**values):
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
def solution(operator : Union[OreOperator, OperatorAlgebra_element], init, check_init=True) -> Sequence:
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
    if isinstance(operator, OreOperator):
        d = operator.order()
        required = required_init(operator)
        universe = operator.parent().base()
        if len(init) < required:
            raise ValueError(f"More data ({required}) is needed")
            
        from_init = required if check_init else len(init)
        @cache
        def __aux_sol(n):
            if n < 0:
                return 0
            elif n < from_init:
                return init[n]
            else:
                coeffs = operator.polynomial().coefficients(False)
                lc = coeffs.pop()
                return -sum(__aux_sol(n-d+i)*coeffs[i](n-d) for i in range(operator.order()))/lc(n-d)
    elif isinstance(operator, OperatorAlgebra_element):
        is_double, algebra = is_q_operator_algebra(operator.parent())
        operator = operator.canonical()
        required = required_init(operator)
        universe = operator.parent().base()
        if is_double is None:
            raise TypeError(f"No valid q-algebra found for {operator}") 
        elif is_double:
            _,(q,_,S,Si) = algebra
            dS = operator.degree(S); dSi = operator.degree(Si) 
            dict_as_poly = {
                **{i : operator.coefficient(S**i) for i in range(1,operator.degree(S)+1)}, 
                **{-i : operator.coefficient(Si**i) for i in range(1,operator.degree(Si)+1)},
                0 : operator.coefficient({str(S) : 0})
            }
        else:
            _,(q,_,S) = algebra
            dS = operator.degree(S); dSi = 0
            dict_as_poly = {
                **{i : operator.coefficient(S**i) for i in range(1,operator.degree(S)+1)}, 
                0 : operator.coefficient({str(S) : 0})
            }
        from_init = required if check_init else len(init)
        def _eval_coeff(i, n):
            res = dict_as_poly[i](Q=q**n)
            if not res in operator.parent().base():
                res = res.constant_coefficient()
            return res
        @cache
        def __aux_sol(n):
            if n < 0:
                return 0
            elif n < from_init:
                return init[n]
            else:
                return -sum(_eval_coeff(i, n-dS)*__aux_sol(n-dS+i) for i in range(-dSi, dS))/_eval_coeff(dS, n-dS)
    else:
        raise TypeError(f"Type {operator.__class__} not valid for method 'solution'")

    return LambdaSequence(__aux_sol, universe)

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

__all__ = ["solution", "OreSequence"]
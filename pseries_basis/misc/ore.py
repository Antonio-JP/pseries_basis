r'''
    Module to define the sequences that are defined using some operator.

    A linear operator `L : \mathbb{K}[[x]] \rightarrow \mathbb{K}[[x]]` can define an 
    element of the power series ring by setting `L \cdot y(x) = 0` and providing some 
    initial conditions (namely, `y(0), y'(0)`, etc.).

    In SageMath these operators can be easily represented using the module :mod:`ore_algebra`.
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

from .sequences import Sequence, LambdaSequence

_Fields = Fields.__classcall__(Fields)

#############################################################################################
###
### METHODS TO CATEGORIZE ALGEBRAS OF OPERATORS
###
#############################################################################################
def is_recurrence_algebra(algebra: OreAlgebra_generic) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` has the first operator a shift operator.
    '''
    return (isinstance(algebra, OreAlgebra_generic) and algebra.ngens() == 1 and algebra.is_S() != False)

def gens_recurrence_algebra(algebra: OreAlgebra_generic) -> Tuple[Any, Any]:
    v = None
    S, = algebra.gens()
    for el in algebra.base().gens(): # looking for the variable where the shift acts
        if S*el != el*S:
            v = el
            break
    return (v, S)

def is_double_recurrence_algebra(algebra: OreAlgebra_generic) -> bool:
    r'''
        Method to check whether if an :class:`OreAlgebra_generic` is a double directional operator ring.

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
    return False

def gens_double_recurrence_algebra(algebra: OreAlgebra_generic) -> Tuple[Any, Any, Any]:
    v = None
    S,Si = algebra.gens()
    for el in algebra.base().gens(): # looking for the variable where the shift acts
        if S*el != el*S:
            v = el
            break
    return (v, S, Si)

def is_differential_algebra(algebra: OreAlgebra_generic) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` has the first operator a differential operator.
    '''
    return isinstance(algebra, OreAlgebra_generic) and algebra.is_D()

def is_q_operator_algebra(algebra : OreAlgebra_generic, name_q : str = "q") -> bool:
    r'''
        Method to check whether an algebra behaves like one with Q-shifts operators.
    '''
    with_q, _ = has_variable(algebra.base(), name_q)
    return with_q and isinstance(algebra, OreAlgebra_generic) and algebra.ngens() == 1 and algebra.is_Q() != False

def gens_q_operator_algebra(algebra: OreAlgebra_generic, name_q : str = "q") -> Tuple[Any,Any,Any]:
    _, q = has_variable(algebra.base(), name_q)
    S, = algebra.gens()
    v = None
    for el in algebra.base().gens(): # looking for the variable where the shift acts
        applied = S*el
        if len(applied.coefficients()) == 1 and applied.coefficients()[0] == q*el:
            v = el
            break
    return (q, v, S)

def is_double_q_operator_algebra(algebra : OreAlgebra_generic, name_q : str = "q"):
    r'''
        Method to check whether an algebra behaves like one with Q-shifts operators.
    '''
    if isinstance(algebra, OreAlgebra_generic):
        with_q, q = has_variable(algebra.base(), name_q)
        if with_q and algebra.ngens() == 2:
            S, Si = algebra.gens()
            Q = None
            for el in algebra.base().gens():
                applied = S*el
                if len(applied.coefficients()) == 1 and applied.coefficients()[0] == q*el:
                    Q = applied
            if Q != None and (not (S*Si*Q).coefficients()[0] != Q or (S*Q).coefficients()[0] != q*Q):
                return False
            return True
    return False

def gens_double_q_operator_algebra(algebra: OreAlgebra_generic, name_q : str = "q") -> Tuple[Any,Any,Any,Any]:
    _, q = has_variable(algebra.base(), name_q)
    S,Si = algebra.gens()
    v = None
    for el in algebra.base().gens(): # looking for the variable where the shift acts
        applied = S*el
        if len(applied.coefficients()) == 1 and applied.coefficients()[0] == q*el:
            v = el
            break
    return (q, v, S, Si)

def is_based_polynomial(algebra: OreAlgebra_generic) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` has the base ring as a polynomial ring.
    '''
    return any(is_poly(algebra.base()) for is_poly in (is_PolynomialRing, is_MPolynomialRing))

def is_based_field(algebra: OreAlgebra_generic) -> bool:
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
        gen_names = [str(el) for el in current.gens()]
        if name in gen_names: return True, current(name)
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
def get_qshift_algebra(name_x : str = "x", name_q = "q", name_qshift : str = "E", rational : bool = True, base : _Fields.parent_class = QQ) -> Tuple[OreAlgebra_generic, Tuple[Any, Any]]:
    r'''
        Method to get always the same ore algebra

        This method unifies the access for OreAlgebra to get the recurrence `q`-shift operator
        `x \mapsto qx`. The method allows to provide the names for the inner variable `x` 
        and the shift operator.

        INPUT:

        * ``name_x``: string with the name of the inner variable (will be the sequence `q^n`)
        * ``name_qshift``: string with the name of the shift operator for the algebra (will be the natural shift `n \mapsto n+1`)
        * ``rational``: boolean (``True`` by default) deciding whether the base ring is 
          the field of rational functions or a polynomial ring.
        * ``base``: the base field for the polynomial ring w.r.t. ``name_x``.

        OUTPUT:

        A tuple `(A, (x, Q))` where `A` is the corresponding `q`-recurrence algebra, `x` is the 
        variable of the inner variable and `Q` the recurrence operator with `Q(x) = qx`.
    '''
    if not (name_x, name_q, name_qshift, rational, base) in __CACHE_QSHIFT_ALGEBRA:
        with_q, q = has_variable(base, name_q)
        if not with_q: raise TypeError(f"The base ring [{base}] must have the variable {name_q}")
        PR, x = get_rational_algebra(name_x, base) if rational else get_polynomial_algebra(name_x, base)
        OE = OreAlgebra(PR, (name_qshift, lambda p : p(**{str(x) : q*x}), lambda _ : 0)); Q = OE.gens()[0]
        __CACHE_QSHIFT_ALGEBRA[(name_x, name_q, name_qshift, rational, base)] = (OE, (x,Q)) 
    
    return __CACHE_QSHIFT_ALGEBRA[(name_x, name_q, name_qshift, rational, base)]

__CACHE_DQSHIFT_ALGEBRA = {}
def get_double_qshift_algebra(name_x : str = "x", name_q = "q", name_qshift : str = "E", rational : bool = True, base : _Fields.parent_class = QQ) -> Tuple[OreAlgebra_generic, Tuple[Any, Any]]:
    r'''
        Method to get always the same ore algebra

        This method unifies the access for OreAlgebra to get the recurrence `q`-shift operator
        `x \mapsto qx` and its inverse in an :class:`OreAlgebra_generic`. The method allows to provide the names for the inner variable `x` 
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
    if not (name_x, name_q, name_qshift, rational, base) in __CACHE_DQSHIFT_ALGEBRA:
        with_q, q = has_variable(base, name_q)
        if not with_q: raise TypeError(f"The base ring [{base}] must have the variable {name_q}")
        PR, x = get_rational_algebra(name_x, base) if rational else get_polynomial_algebra(name_x, base)
        OE = OreAlgebra(PR, (name_qshift, lambda p : p(**{str(x) : q*x}), lambda _ : 0), (name_qshift+"i", lambda p : p(**{str(x) : (1/q)*x}), lambda _ : 0)); E, Ei = OE.gens()
        __CACHE_DQSHIFT_ALGEBRA[(name_x, name_q, name_qshift, rational, base)] = (OE, (x,E,Ei)) 
    
    return __CACHE_DQSHIFT_ALGEBRA[(name_x, name_q, name_qshift, rational, base)]
#############################################################################################
###
### METHODS INVOLVING ORE ALGEBRAS AND OTHER STRUCTURES
###
#############################################################################################
def apply_operator_to_seq(operator : OreOperator, sequence : Sequence, **kwds) -> Sequence:
    r'''
        Method to apply an operator to a sequence.
        
        This method will be similar to the usual __call__ method from ``ore_algebra``, but with sequences that
        are given as functions.
        
        INPUT:
        
        * ``operator``: and operator with 1 generator (the shift) over the polynomial ring `\mathbb{R}[x]`.
        * ``seq``: a sequence in functional format. This means that we can call it with integer values and we 
          obtain the values of the sequence at each point.
        * ``kwds``: optional named arguments. If "q_name" is given, we pass it to the q_operator checkers
          
        OUTPUT:
        
        A sequence in function format.
    '''
    q_name = kwds.get("q_name", "q")

    if is_recurrence_algebra(operator.parent()):
        v, S = gens_recurrence_algebra(operator.parent())
        coefficients = operator.coefficients(sparse=False)
        R = operator.parent().base() if v is None else operator.parent().base().base()
        gen = (lambda i : sum(coefficients[j]*sequence[i+j] for j in range(len(coefficients)))) if v is None else (lambda i : sum(coefficients[j](**{str(v): i})*sequence[i+j] for j in range(len(coefficients))))
    elif is_q_operator_algebra(operator.parent(), q_name):
        q, v, S = gens_q_operator_algebra(operator.parent(), q_name)
        coefficients = operator.coefficients(sparse=False)
        R = operator.parent().base() if v is None else operator.parent().base().base()
        gen = (lambda i : sum(coefficients[j]*sequence[i+j] for j in range(len(coefficients)))) if v is None else (lambda i : sum(coefficients[j](**{str(v): q**i})*sequence[i+j] for j in range(len(coefficients))))
    elif is_double_recurrence_algebra(operator.parent()):
        v, S, Si = gens_double_recurrence_algebra(operator.parent())
        monomials, coefficients = poly_decomposition(operator.polynomial())
        _eval_monomial = lambda m, n : sequence(n+m.degree(S)-m.degree(Si))
        _eval_coeff = (lambda c,_ : c) if v is None else (lambda c,n : c(**{str(v): n}))
        R = operator.parent().base() if v is None else operator.parent().base().base()        
        gen = lambda i : sum(_eval_monomial(monomials[j], i) * _eval_coeff(coefficients[j], i) for j in range(len(monomials)))
    elif is_double_q_operator_algebra(operator.parent(), q_name):
        q, v, S, Si = gens_double_q_operator_algebra(operator.parent(), q_name)
        monomials, coefficients = poly_decomposition(operator.polynomial())
        _eval_monomial = lambda m, n : sequence(n+m.degree(S)-m.degree(Si))
        _eval_coeff = (lambda c,_ : c) if v is None else (lambda c,n : c(**{str(v): q**n}))
        R = operator.parent().base() if v is None else operator.parent().base().base()        
        gen = lambda i : sum(_eval_monomial(monomials[j], i) * _eval_coeff(coefficients[j], i) for j in range(len(monomials)))
    else:
        raise TypeError(f"Type {operator.__class__} not valid for method 'apply_operator_to_seq'")

    return LambdaSequence(gen, R, 1, False)

def required_init(operator : OreOperator) -> int:
    r'''
        Method to compute the number of required initial values for a sequence.

        When given a recurrence operator, we need some initial conditions to completely define 
        a solution to the recurrence operator. This method computes the maximal index we need to compute
        in order to have a fully defined sequence.
    '''
    if is_recurrence_algebra(operator.parent()):
        _, S = gens_recurrence_algebra(operator.parent())
        dS = operator.polynomial().degree(S.polynomial())
        if is_based_field(operator.parent()):
            _, coeffs = poly_decomposition(operator.polynomial())
            to_check = lcm([el.denominator() for el in coeffs] + [operator.polynomial().coefficient(S.polynomial()**dS).numerator()])
        elif is_based_polynomial(operator.parent()):
            to_check = operator.polynomial().coefficient(S.polynomial()**dS)
        return max(-min([0]+[el[0]-1 for el in to_check.roots() if el[0] in ZZ]), dS)
    elif is_double_recurrence_algebra(operator.parent()):
        _, S, Si = gens_double_recurrence_algebra(operator.parent())
        dS = operator.polynomial().degree(S.polynomial()); dSi = operator.polynomial().degree(Si.polynomial())
        if is_based_field(operator.parent()):
            _, coeffs = poly_decomposition(operator.polynomial())
            to_check = lcm([el.denominator() for el in coeffs] + [operator.polynomial().coefficient(S.polynomial()**dS).numerator()])
        elif is_based_polynomial(operator.parent()):
            to_check = operator.polynomial().coefficient(S.polynomial()**dS)
        return max(-min([0]+[el[0]-1 for el in to_check.roots() if el[0] in ZZ]), dS+dSi)
    elif is_q_operator_algebra(operator.parent()):
        q, _, S = gens_recurrence_algebra(operator.parent())
        dS = operator.polynomial().degree(S.polynomial())
        if is_based_field(operator.parent()):
            _, coeffs = poly_decomposition(operator.polynomial())
            to_check = lcm([el.denominator() for el in coeffs] + [operator.polynomial().coefficient(S.polynomial()**dS).numerator()])
        elif is_based_polynomial(operator.parent()):
            to_check = operator.polynomial().coefficient(S.polynomial()**dS)

        try:
            roots_to_check = to_check.roots()
            invalid_indices = []
            for root in roots_to_check:
                if root.numerator() == 1: 
                    if len(root.denominator().coefficients()) == 1:
                        invalid_indices.append(-root.denominator().degree(q))
                elif root.denominator() == 1:
                    if len(root.numerator().coefficients()) == 1:
                        invalid_indices.append(root.numerator().degree(q))

            bound_found = max(0,max(invalid_indices,default=0))
        except:
            bound_found = 0
        
        return max(bound_found, dS)
    if is_double_q_operator_algebra(operator.parent()):
        q, _, S, Si = gens_recurrence_algebra(operator.parent())
        dS = operator.polynomial().degree(S.polynomial()); dSi = operator.polynomial().degree(Si.polynomial())
        if is_based_field(operator.parent()):
            _, coeffs = poly_decomposition(operator.polynomial())
            to_check = lcm([el.denominator() for el in coeffs] + [operator.polynomial().coefficient(S.polynomial()**dS).numerator()])
        elif is_based_polynomial(operator.parent()):
            to_check = operator.polynomial().coefficient(S.polynomial()**dS)

        try:
            roots_to_check = to_check.roots()
            invalid_indices = []
            for root in roots_to_check:
                if root.numerator() == 1: 
                    if len(root.denominator().coefficients()) == 1:
                        invalid_indices.append(-root.denominator().degree(q))
                elif root.denominator() == 1:
                    if len(root.numerator().coefficients()) == 1:
                        invalid_indices.append(root.numerator().degree(q))

            bound_found = max(0,max(invalid_indices,default=0))
        except:
            bound_found = 0
        
        return max(bound_found, dS+dSi)
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
    monomials,coefficients = poly_decomposition(operator.polynomial())
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
def solution(operator : OreOperator, init, check_init=True) -> Sequence:
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
    if is_recurrence_algebra(operator.parent()):
        v,S = gens_recurrence_algebra(operator.parent()); Si = 0
        dS = operator.polynomial().degree(S.polynomial()); dSi = 0
        _eval_coeff = None # TODO Add this
    elif is_double_recurrence_algebra(operator.parent()):
        v,S,Si = gens_double_recurrence_algebra(operator.parent())
        dS = operator.polynomial().degree(S.polynomial()); dSi = operator.polynomial().degree(Si.polynomial())
        _eval_coeff = None # TODO Add this
    elif is_q_operator_algebra(operator.parent()):
        q,v,S = gens_q_operator_algebra(operator.parent()); Si = 0
        dS = operator.polynomial().degree(S.polynomial()); dSi = 0
        _eval_coeff = None # TODO Add this
    elif is_double_q_operator_algebra(operator.parent()):
        q,v,S,Si = gens_double_q_operator_algebra(operator.parent())
        dS = operator.polynomial().degree(S.polynomial()); dSi = operator.polynomial().degree(Si.polynomial())
        _eval_coeff = None # TODO Add this
    else:
        raise TypeError(f"Type {operator.__class__} not valid for method 'solution'")

    universe = operator.parent().base() if v is None else operator.parent().base().base()
    required = required_init(operator)
    if len(init) < required:
        raise ValueError(f"More data ({required}) is needed")
    from_init = required if check_init else len(init)

    monomials, coefficients = poly_decomposition(operator.polynomial())
    lc = None # TODO: Compute the leading coefficient and remove it from the list above

    @cache
    def __aux_sol(n):
        def _eval_monomial(m, n):
            if Si != None and m.degree(Si.polynomial()) > 0:
                return __aux_sol(n-dS-m.degree(Si))
            elif m.degree(S.polynomial()) > 0:
                return __aux_sol(n-dS+m.degree(S))
            else:
                return __aux_sol(n-dS)
        if n < 0: 
            return 0
        elif n < from_init:
            return init[n]
        else:
            return -sum(_eval_coeff(coefficients[i], n-dS)*_eval_monomial(monomials[i], n) for i in range(-dSi, dS))/_eval_coeff(lc, n-dS)
    return LambdaSequence(__aux_sol, universe)

class OreSequence(Sequence):
    r'''
        Class to represent a sequence defined by a linear operator. This will include
        C-finite, D-finite, Q-finite and similar type of sequences.

        These sequences are always defined with a linear operator acting (somehow) on the 
        sequences and some initial conditions. 

        TODO: Implement or use other class for this idea
    '''
    pass

####################################################################################################
###
### AUXILIARY METHODS
###
####################################################################################################
def poly_decomposition(polynomial):
    r'''
        Method that splits a polynomial into a lists of monomials and a list of coefficients indexed in 
        such a way that summing through both lists gives the original polynomial.
        
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
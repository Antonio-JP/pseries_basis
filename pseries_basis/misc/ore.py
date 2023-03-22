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
from __future__ import annotations

try: # python 3.9 or higher
    from functools import cache
except ImportError: #python 3.8 or lower
    from functools import lru_cache as cache
from typing import Collection, TypeVar

from ore_algebra.ore_algebra import OreAlgebra, OreAlgebra_generic
from ore_algebra.ore_operator import OreOperator

from sage.all import QQ, ZZ, prod, lcm, cached_method, Parent
from sage.categories.fields import Fields
from sage.categories.pushout import pushout
from sage.rings.polynomial.polynomial_ring import PolynomialRing_field, is_PolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.fraction_field import FractionField_1poly_field
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing
from sage.rings.ring import Algebra # pylint: disable=no-name-in-module
from sage.structure import element

from .sequences import Sequence, LambdaSequence

_Fields = Fields.__classcall__(Fields)
Element = element.Element

#############################################################################################
###
### METHODS TO CATEGORIZE ALGEBRAS OF OPERATORS
###
#############################################################################################
def is_recurrence_algebra(algebra: OreAlgebra_generic) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` has the first operator a shift operator.
        See :func:`gens_recurrence_algebra` to see what is a valid shift operator.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: from ore_algebra import OreAlgebra
            sage: is_recurrence_algebra(OreAlgebra(QQ[x], ('a', lambda p : p^2, lambda p : 0)))
            False
            sage: is_recurrence_algebra(OreAlgebra(QQ[x], ('a', lambda p : p(x = x+1), lambda p : 0)))
            True
            sage: is_recurrence_algebra(OreAlgebra(QQ[x], ('a', lambda p : p(x = x+10), lambda p : 0)))
            True
            sage: R.<x,y> = QQ[]
            sage: is_recurrence_algebra(OreAlgebra(R, ('a', lambda p : p(x = y*x), lambda p : 0)))
            False
            sage: is_recurrence_algebra(OreAlgebra(QQ[x], ('a', lambda p : p, lambda p : p.derivative())))
            False
            sage: is_recurrence_algebra(OreAlgebra(QQ[x], ('a', lambda p : p, lambda p : 0)))
            True
    '''
    return gens_recurrence_algebra(algebra) != None

@cache
def gens_recurrence_algebra(algebra: OreAlgebra_generic) -> tuple[Element, Element, Element]:
    r'''
        Method that returns the information about a recurrence ore algebra.

        A recurrence ore algebra is defined by a :class:`~ore_algebra.ore_algebra.OreAlgebra_generic`
        with exactly 1 generator `S` that acts over a ring where (at most) one generator `v` is affected like `S(v) = v + \alpha`. The 
        other generators must commute with `S` (i.e., `S(x) = x`).

        This method returns the triplet `(v, S, \alpha)`.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: from ore_algebra import OreAlgebra
            sage: gens_recurrence_algebra(OreAlgebra(QQ[x], ('a', lambda p : p^2, lambda p : 0)))
            sage: gens_recurrence_algebra(OreAlgebra(QQ[x], ('a', lambda p : p(x = x+1), lambda p : 0)))
            (x, a, 1)
            sage: R.<t> = QQ[]
            sage: gens_recurrence_algebra(OreAlgebra(R, ('b', lambda p : p(t = t+10), lambda p : 0)))
            (t, b, 10)
            sage: R.<x,y> = QQ[]
            sage: gens_recurrence_algebra(OreAlgebra(R, ('a', lambda p : p(x = y*x), lambda p : 0)))
            sage: gens_recurrence_algebra(OreAlgebra(QQ[x], ('a', lambda p : p, lambda p : p.derivative())))
        
        We allow an operators that commutes with all the variables. In that case, the `\alpha` return is ``None``
        since we can not know which variable is the shift referring to::

            sage: gens_recurrence_algebra(OreAlgebra(QQ[x], ('a', lambda p : p, lambda p : 0)))
            (None, a, None)
    '''
    if isinstance(algebra, OreAlgebra_generic) and algebra.ngens() == 1:
        S = algebra.gens()[0]
        found = None
        for v in algebra.base().gens():
            applied_S = S*v
            if len(applied_S.coefficients()) > 1: # some weird behavior for one generator
                return None
            elif applied_S.coefficients() != [v]: # S does not commute with v
                if found: # more than one generator is not commuting
                    return None
                diff = applied_S.coefficients()[0] - v
                if not diff in algebra.base().base_ring(): # it is not a shift
                    return None
                else:
                    found = (v, diff)
        return (None, S, None) if found is None else (found[0], S, found[1])
    return None

def is_double_recurrence_algebra(algebra: OreAlgebra_generic) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` is a double recurrence algebra.
        See :func:`gens_double_recurrence_algebra` to see what is a valid shift operator.
        
        TODO: Add examples using those in :func:`gens_double_recurrence_algebra`
    '''
    return gens_double_recurrence_algebra(algebra) != None

@cache
def gens_double_recurrence_algebra(algebra: OreAlgebra_generic) -> tuple[Element, Element, Element, Element]:
    r'''
        Method that returns the information about a double recurrence ore algebra.

        A double recurrence ore algebra is defined by a :class:`~ore_algebra.ore_algebra.OreAlgebra_generic`
        with exactly 2 generator `S` and `S_i` that acts over a ring where (at most) one generator `v` is affected like `S(v) = v + \alpha`
        and `S_i(v) = v - \alpha`. The other generators must commute with `S` and `S_i` (i.e., `S(x) = x`), which implies that `S S_i = S_i S = 1`.

        This method returns the tuple `(v, S, S_i, \alpha)`.

        EXAMPLES::

        TODO -- add examples
    '''
    if isinstance(algebra, OreAlgebra_generic) and algebra.ngens() == 2:
        S, Si = algebra.gens()
        found = None
        for v in algebra.base().gens():
            applied_S = S*v
            if len(applied_S.coefficients()) > 1: # some weird behavior for one generator
                return None
            elif applied_S.coefficients() != [v]: # S does not commute with v
                if found: # more than one generator is not commuting
                    return None
                diff = applied_S.coefficients()[0] - v
                if not diff in algebra.base().base_ring(): # it is not a shift
                    return None
                else:
                    found = (v, diff)
            applied_SSi = (S*Si*v, Si*S*v)
            # both double applications must be the identity
            if any(len(el.coefficients()) != 1 or el.coefficients()[0] != v for el in applied_SSi):
                return None

        return (None, S, Si, None) if found is None else (found[0], S, Si, found[1])

    return None

def is_differential_algebra(algebra: OreAlgebra_generic) -> bool:
    r'''
        Method to check whether an :class:`OreAlgebra_generic` has the first operator a differential operator.
    '''
    return isinstance(algebra, OreAlgebra_generic) and algebra.is_D()

def __get_power_q(element, q):
    r'''Check whether element is a power of q'''
    element = q.parent()(element)

    if element.parent().is_field(): # fraction field
        if not element.denominator() == 1:
            return -1
        element = element.numerator()
    if not q in element.parent().gens():
        return -1
    if len(element.coefficients()) > 1 or element.coefficients()[0] != 1:
        return -1
    
    return element.degree() if is_PolynomialRing(element.parent()) else element.degree(q)
         

def is_qshift_algebra(algebra : OreAlgebra_generic, name_q : str = None) -> bool:
    r'''
        Method to check whether an algebra behaves like one with Q-shifts operators.
        See :func:`gens_qshift_algebra` for further information on the definition of a `q`-shift operator
        
        TODO: Add examples using those in :func:`gens_qshift_algebra`
    '''
    return gens_qshift_algebra(algebra, name_q) != None

@cache
def gens_qshift_algebra(algebra: OreAlgebra_generic, name_q : str = None) -> tuple[Element,Element,Element]:
    r'''
        Method that returns the information about a `q`-shift recurrence ore algebra.

        A `q`-shift recurrence ore algebra is defined by a :class:`~ore_algebra.ore_algebra.OreAlgebra_generic`
        with exactly 1 generator `S` that acts over a ring where (at most) one generator `v` is affected like `S(v) = qv`. 
        The other generators must commute with `S` (i.e., `S(x) = x`).

        This method returns the tuple `(v, S, q)`.

        In the case ``name_q`` is given, we require that the value for `q` found in the previous description must be ``name_q``.

        EXAMPLES::

        TODO -- add examples
    '''
    q = None
    if name_q != None:
        with_q, q = has_variable(algebra.base(), name_q)
        if not with_q:
            return None

    if isinstance(algebra, OreAlgebra_generic) and algebra.ngens() == 1:
        S = algebra.gens()[0]
        found = None
        for v in algebra.base().gens():
            applied_S = S*v
            if len(applied_S.coefficients()) > 1: # some weird behavior for one generator
                return None
            elif applied_S.coefficients() != [v]: # S does not commute with v
                if found: # more than one generator is not commuting
                    return None
                if applied_S.coefficients()[0] % v != 0: # it is not a multiple of v
                    return None
                quot = applied_S.coefficients()[0]//v
                power = -1 if q is None else __get_power_q(quot, q)
                if q != None and power < 0: # the required `q` is not the one obtained
                    return None
                elif not quot in algebra.base().base_ring(): # it is not a `q`-shift
                    return None
                else:
                    found = (v, q if q != None else quot, power if q != None else 1)
        return (None, S, q, 1) if found is None else (found[0], S, found[1], found[2])
    return None

def is_double_qshift_algebra(algebra : OreAlgebra_generic, name_q : str = None):
    r'''
        Method to check whether an algebra behaves like a double Q-shifts operators.
        See :func:`gens_double_qshift_algebra` for further information on the definition of a double `q`-shift operator

        TODO: Add examples using those in :func:`gens_double_qshift_algebra`
    '''
    return gens_double_qshift_algebra(algebra, name_q) != None

@cache
def gens_double_qshift_algebra(algebra: OreAlgebra_generic, name_q : str = None) -> tuple[Element,Element,Element,Element]:
    r'''
        Method that returns the information about a double `q`-shift ore algebra.

        A double `q`-shift ore algebra is defined by a :class:`~ore_algebra.ore_algebra.OreAlgebra_generic`
        with exactly 2 generator `S` and `S_i` that acts over a ring where (at most) one generator `v` is affected like `S(v) = qv`
        and `S_i(v) = v/q`. The other generators must commute with `S` and `S_i` (i.e., `S(x) = x`), which implies that `S S_i = S_i S = 1`.

        This method returns the tuple `(v, S, S_i, q)`.

        In the case ``name_q`` is given, we require that the value for `q` found in the previous description must be ``name_q``.

        EXAMPLES::

        TODO -- add examples
    '''
    q = None
    if name_q != None:
        with_q, q = has_variable(algebra.base(), name_q)
        if not with_q:
            return None

    if isinstance(algebra, OreAlgebra_generic) and algebra.ngens() == 2:
        S, Si = algebra.gens()
        found = None
        for v in algebra.base().gens():
            applied_S = S*v
            if len(applied_S.coefficients()) > 1: # some weird behavior for one generator
                return None
            elif applied_S.coefficients() != [v]: # S does not commute with v
                coeff = applied_S.coefficients()[0]
                if is_based_field(algebra):
                    coeff = algebra.base().base()(coeff)

                if found: # more than one generator is not commuting
                    return None
                if coeff % v != 0: # it is not a multiple of v
                    return None
                quot = coeff/v
                power = -1 if q == None else __get_power_q(quot, q)
                if q != None and power < 0: # the required `q` is not the one obtained
                    return None
                elif not quot in algebra.base().base_ring(): # it is not a `q`-shift
                    return None
                else:
                    found = (v, q if q != None else algebra.base().base_ring()(quot), power if q != None else 1)
            applied_SSi = (S*Si*v, Si*S*v)
            # both double applications must be the identity
            if any(len(el.coefficients()) != 1 or el.coefficients()[0] != v for el in applied_SSi):
                return None

        return (None, S, Si, q, 1) if found is None else (found[0], S, Si, found[1], found[2])

    return None

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

def has_variable(algebra: Algebra, name: str) -> bool:
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
def get_polynomial_algebra(name_x: str = "x", base : _Fields.parent_class = QQ) -> tuple[PolynomialRing_field, Element]:
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

def get_rational_algebra(name_x: str = "x", base : _Fields.parent_class = QQ) -> tuple[FractionField_1poly_field, Element]:
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
def get_recurrence_algebra(name_x : str = "x", name_shift : str = "E", rational : bool = True, base : _Fields.parent_class = QQ) -> tuple[OreAlgebra_generic, tuple[Element, Element]]:
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
def get_double_recurrence_algebra(name_x : str = "x", name_shift : str = "E", rational : bool = True, base : _Fields.parent_class = QQ) -> tuple[OreAlgebra_generic, tuple[Element, Element, Element]]:
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
def get_differential_algebra(name_x : str = "x", name_der : str = "Dx", rational : bool = True, base : _Fields.parent_class = QQ) -> tuple[OreAlgebra_generic, tuple[Element, Element]]:
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
def get_qshift_algebra(name_x : str = "x", name_q = "q", name_qshift : str = "E", power: int = 1, rational : bool = True, base : _Fields.parent_class = QQ) -> tuple[OreAlgebra_generic, tuple[Element, Element]]:
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
    if not power in ZZ or power < 0:
        raise ValueError("The value for the power must be a non-negative integer")
    power = ZZ(power)
    if not (name_x, name_q, name_qshift, power, rational, base) in __CACHE_QSHIFT_ALGEBRA:
        with_q, q = has_variable(base, name_q)
        if not with_q: raise TypeError(f"The base ring [{base}] must have the variable {name_q}")
        PR, x = get_rational_algebra(name_x, base) if rational else get_polynomial_algebra(name_x, base)
        OE = OreAlgebra(PR, (name_qshift, lambda p : p(**{str(x) : (q**power)*x}), lambda _ : 0)); Q = OE.gens()[0]
        __CACHE_QSHIFT_ALGEBRA[(name_x, name_q, name_qshift, power, rational, base)] = (OE, (x,Q)) 
    
    return __CACHE_QSHIFT_ALGEBRA[(name_x, name_q, name_qshift, power, rational, base)]

__CACHE_DQSHIFT_ALGEBRA = {}
def get_double_qshift_algebra(name_x : str = "x", name_q = "q", name_qshift : str = "E", power: int = 1, rational : bool = True, base : _Fields.parent_class = QQ) -> tuple[OreAlgebra_generic, tuple[Element, Element]]:
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
    if not power in ZZ or power < 0:
        raise ValueError("The value for the power must be a non-negative integer")
    power = ZZ(power)
    if not (name_x, name_q, name_qshift, power, rational, base) in __CACHE_DQSHIFT_ALGEBRA:
        with_q, q = has_variable(base, name_q)
        if not with_q: raise TypeError(f"The base ring [{base}] must have the variable {name_q}")
        PR, x = get_rational_algebra(name_x, base) if rational else get_polynomial_algebra(name_x, base)
        OE = OreAlgebra(PR, (name_qshift, lambda p : p(**{str(x) : (q**power)*x}), lambda _ : 0), (name_qshift+"i", lambda p : p(**{str(x) : (1/(q**power))*x}), lambda _ : 0)); E, Ei = OE.gens()
        __CACHE_DQSHIFT_ALGEBRA[(name_x, name_q, name_qshift, power, rational, base)] = (OE, (x,E,Ei)) 
    
    return __CACHE_DQSHIFT_ALGEBRA[(name_x, name_q, name_qshift, power, rational, base)]
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
        * ``kwds``: optional named arguments. If "q_name" is given, we pass it to the qshift checkers
          
        OUTPUT:
        
        A sequence in function format.
    '''
    q_name = kwds.get("q_name", "q")

    if is_recurrence_algebra(operator.parent()):
        v, S, alpha = gens_recurrence_algebra(operator.parent())
        if not alpha in ZZ: 
            raise ValueError(f"The shift must be an integer shift (got {alpha})")
        coefficients = operator.coefficients(sparse=False)
        R = operator.parent().base() if v is None else operator.parent().base().base_ring()
        gen = (lambda i : sum(coefficients[j]*sequence[i+alpha*j] for j in range(len(coefficients)))) if v is None else (lambda i : sum(coefficients[j](**{str(v): i})*sequence[i+alpha*j] for j in range(len(coefficients))))
    elif is_qshift_algebra(operator.parent(), q_name):
        v, S, q, p = gens_qshift_algebra(operator.parent(), q_name)
        if q is None:
            raise ValueError(f"The `q`-shift must be fully defined (got None)")
        coefficients = operator.coefficients(sparse=False)
        R = operator.parent().base() if v is None else operator.parent().base().base_ring()
        gen = (lambda i : sum(coefficients[j]*sequence[i+j] for j in range(len(coefficients)))) if v is None else (lambda i : sum(coefficients[j](**{str(v): q**(p*i)})*sequence[i+j] for j in range(len(coefficients))))
    elif is_double_recurrence_algebra(operator.parent()):
        v, S, Si, alpha = gens_double_recurrence_algebra(operator.parent())
        if not alpha in ZZ: 
            raise ValueError(f"The shift must be an integer shift (got {alpha})")
        monomials, coefficients = poly_decomposition(operator.polynomial()); S = S.polynomial(); Si = Si.polynomial()
        _eval_monomial = lambda m, n : sequence(n+alpha*(m.degree(S)-m.degree(Si)))
        _eval_coeff = (lambda c,_ : c) if v is None else (lambda c,n : c(**{str(v): n}))
        R = operator.parent().base() if v is None else operator.parent().base().base_ring()        
        gen = lambda i : sum(_eval_monomial(monomials[j], i) * _eval_coeff(coefficients[j], i) for j in range(len(monomials)))
    elif is_double_qshift_algebra(operator.parent(), q_name):
        v, S, Si, q, p = gens_double_qshift_algebra(operator.parent(), q_name)
        if q is None:
            raise ValueError(f"The `q`-shift must be fully defined (got None)")
        monomials, coefficients = poly_decomposition(operator.polynomial()); S = S.polynomial(); Si = Si.polynomial()
        _eval_monomial = lambda m, n : sequence(n+m.degree(S)-m.degree(Si))
        _eval_coeff = (lambda c,_ : c) if v is None else (lambda c,n : c(**{str(v): q**(p*n)}))
        R = operator.parent().base() if v is None else operator.parent().base().base_ring()        
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

        TODO: add examples
    '''
    monomials, coeffs = poly_decomposition(operator.polynomial())
    if is_recurrence_algebra(operator.parent()):
        _, S, alpha = gens_recurrence_algebra(operator.parent()); S = S.polynomial()
        if (not alpha in ZZ) or alpha < 0: 
            raise ValueError(f"The shift must be a positive integer shift (got {alpha})")
        dS = operator.polynomial().degree(S)
        if is_based_field(operator.parent()):
            to_check = lcm([el.denominator() for el in coeffs] + [coeffs[monomials.index(S**dS)].numerator()])
        elif is_based_polynomial(operator.parent()):
            to_check = coeffs[monomials.index(S**dS)]
        output = max(-min([0]+[el[0]-1 for el in to_check.roots() if el[0] in ZZ]), alpha*dS)
    elif is_double_recurrence_algebra(operator.parent()):
        _, S, Si, alpha = gens_double_recurrence_algebra(operator.parent()); S = S.polynomial(); Si = Si.polynomial()
        if (not alpha in ZZ): 
            raise ValueError(f"The shift must be an integer shift (got {alpha})")
        elif alpha < 0: # we switch the shift and its inverse
            S, Si, alpha = Si, S, -alpha
        dS = operator.polynomial().degree(S); dSi = operator.polynomial().degree(Si)
        if is_based_field(operator.parent()):
            to_check = lcm([el.denominator() for el in coeffs] + [coeffs[monomials.index(S**dS)].numerator()])
        elif is_based_polynomial(operator.parent()):
            to_check = coeffs[monomials.index(S**dS)]
        output = max(-min([0]+[el[0]-1 for el in to_check.roots() if el[0] in ZZ]), alpha*(dS+dSi))
    elif is_qshift_algebra(operator.parent()):
        _, S, q, p = gens_qshift_algebra(operator.parent()); S = S.polynomial()
        if q == None:
            raise ValueError(f"The `q`-shift must be fully defined (got None)")
        dS = operator.polynomial().degree(S)
        if is_based_field(operator.parent()):
            to_check = lcm([el.denominator() for el in coeffs] + [coeffs[monomials.index(S**dS)].numerator()])
        elif is_based_polynomial(operator.parent()):
            to_check = coeffs[monomials.index(S**dS)]

        try:
            roots_to_check = to_check.roots()
            invalid_indices = []
            for root in roots_to_check:
                if root.numerator() == 1: 
                    if len(root.denominator().coefficients()) == 1:
                        invalid_indices.append(-p*root.denominator().degree(q))
                elif root.denominator() == 1:
                    if len(root.numerator().coefficients()) == 1:
                        invalid_indices.append(p*root.numerator().degree(q))

            bound_found = max(0,max(invalid_indices,default=0))
        except:
            bound_found = 0
        
        output = max(bound_found, dS)
    elif is_double_qshift_algebra(operator.parent()):
        _, S, Si, q, p = gens_double_qshift_algebra(operator.parent()); S = S.polynomial(); Si = Si.polynomial()
        dS = operator.polynomial().degree(S); dSi = operator.polynomial().degree(Si)
        if q == None:
            raise ValueError(f"The `q`-shift must be fully defined (got None)")
        if is_based_field(operator.parent()):
            to_check = lcm([el.denominator() for el in coeffs] + [coeffs[monomials.index(S**dS)].numerator()])
        elif is_based_polynomial(operator.parent()):
            to_check = coeffs[monomials.index(S**dS)]

        try:
            roots_to_check = to_check.roots()
            invalid_indices = []
            for root in roots_to_check:
                if root.numerator() == 1: 
                    if len(root.denominator().coefficients()) == 1:
                        invalid_indices.append(-p*root.denominator().degree(q))
                elif root.denominator() == 1:
                    if len(root.numerator().coefficients()) == 1:
                        invalid_indices.append(p*root.numerator().degree(q))

            bound_found = max(0,max(invalid_indices,default=0))
        except:
            bound_found = 0
        
        output = max(bound_found, dS+dSi)
    else:
        raise TypeError(f"Type {operator.__class__} not valid for method 'required_init'")

    return int(output)

def eval_ore_operator(operator : OreOperator, ring: Parent = None, **values: Element) -> Element:
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
def solution(operator: OreOperator, init: Collection[Element], check_init=True) -> Sequence:
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

        TODO: add examples
    '''
    if is_recurrence_algebra(operator.parent()):
        v,S,alpha = gens_recurrence_algebra(operator.parent()); S = S.polynomial(); Si = None
        if (not alpha in ZZ): 
            raise ValueError(f"The shift must be an integer shift (got {alpha})")
        alpha = ZZ(alpha)

        dS = operator.polynomial().degree(S); dSi = 0
        _eval_coeff = (lambda c,_ : c) if v is None else (lambda c,n : c(**{str(v) : n}))
        _shift = lambda i : alpha*i
    elif is_double_recurrence_algebra(operator.parent()):
        v,S,Si,alpha = gens_double_recurrence_algebra(operator.parent()); S = S.polynomial(); Si = Si.polynomial()
        if (not alpha in ZZ): 
            raise ValueError(f"The shift must be an integer shift (got {alpha})")
        alpha = ZZ(alpha)

        dS = operator.polynomial().degree(S); dSi = operator.polynomial().degree(Si)
        _eval_coeff = (lambda c,_ : c) if v is None else (lambda c,n : c(**{str(v) : n}))
        _shift = lambda i : alpha*i
    elif is_qshift_algebra(operator.parent()):
        v,S,q,p = gens_qshift_algebra(operator.parent()); S = S.polynomial(); Si = None
        if q == None:
            raise ValueError(f"The `q`-shift must be fully defined (got None)")

        dS = operator.polynomial().degree(S); dSi = 0
        _eval_coeff = (lambda c,_ : c) if v is None else (lambda c,n : c(**{str(v) : q**(p*n)}))
        _shift = lambda i : i
    elif is_double_qshift_algebra(operator.parent()):
        v,S,Si,q,p = gens_double_qshift_algebra(operator.parent()); S = S.polynomial(); Si = Si.polynomial()
        if q == None:
            raise ValueError(f"The `q`-shift must be fully defined (got None)")

        dS = operator.polynomial().degree(S); dSi = operator.polynomial().degree(Si)
        _eval_coeff = (lambda c,_ : c) if v is None else (lambda c,n : c(**{str(v) : q**(p*n)}))
        _shift = lambda i : i
    else:
        raise TypeError(f"Type {operator.__class__} not valid for method 'solution'")

    universe = operator.parent().base() if v is None else operator.parent().base().base_ring()
    required = required_init(operator)
    if len(init) < required:
        raise ValueError(f"More data ({required}) is needed")
    from_init = required if check_init else len(init)

    monomials, coefficients = poly_decomposition(operator.polynomial())
    lc_index = monomials.index(S**dS)
    monomials.pop(lc_index); lc = coefficients.pop(lc_index)

    @cache
    def __aux_sol(n):
        def _eval_monomial(m, n):
            if Si != None and m.degree(Si) > 0:
                return __aux_sol(n+_shift(-dS-m.degree(Si)))
            elif m.degree(S) > 0:
                return __aux_sol(n+_shift(-dS+m.degree(S)))
            else:
                return __aux_sol(n+_shift(-dS))
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
    def __init__(self, operator : OreOperator, init: Collection[Element], universe: Parent = None):
        self.__sequence = solution(operator, init, True)
        self.__operator = operator

        universe = self.__sequence.universe if universe is None else pushout(universe, self.__sequence.universe)

        super().__init__(universe, 1, False)

    @property
    def operator(self) -> OreOperator: return self.__operator

    @cached_method
    def required_init(self) -> int: 
        return required_init(self.operator)

    @property
    def type(self) -> str:
        if is_recurrence_algebra(self.operator.parent()):
            return "recurrence"
        elif is_double_recurrence_algebra(self.operator.parent()):
            return "double_recurrence"
        elif is_qshift_algebra(self.operator.parent()):
            return "qshift"
        elif is_double_qshift_algebra(self.operator.parent()):
            return "double_qshift"
        else:
            return "none"

    @cached_method
    def op_gen(self) -> OreOperator:
        r'''
            Method that returns the main operator of the Ore Algebra associated to this sequence
        '''
        if is_recurrence_algebra(self.operator.parent()):
            method = gens_recurrence_algebra
        elif is_double_recurrence_algebra(self.operator.parent()):
            method = gens_double_recurrence_algebra
        elif is_qshift_algebra(self.operator.parent()):
            method = gens_qshift_algebra
        elif is_double_qshift_algebra(self.operator.parent()):
            method = gens_double_qshift_algebra
        else:
            raise TypeError(f"Type of operator [{self.operator.parent()}] not valid")

        return method(self.operator.parent())[1]

    @cached_method
    def op_gen_inv(self) -> OreOperator:
        r'''
            Method that returns the main operator of the Ore Algebra associated to this sequence
        '''
        if is_recurrence_algebra(self.operator.parent()):
            return None
        elif is_double_recurrence_algebra(self.operator.parent()):
            method = gens_double_recurrence_algebra
        elif is_qshift_algebra(self.operator.parent()):
            return None
        elif is_double_qshift_algebra(self.operator.parent()):
            method = gens_double_qshift_algebra
        else:
            raise TypeError(f"Type of operator [{self.operator.parent()}] not valid")

        return method(self.operator.parent())[2]

    def _element(self, *indices: int) -> Element:
        return self.__sequence._element(*indices)

    def _shift(self) -> OreSequence:
        if self.type.find("double") >= 0:
            # since we can have the inverse shift, we multiply by it
            Si = self.op_gen_inv()
            new_operator = self.operator*Si
        elif self.type != "none":
            # this is the usual, we can use ore_algebra functions
            S = self.op_gen()
            new_operator = self.operator.annihilator_of_associate(S)
        
        new_init = [self(i+1) for i in range(required_init(new_operator))]
        return OreSequence(new_operator, new_init, self.universe)

    def __add__(self, other) -> Sequence:
        if self.type.find("double") < 0 and self.type != "none" and isinstance(other, OreSequence) and self.operator.parent() == other.operator.parent():
            # This is the only case we can use the methods from Ore Algebra
            new_operator = self.operator.lclm(other.operator)
            new_init = [self(i) + other(i) for i in range(required_init(new_operator))]

            return OreSequence(new_operator, new_init, pushout(self.universe, other.universe))
        return super().__add__(other)

    def __sub__(self, other) -> Sequence:
        if self.type.find("double") < 0 and self.type != "none" and isinstance(other, OreSequence) and self.operator.parent() == other.operator.parent():
            # This is the only case we can use the methods from Ore Algebra
            new_operator = self.operator.lclm(other.operator)
            new_init = [self(i) - other(i) for i in range(required_init(new_operator))]

            return OreSequence(new_operator, new_init, pushout(self.universe, other.universe))
        return super().__sub__(other)

    def __mul__(self, other) -> Sequence:
        if self.type.find("double") < 0 and self.type != "none" and isinstance(other, OreSequence) and self.operator.parent() == other.operator.parent():
            # This is the only case we can use the methods from Ore Algebra
            new_operator = self.operator.symmetric_product(other.operator)
            new_init = [self(i) * other(i) for i in range(required_init(new_operator))]

            return OreSequence(new_operator, new_init, pushout(self.universe, other.universe))
        return super().__mul__(other)
        
    def __neg__(self) -> OreSequence:
        return OreSequence(self.operator, [(-1)*self(i) for i in range(required_init(self.operator))], self.universe)

    def __repr__(self) -> str:
        return f"Sequence over [{self.universe}] defined by the {self.type} ({self.operator}) with initial values {self[:self.required_init()]}."
        
####################################################################################################
###
### AUXILIARY METHODS
###
####################################################################################################
def poly_decomposition(polynomial : Element) -> tuple[list[Element],list[Element]]:
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

__all__ = [
    "is_recurrence_algebra", 
    "gens_recurrence_algebra",
    "is_double_recurrence_algebra",
    "gens_double_recurrence_algebra",
    "is_differential_algebra",
    "is_qshift_algebra",
    "gens_qshift_algebra",
    "is_double_qshift_algebra",
    "gens_double_qshift_algebra",
    "get_recurrence_algebra",
    "get_double_recurrence_algebra",
    "get_differential_algebra",
    "get_qshift_algebra",
    "get_double_qshift_algebra",
    "apply_operator_to_seq",
    "required_init",
    "solution", 
    "OreSequence"
]
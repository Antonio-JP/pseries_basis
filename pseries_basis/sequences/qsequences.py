r'''
    Module to define q-sequences

    A q-sequence is a sequence whose value is define for `q^n` instead of simply `n`. This, in particular,
    has a main implication on how things are computed. More precisely, we need an element in the universe 
    that will act as `q`.

    Usually, `q` is a transcendental variable over the original universe, however, we may allow any element 
    in the universe of a sequence to take the role of `q`.

    Since the behavior of these sequences are quite different to the standard sequences, the relation 
    in terms of class casting (see :class:`.base.Sequence`) is reset for q-sequences, meaning that 
    besides the basic relation with base sequences and constant sequences, they are not related with any 
    other classical sequences. This imply that any operation will fall to the basic callable sequences.

    These classes inherit from the main class :class:`.base.Sequence`, hence they need to implement the 
    following method in order to interact properly with other sequences:

    * Override the class method :func:`register_class` with the desired classes we want to be directly below.
    * Implement the following methods:

        - :func:`_change_class`: receives a class (given when registering the sequence class) and cast the current sequence to the new class.
        - :func:`_neg_`: implement the negation of a sequence for a given class.
        - :func:`_final_add`: implement the addition for two sequences of the same parent and class.
        - :func:`_final_sub`: implement the difference for two sequences of the same parent and class.
        - :func:`_final_mul`: implement the hadamard product for two sequences of the same parent and class.
        - :func:`_final_div`: implement the hadamard division for two sequences of the same parent and class.
        - :func:`_final_mod`: implement the hadamard module for two sequences of the same parent and class.
        - :func:`_final_floordiv`: implement the hadamard floor division for two sequences of the same parent and class.
'''
from __future__ import annotations

import logging

from collections.abc import Callable
from typing import Any

from sage.structure.element import Element
from sage.structure.parent import Parent
from sage.symbolic.expression import Expression
from sage.symbolic.ring import SR

from .base import Sequence
from .element import ExpressionSequence, RationalSequence

logger = logging.getLogger(__name__)

def is_QSequence(element: Sequence) -> bool:
    r'''
        Method to check whether a given sequence is a `q`-sequence or not.

        ::NO EXAMPLE::
    '''
    return isinstance(element, Sequence) and hasattr(element, "q")

def QSequence(sequence: Callable[..., Any], universe=None, dim: int = 1, *, q, _extend_by_zero=False, **kwds: Any) -> Sequence:
    r'''
        Generic method to create a `q`-sequence.

        This method creates a :class:`~pseries_basis.element.Sequence` from a callable object (usually a function) that get as many
        inputs as ``dim`` and returns elements in the ``universe``.

        The main difference with the standard :class:`~pseries_basis.element.Sequence` is that there must be a special element `q`
        that must be kept as extra information for the sequence.

        INPUT:

        * ``sequence``: a callable object (usually a function) that takes as many inputs as ``dim`` and returns elements in the ``universe``.
        * ``universe``: the universe of the sequence. This is usually a field.
        * ``dim``: an integer indicating the dimension of the sequence. The default is 1, which means that the sequence will be a univariate sequence. If ``dim`` is 2, then the sequence will be a bivariate sequence, and so on.
        * ``q``: an element in the universe that will act as `q` in the sequence. This is usually a transcendental element.
        * ``_extend_by_zero``: a boolean that indicates whether the sequence should be extended by zeros when undefined.

        OUTPUT:

        A :class:`~pseries_basis.element.Sequence` object that represents the `q`-sequence.

        EXAMPLES::

            sage: from pseries_basis.sequences.qsequences import QSequence
            sage: B.<q> = QQ[].fraction_field()
            sage: QSequence(lambda n: q^(2*n) - q, B, 1, q=q)[:5]
            [-q + 1, q^2 - q, q^4 - q, q^6 - q, q^8 - q]
            sage: bi_seq = QSequence(lambda n, m: q^(n*m), B, 2, q=q)
            sage: bi_seq[:5,:5]
            [1    1    1    1      1]
            [1    q  q^2   q^3   q^4]
            [1  q^2  q^4   q^6   q^8]
            [1  q^3  q^6   q^9  q^12]
            [1  q^4  q^8  q^12  q^16]
    '''
    output = Sequence(sequence, universe, dim, _extend_by_zero=_extend_by_zero, q=q, **kwds)
    if q not in output.universe:
        raise TypeError("[QSequence] The element `q` must be in the universe.")
    return output

def QPower(power: int, universe: Parent, *, q: Element, **kwds: Any) -> Sequence:
    r'''
        Method to create the sequence `q^(power*n)` for a given `power` and `q`.

        INPUT:

        * ``power``: an integer indicating the power of `q` used for the sequence. The default is 1, which means that the sequence will be `q^n`. If `power` is 2, then the sequence will be `q^(2n)`, and so on.
        * ``universe``: the universe of the sequence. This is usually a field.
        * ``q``: an element in the universe that will act as `q` in the sequence. This is usually a transcendental element.

        OUTPUT:
        
        A :class:`~pseries_basis.element.Sequence` object that represents the sequence `q^(power*n)`.

        EXAMPLES::

            sage: from pseries_basis.sequences.qsequences import QPower
            sage: B.<q> = QQ[].fraction_field()
            sage: QPower(2, B, q=q)[:5]
            [1, q^2, q^4, q^6, q^8]
            sage: QPower(1, B, q=q)[:5]
            [1, q, q^2, q^3, q^4]

        This can be also used when `q` is not a transcedental element::

            sage: QPower(2, QQ, q=5)[:5]
            [1, 25, 625, 15625, 390625]
    '''
    return QSequence(lambda n : q**(power*n), universe, 1, q=q, power=power, **kwds)

def QExpressionSequence(
        expression: Expression, variables: tuple[Element] = None, universe: Parent = None, *, 
        q: Element, power: int = 1, _extend_by_zero: bool = False, **kwds: Any) -> ExpressionSequence:
    r'''
        Method to create a `q`-sequence from an expression. 

        As it happens with `q`-sequences, the main difference is that the variables now represents the sequence `q^n` where `n` is the 
        variable now. We also indicate the value of `q` in the universe, which is usually a transcendental element.

        INPUT:

        * ``expression``: a SageMath expression in the variables given by ``variables``. This is the main input to define the sequence.
        * ``variables``: a list of variables that will be used in the rational expression.
        * ``universe``: the universe of the sequence. This is usually a field.
        * ``q``: an element in the universe that will act as `q` in the sequence. This is usually a transcendental element.
        * ``power``: an integer that indicates the power of `q` used for the variables. The default is 1, which means that the variables will be `q^n`. If `power` is 2, then the variables will be `q^(2n)`, and so on.
        * ``_extend_by_zero``: a boolean that indicates whether the sequence should be extended by zeros.

        OUTPUT:

        A :class:`~pseries_basis.element.RationalSequence` object that represents the rational `q`-sequence.

        TODO (update_sage_10): add examples

        ::NO EXAMPLE::
    '''
    meanings = QPower(power, universe, q=q, **kwds)
    output = ExpressionSequence(expression, variables, universe, meanings=meanings, _extend_by_zero=_extend_by_zero, q=q, **kwds)
    if q not in output.universe:
        raise TypeError("[QSequence] The element `q` must be in the universe.")
    return output

def QRationalSequence(
        rational: Element, variables: tuple[Element] = None, universe: Parent = None, *, 
        q: Element, power: int = 1, _extend_by_zero: bool = False, **kwds: Any) -> RationalSequence:
    r'''
        Method to create a rational `q`-sequence. 

        A rational `q`-sequence is a rational sequence (i.e., a rational expression in a set of variables) as was the case 
        of the class :class:`~pseries_basis.element.RationalSequence`. 

        As it happens with `q`-sequences, the main difference is that the variables now represents the sequence `q^n` where `n` is the 
        variable now. We also indicate the value of `q` in the universe, which is usually a transcendental element.

        INPUT:

        * ``rational``: a rational expression in the variables given by ``variables``. This is the main input to define the sequence.
        * ``variables``: a list of variables that will be used in the rational expression.
        * ``universe``: the universe of the sequence. This is usually a field.
        * ``q``: an element in the universe that will act as `q` in the sequence. This is usually a transcendental element.
        * ``power``: an integer that indicates the power of `q` used for the variables. The default is 1, which means that the variables will be `q^n`. If `power` is 2, then the variables will be `q^(2n)`, and so on.
        * ``_extend_by_zero``: a boolean that indicates whether the sequence should be extended by zeros.

        OUTPUT:

        A :class:`~pseries_basis.element.RationalSequence` object that represents the rational `q`-sequence.

        TODO (update_sage_10): add examples

        ::NO EXAMPLE::
    '''
    meanings = QPower(power, universe, q=q, **kwds)
    output = RationalSequence(rational, variables, universe, meanings=meanings, _extend_by_zero=_extend_by_zero, q=q, **kwds)
    if q not in output.universe:
        raise TypeError("[QSequence] The element `q` must be in the universe.")
    return output
    

__all__ = ["QSequence", "QPower", "QExpressionSequence", "QRationalSequence", "is_QSequence"]
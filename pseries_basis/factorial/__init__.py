r'''
    Subpackage for Factorial basis. 
    
    A factorial basis is a basis of the ring of formal power series `\mathbb{K}[[x]]` where the 
    elements of the basis are polynomials of increasing degree such that:

    .. MATH::

        P_{n+1}(x) = (a_n*x + b_n)P_n(x)

    There are plenty of types and extensions of Factorial basis. This subpackage will include
    all of them, such as :class:`~pseries_basis.factorial.product_basis.ProductBasis` or 
    :class:`~pseries_basis.factorial.product_basis.SievedBasis`.
'''

from .factorial_basis import *
from .product_basis import *
from .gen_binomial_basis import *
r'''
    Sage package for Functional order Basis.
'''

# Local imports
from ..psbasis import PSBasis

class FunctionalBasis(PSBasis):
    r'''
        Class for representing a basis of power functions.

        A basis of power functions is a type of order basis for power series
        where the `n`-th element is the `n`-th power of an order 1 power series `f(x)`.

        The first element in the sequence will always be the constant polynomial 1.

        The second element in the sequence is a function `f(x)` such that `f(0) = 0`
        and `f'(0) \neq 0`

        INPUT:
            - ``var_name``: the name for the operator representing the multiplication by `f(x)`.
    '''
    def __init__(self, f, var_name='f'):
        raise NotImplementedError(f"Class FunctionalBasis not yet implemented.")
    
def ExponentialBasis(exponent, E='E', Dx='Dx') -> FunctionalBasis:
        raise NotImplementedError(f"Method ExponentialBasis not yet implemented.")

###############################################################
### EXAMPLES OF PARTICULAR ORDER BASIS
###############################################################
class BesselBasis(PSBasis):
    r'''
        Class for representing a basis of Bessel functions.

        The Bessel functions with integer index form naturally an order basis. They are usually denoted
        by `J_n(x)` where `n` is a natural number.

        INPUT:
            - ``Xi``: the name for the operator representing the multiplication by `1/x`.
            - ``Dx``: the name of the operator representing the standard derivation.
    '''
    def __init__(self, Xi='Xi', Dx='Dx'):
        ## Initializing the PolyBasis structure
        raise NotImplementedError("Class BesselBasis not yet implemented")

        ## The multiplication by X compatibility is given
        # Sni = self.Sni(); n = self.n(); Sn = self.Sn(); Q12 = 1/Integer(2)
        # self.set_compatibility(Xi, (Q12/n)*Sn + (Q12/n)*Sni)
        # self.set_derivation(Dx, Q12*Sn - Q12*Sni)

__all__ = ["FunctionalBasis", "ExponentialBasis", "BesselBasis"]
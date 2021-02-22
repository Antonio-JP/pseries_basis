r'''
    Sage package for Functional order Basis.
'''
# Sage imports
from sage.all import cached_method, Integer, bessel_J
from sage.all_cmdline import x

# Local imports
from .psbasis import OrderBasis

class FunctionalBasis(OrderBasis):
    r'''
        Class for representing a basis of power functions.

        A basis of power functions is a type of order basis for power series
        where the `n`th element is the `n`th power of an order 1 power series `f(x)`.

        The first element in the sequence will always be the constant polynomial 1.

        The second element in the sequence is a function `f(x)` such that `f(0) = 0`
        and `f'(0) \neq 0'

        INPUT:
            - ``X``: the name for the operator representing the multiplication by `f(x)`.
    '''
    def __init__(self, X='f'):
        ## Initializing the PolyBasis structure
        super(FunctionalBasis,self).__init__()

        ## Adding the extra information
        self.__fun_name = X

        ## The multiplication by X compatibility is given
        Sni = self.Sni()
        self.set_compatibility(X, Sni)

    @cached_method
    def element(self, n):
        R = self.polynomial_ring(self.__fun_name)
        f = R.gens()[0]
        return f**n

    def __repr__(self):
        return "Functional Power Basis (%s)" %(self.__fun_name)

    def _latex_(self):
        return r"\left\{%s(x)^n\right\}_{n \geq 0}" %(self.__fun_name)

###############################################################
### EXAMPLES OF PARTICULAR GENERALIZED POWER BASIS
###############################################################
class ExponentialBasis(FunctionalBasis):
    r'''
        Class for the power functional basis generated by the exponential.

        This class represents the FunctionalBasis formed by the set of powers
        of the exponential function `f(x) = exp(x)-1`. The minus one only shows up
        because `f(x)` must have order 0.

        Following the notation in :arxiv:`1804.02964v1`, we can find that
        this basis has compatibilities with the multiplication by `exp(x)` and the
        derivation operator

        INPUT:
            - ``E``: the name for the operator representing the multiplication by `exp(x)`.
            - ``Dx``: the name for the operator representing the derivation by `x`.
    '''
    def __init__(self, E='E', Dx='Dx'):
        super(ExponentialBasis, self).__init__()

        Sni = self.Sni(); n = self.n(); Sn = self.Sn()

        self.set_compatibility(E, Sni + 1)
        self.set_compatibility(Dx, n + (n+1)*Sn)



    def __repr__(self):
        return "Exponential basis (1, e^x-1, (e^x-1)^2,...)"

    def _latex_(self):
        return r"\left\{\left(e^x-1\right)^n\right\}_{n \geq 0}"

###############################################################
### EXAMPLES OF PARTICULAR ORDER BASIS
###############################################################
class BesselBasis(OrderBasis):
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
        super(BesselBasis,self).__init__()

        ## The multiplication by X compatibility is given
        Sni = self.Sni(); n = self.n(); Sn = self.Sn(); Q12 = 1/Integer(2)
        self.set_compatibility(Xi, (Q12/n)*Sn + (Q12/n)*Sni)
        self.set_compatibility(Dx, Q12*Sn - Q12*Sni)

    @cached_method
    def element(self, n):
        return bessel_J(n,x)

    def __repr__(self):
        return "Bessel Basis (J_n)"

    def _latex_(self):
        return r"\left\{J_n(x)\right\}_{n \geq 0}"
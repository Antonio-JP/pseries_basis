r'''
    Module with the basic classes for implementing `q`-series.
'''

from sage.all import prod, PolynomialRing, QQ, ZZ, var, binomial

from ..psbasis import SequenceBasis
from ..misc.ore import get_qshift_algebra, get_double_qshift_algebra, has_variable
from ..misc.sequences import LambdaSequence, Sequence

#######################################################################################################
### Classes for some basis of Q-series
#######################################################################################################
class QBasis(SequenceBasis):
    r'''
        Class for generic `q`-formal power series.

        The ring of `q`-formal power series can be seen as the ring `\mathbb{K}(q)[[x]]`, where 
        there is a parameter `q` whose field of rational functions is included in the ring of 
        coefficients for the formal pwoer series.
    '''
    def __init__(self, base, sequence: Sequence, degree: bool = True, q_name: str = None):
        # if the base has no q, we add it
        with_q, self.__q = has_variable(base, q_name)
        if not with_q:
            base = PolynomialRing(base, q_name).fraction_field()
            self.__q = base.gens()[0]
        super().__init__(base, sequence, degree)

    ### Getters from the module variable as objects of the class
    def OB(self):
        return self.base

    def n(self):
        return var('n')

    def q(self):
        r'''
            Method to get the generic variable `n` for the recurrences.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal building, do not use in general
                sage: B.n()
                n
                sage: B.n().parent()
                Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return self.__q

    def OS(self):
        r'''
            Method to get the generic variable :class:`~ore_algebra.OreAlgebra` with only the direct shift 
            over the rational functions in `n`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal building, do not use in general
                sage: B.OSS()
                Univariate Ore algebra in Sn over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return get_double_qshift_algebra("Sn", str(self.q()), "Q", base=self.base)[0]

    def OSS(self):
        r'''
            Method to get the generic variable :class:`~ore_algebra.OreAlgebra` with only the direct shift 
            over the rational functions in `n`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal building, do not use in general
                sage: B.OSS()
                Univariate Ore algebra in Sn over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return get_qshift_algebra("Sn", str(self.q()), "Q", base=self.base)[0]

    def Q(self):
        r'''
            Method to get the variable for the `q`-shift.

            This object is in the ring :func:`~PSBasis.OS`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal building, do not use in general
                sage: B.Sn()
                Sn
                sage: B.Sn().parent()
                Multivariate Ore algebra in Sn, Sni over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return get_double_qshift_algebra("Sn", str(self.q()), "Q", base=self.base)[1][1]

    def Sn(self):
        r'''
            Method to get the generic variable for the direct shift operator.

            This object is in the ring :func:`~PSBasis.OS`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal building, do not use in general
                sage: B.Sn()
                Sn
                sage: B.Sn().parent()
                Multivariate Ore algebra in Sn, Sni over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return get_double_qshift_algebra("Sn", str(self.q()), "Q", base=self.base)[1][2]

    def Sni(self):
        r'''
            Method to get the generic variable for the inverse shift operator.

            This object is in the ring :func:`~PSBasis.OS`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal building, do not use in general
                sage: B.Sni()
                Sni
                sage: B.Sni().parent()
                Multivariate Ore algebra in Sn, Sni over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return get_double_qshift_algebra("Sn", str(self.q()), "Q", base=self.base)[1][3]
    
    def QPower(self):
        r'''
            Method that obtains the corresponding sequence `1, q, q^2, \ldots` for the corresponding `q`.

            The multiplication by this sequence is the equivalent action related with the `Q` operator obtained 
            by `:func:`Q`.
        '''
        return LambdaSequence(lambda n : self.q()**n, self.base)

    def QFactorial(self):
        r'''
            Method that obtains the sequence of `q`-factorials.
        '''
        return LambdaSequence(lambda n : qfactorial(n, self.q()), self.base)

    def QPochammer(self, a):
        r'''
            Method that obtains the sequence of `q`-pochammer symbols from a fixed base value.
        '''
        return LambdaSequence(lambda n : qpochammer(a, n, self.q()), self.base)

    ## Other ovewriten methods
    def change_base(self, base):
        return QBasis(base, self.functional_seq, self.by_degree(), str(self.q()))

    ## TODO: Check methods of compatibility

    ## TODO: Check method scalar

    def shift_in(self, shift):
        return QBasis(self.base, self.functional_seq.shift(0, shift), self.by_degree, str(self.q()))

    ## Reperesentation methods
    def __repr__(self):
        return f"QBasis -- WARNING: this is an abstract class"

class QBinomialBasis(QBasis):
    r'''
        Class that represent a q-binomial basis.

        Mathematically, the `q`-binomial coefficient is defined as follows:

        .. MATH:: 

            \left[\begin{array}{c}n\\ m\end{array}\right]_q = \frac{[n]_q!}{[m]_q![n-m]_q!},

        where `[\cdot]_q!` represent the `q`-factorial. We recommend to see more information about the
        `q`-analogs on the Internet.

        For `m \in \mathbb{N}`, the sequences of `q`-binomial coefficients make a basis of `q`-series for the 
        more generic basis:

        .. MATH::

            \left[\begin{array}{c}an + bm + c\\ m\end{array}\right]_q

        This class represents precisely this basis and allow to compute compatible operators for some 
        default operators such as:

        * ``Sn``: the shift operator in `n`.
        * ``Q``: the multiplication by the sequence `(q^n)_n`.

        INPUT:

        * ``base``: (`\mathbb{Q}` by default) base ring for the `q`-basis. Changing this allow to add some parameters to the computations.
        * ``a``: (1 by default) value for the coefficient `a`.
        * ``b``: (0 by default) value for the coefficient `b`.
        * ``c``: (0 by default) value for the coefficient `c`.
        * ``q_name``: ("q" by default) the name for the `q` parameter.
    '''
    def __init__(self, base = QQ, a = 1, b = 0, c = 0, q_name = "q"):
        # checking the input
        if not a in ZZ or a == 0:
            raise ValueError("The value for the parameter 'a' must be a non-zero integer")
        if not b in ZZ:
            raise ValueError("The value for the parameter 'b' must be an integer.")
        if not c in ZZ or c < 0:
            raise ValueError("The value for the parameter 'c' must be an non-negative integer.")

        self.__a = ZZ(a)
        self.__b = ZZ(b)
        self.__c = ZZ(c)

        QBasis.__init__(self, base, LambdaSequence(lambda *n : 1, base, 2), False, q_name)
        seq = LambdaSequence(lambda m, n : qbinomial(a*n+b*m+c, m, self.q()), self.base, 2)
        super().__init__(base, seq, False, q_name)

        self.__recurrences = {}
        if a > 0:
            Q = self.Q(); Sn = self.Sn(); q = self.q()
            self.__recurrences['Ek'] = sum(binomial(a,i)*q**(a*i-i**2)*Q**(a-i)*Sn**(i) for i in range(a+1))
        if a == 1 and b <= 1:
            Q = self.Q(); Sni = self.Sni(); q = self.q()
            self.__recurrences['Q'] = q**(-c)*Q**(1-b) + q**(b-1-c)*Q**(1-b)*(Q - 1)*Sni

    @property
    def qrecurrences(self):
        return self.__recurrences

    def is_quasi_func_triangular(self):
        return self.__b == 0

    def __repr__(self):
        upper = [("" if self.__a == 1 else f"{self.__a}") + "n"]
        if self.__b != 0:
            upper.append(("" if self.__b == 1 else f"{self.__b}") + "m")
        if self.__c != 0:
            upper.append(f"{self.__c}")

        return f"Q-Binomial Basis [{' + '.join(upper)}; m]_q"

    def _latex_(self):
        upper = [("" if self.__a == 1 else f"{self.__a}") + "n"]
        if self.__b != 0:
            upper.append(("" if self.__b == 1 else f"{self.__b}") + "m")
        if self.__c != 0:
            upper.append(f"{self.__c}")
        return r"\left[\begin{array}{c} " + " + ".join(upper) + r"\\ m \end{array}\right]_q"

#######################################################################################################
### Some useful functions and sequences
#######################################################################################################
def qpochammer(a, n : int, q):
    '''Method for obtaining (a;q)_n'''
    return prod(1-a*q**k for k in range(n))

def qfactorial(n:int, q):
    '''Method for obtaining [n]_q! = (q;q)_n/(1-q)^n'''
    return qpochammer(q, n, q)/(1-q)**n

def qbinomial(n:int, m:int, q):
    '''Method for obtaining [n,m]_q = [n]_q!/([m]_q![n-m]_q!)'''
    if m < 0 or m > n:
        return 0
    return qfactorial(n,q)/(qfactorial(m, q)*qfactorial(n-m, q))

__all__ = ["QBasis", "QBinomialBasis"]
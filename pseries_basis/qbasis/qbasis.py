r'''
    Module with the basic classes for implementing `q`-series.
'''

from sage.all import prod, PolynomialRing, QQ, ZZ, cached_method, var, SR

from ..psbasis import PSBasis, PolyBasis, SequenceBasis
from ..factorial.factorial_basis import FactorialBasis, RootSequenceBasis, SFactorialBasis, ScalarBasis
from ..misc.ore import get_qshift_algebra, get_double_qshift_algebra, get_rational_algebra, has_variable
from ..misc.sequences import ExpressionSequence, LambdaSequence, Sequence

#######################################################################################################
### Classes for some basis of Q-series
#######################################################################################################
class QBasis(PSBasis):
    r'''
        Class for generic `q`-formal power series.

        The ring of `q`-formal power series can be seen as the ring `\mathbb{K}(q)[[x]]`, where 
        there is a parameter `q` whose field of rational functions is included in the ring of 
        coefficients for the formal pwoer series.
    '''
    def __init__(self, base, universe=None, degree=True, var_name = None, q_name="q"):
            # if the base has no q, we add it
            with_q, self.__q = has_variable(base, q_name)
            if not with_q:
                base = PolynomialRing(base, q_name).fraction_field()
                self.__q = base.gens()[0]
            PSBasis.__init__(self,base, universe, degree, var_name)

    ### Getters from the module variable as objects of the class
    def OB(self):
        return get_rational_algebra('q_n', self.base)[0]

    def n(self):
        r'''
            Method to get the generic multiplication for the recurrences.

            For the `q`-series, this operator is the multiplication by `q^n` (see
            :func:`QPower` to obtain the actual sequence). Here, we return a polynomial
            variable that will represent this multiplication (with name `q_n`)

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = QBasis(QQ) # illegal building, do not use in general
                sage: B.n()
                q_n
                sage: B.n().parent()
                Fraction Field of Univariate Polynomial Ring in q_n over Fraction Field of Univariate Polynomial Ring in q over Rational Field
        '''
        return self.OB().gens()[0]

    def q(self):
        r'''
            Method to get the parameter `q` for the recurrences. This value is defined with the 
            argument ``q_name`` when initializing the basis (see :class:`QBasis`)

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal building, do not use in general
                sage: B.q()
                q
                sage: B.q().parent()
                Fraction Field of Univariate Polynomial Ring in q over Rational Field
        '''
        return self.__q

    def OS(self):
        r'''
            Method to get the generic variable :class:`~ore_algebra.OreAlgebra` with only the direct shift 
            over the rational functions in `n`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = QBasis(QQ) # illegal building, do not use in general
                sage: B.OSS()
                Univariate Ore algebra in Sn over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return get_double_qshift_algebra(str(self.n()), str(self.q()), "Sn", base=self.base)[0]

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
        return get_qshift_algebra(str(self.n()), str(self.q()), "Sn", base=self.base)[0]

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
        return self.n()

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
        return self.OS().gens()[0]

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
        return self.OS().gens()[1]
    
    def valid_factor(self, element):
        if(not element in self.OB()):
            return False
        element = self.OB()(element)

        ## We check the denominator never vanishes on positive integers
        for root in element.denominator().roots(): # checking if they are of the form q^m for positive m
            if len(root[0].numerator().coefficients()) == 1 and root[0].denominator().is_one():
                return False

        ## We check the numerator never vanishes on the positive integers
        for root in element.numerator().roots():
            if len(root[0].numerator().coefficients()) == 1 and root[0].denominator().is_one():
                return False
            
        return True

    @cached_method
    def QPower(self, a : int = 1):
        r'''
            Method that obtains the corresponding sequence `1, q, q^2, \ldots` for the corresponding `q`.

            The multiplication by this sequence is the equivalent action related with the `Q` operator obtained 
            by `:func:`Q`.
        '''
        return LambdaSequence(lambda n : self.q()**(a*n), self.base)

    @cached_method
    def QFactorial(self):
        r'''
            Method that obtains the sequence of `q`-factorials.
        '''
        return LambdaSequence(lambda n : qfactorial(n, self.q()), self.base)

    @cached_method
    def QPochammer(self, a):
        r'''
            Method that obtains the sequence of `q`-pochammer symbols from a fixed base value.
        '''
        return LambdaSequence(lambda n : qpochammer(a, n, self.q()), self.base)

    ## Reperesentation methods
    def __repr__(self):
        return f"QBasis -- WARNING: this is an abstract class"

class QSequentialBasis(QBasis, SequenceBasis):
    r'''
        Class for `q`-formal power series given as a 2-dimensional sequence.
    '''
    def __init__(self, base, sequence: Sequence, degree: bool = True, q_name: str = "q"):
        QBasis.__init__(self, base, None, False, None, q_name)
        SequenceBasis.__init__(self, self.base, sequence, degree)
        
    ## Other ovewriten methods
    def change_base(self, base):
        return QSequentialBasis(base, self.functional_seq, self.by_degree(), str(self.q()))

    ## TODO: Check methods of compatibility

    ## TODO: Check method scalar

    def shift_in(self, shift):
        return QSequentialBasis(self.base, self.functional_seq.shift(0, shift), self.by_degree, str(self.q()))

    def mult_in(self, prod):
        return QSequentialBasis(self.base, LambdaSequence(lambda n,k : (prod*self[n])[k], self.base, 2), self.by_degree())

    ## Reperesentation methods
    def __repr__(self):
        return f"QSequentialBasis -- WARNING: this is an abstract class"

## Transforming Polynomial bases to the `q`-setting
class QPolyBasis(QBasis, PolyBasis):
    r'''
        Abstract class for a polynomial `q`-power series basis. 
        
        Their elements must be indexed by natural numbers such that the n-th
        element of the basis has degree exactly `n`.

        Since this is a `q`-basis, the variable will represent the sequence `q^n`.
        
        This class **must never** be instantiated.
    '''
    def __init__(self, base=QQ, var_name="q_n", q_name = "q"):
        QBasis.__init__(self, base, None, True, var_name, q_name)
        PolyBasis.__init__(self, self.base, var_name)

    @PSBasis.functional_seq.getter
    def functional_seq(self) -> Sequence:
        q = self.q()
        return LambdaSequence(lambda k,n : self[k](**{self.var_name : q**n}), self.base, 2, True)

    @PSBasis.evaluation_seq.getter
    def evaluation_seq(self) -> Sequence:
        raise NotImplementedError("The evaluation of `q`-series is not property defined")

    def __repr__(self):
        return "QPolyBasis -- WARNING: this is an abstract class"

class QFactorialBasis(FactorialBasis, QPolyBasis):
    def __init__(self, q='q', X='x', base=QQ):
        QPolyBasis.__init__(self, base, var_name=X, q_name=q)
        super().__init__(X, self.base)

class QSFactorialBasis(SFactorialBasis, QFactorialBasis):
    def __init__(self, an, bn, q='q', X='x', init=1, base=QQ):
        QFactorialBasis.__init__(self, q, X, base)
        super().__init__(an, bn, X, init, self.base)

class QRootSequenceBasis(RootSequenceBasis, QFactorialBasis):
    def __init__(self, rho, cn, q='q', X='q_n', base=QQ):
        QFactorialBasis.__init__(self, q, X, base)
        super().__init__(rho, cn, X, self.base)

class QScalarBasis(ScalarBasis, QFactorialBasis):
    def __init__(self, basis: QFactorialBasis, scale):
        super().__init__(basis, scale)

class QBinomialBasis(QSFactorialBasis):
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

            \left[\begin{array}{c}an + b\\ m\end{array}\right]_q

        This class represents precisely this basis and allow to compute compatible operators for some 
        default operators such as:

        * ``Sn``: the shift operator in `n`.
        * ``Q``: the multiplication by the sequence `(q^n)_n`.

        INPUT:

        * ``base``: (`\mathbb{Q}` by default) base ring for the `q`-basis. Changing this allow to add some parameters to the computations.
        * ``a``: (1 by default) value for the coefficient `a`.
        * ``b``: (0 by default) value for the coefficient `b`.
        * ``q``: ("q" by default) the name for the `q` parameter.
        * ``X``: ("q_n" by default) the name of the polynomial variable. It represent the multiplication by :func:`QPower` (i.e., `q^n`).
        * ``E``: ("E" by default) the name for the compatible shift with this basis.
    '''
    def __init__(self, a=1, b=0, name_q='q', name_X='q_n', name_E='E', base=QQ):
        QBasis.__init__(self, base, degree=True, var_name=name_X, q_name=name_q) # initializing some default variables for using ``self.q``
        if(not a in ZZ or a <= 0):
            raise b("The dilation of the basis must be a natural number")
        a = ZZ(a); b = self.base(b)
        self.__a = a
        self.__b = b
        self.__E_name = name_E

        qn = self.Q(); q = self.q()
        super(QBinomialBasis, self).__init__(
            -q/(qn*(1-qn)), 
            1/(1-qn),
            name_q,
            name_X, 
            base=self.base)

        ## TODO: Add compatibilities

    def linear_coefficient(self) -> Sequence:
        q = self.q()
        return LambdaSequence(lambda n : self.symb_an(**{self.var_name : q**n}), self.base, allow_sym=False)

    def constant_coefficient(self) -> Sequence:
        q = self.q()
        return LambdaSequence(lambda n : self.symb_bn(**{self.var_name : q**n}), self.base, allow_sym=False)

    def change_base(self, base):
        return QBinomialBasis(
            self.__a,                       # the dilation value does not change
            self.__b,                       # the shift value does not change
            str(self.universe.gens()[0]),   # the name of the variable does not change
            self.__E_name,                  # the name for the shifts does not change
            base                            # the base is set to the given value
        )

    def is_quasi_func_triangular(self):
        return self.__b == 0

    def __repr__(self):
        upper = [("" if self.__a == 1 else f"{self.__a}") + "n"]
        if self.__b != 0:
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
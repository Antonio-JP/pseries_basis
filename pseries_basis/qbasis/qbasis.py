r'''
    Module with the basic classes for implementing `q`-series.
'''
import logging
logger = logging.getLogger(__name__)

from sage.all import prod, PolynomialRing, QQ, ZZ, cached_method, Matrix, ceil

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
        coefficients for the formal power series.

        In general, the best analogy between the `q`-sequences and the usual sequences defined by
        the class :class:`PSBasis` is that the power basis `((n^k)_n)_k` is transformed to the `q`-power 
        basis `((q^{kn})_n)_k` (i.e., `n \mapsto q^n`).

        Hence, many of the structures developed for the classical formal power series still work
        for the `q`-analog, and we can reuse them by adapting slightly the ground ring over which the 
        formal power series are defined. In particular, we need to provide a name for `q` (which is "q"
        by default) and the variable name will be interpreted as the sequence `q^n` (and it is "q_n" by default)

            sage: from pseries_basis import *
            sage: B = QBasis(QQ) # abstract class
            sage: B.OB() # returns the ring with `q_n`
            Fraction Field of Univariate Polynomial Ring in q_n over Fraction Field of Univariate Polynomial Ring in q over Rational Field
            sage: B.n() # returns the `q_n`
            q_n
            sage: B.q() # returns the actual `q` in the most basic field
            q
            sage: B.q().parent()
            Fraction Field of Univariate Polynomial Ring in q over Rational Field
            sage: B2 = QBasis(QQ['a'], var_name = "Q", q_name="b")
            sage: B2.OB() # the q_n appears for future uses in the recurrences
            Fraction Field of Univariate Polynomial Ring in q_n over Fraction Field of Univariate Polynomial Ring in b over Univariate Polynomial Ring in a over Rational Field
            sage: B2.universe # the Q is the variable interpreted as the `q^n` sequence
            Univariate Polynomial Ring in Q over Fraction Field of Univariate Polynomial Ring in b over Univariate Polynomial Ring in a over Rational Field
            sage: B2.n()
            q_n
            sage: B2.n().parent()
            Fraction Field of Univariate Polynomial Ring in q_n over Fraction Field of Univariate Polynomial Ring in b over Univariate Polynomial Ring in a over Rational Field
            sage: B2.q() # we have chosen that our `q` is called `b`
            b
            sage. B2.q().parent()
            Fraction Field of Univariate Polynomial Ring in b over Univariate Polynomial Ring in a over Rational Field

        The operator rings for the recurrences that originally was defined using the natural shift `x \mapsto x+1`
        now can be defined with the shift `q_n \mapsto qq_n`. This can be done using the module :mod:`ore_algebra`.
        The usual method to obtain these structures has been adapted for :class:`QBasis`::

            sage: B.OS()
            Multivariate Ore algebra in Sn, Sni over Fraction Field of Univariate Polynomial Ring in q_n over Fraction Field of Univariate Polynomial Ring in q over Rational Field
            sage: B.OSS()
            Univariate Ore algebra in Sn over Fraction Field of Univariate Polynomial Ring in q_n over Fraction Field of Univariate Polynomial Ring in q over Rational Field
            sage: B.Sn()
            Sn
            sage: B.Sn().parent()
            Multivariate Ore algebra in Sn, Sni over Fraction Field of Univariate Polynomial Ring in q_n over Fraction Field of Univariate Polynomial Ring in q over Rational Field
        
        Moreover, the method :func:`~pseries_basis.psbasis.PSBasis.recurrence_vars` includes the `q` name before the 
        names corresponding to the variable name and the shifts::

            sage: B.recurrence_vars()
            (q, q_n, Sn, Sni)
            sage: B2.recurrence_vars()
            (b, q_n, Sn, Sni)

        INPUT:

        * ``base``: the base ring for the `q`-basis. If this ring has no `q` (given by ``q_name``), we will include this new variable.
        * ``universe``: the universe for the `q`-basis viewed as a sequence of sequences. If ``None`` is given, we will decide the universe
          as was done in :class:`~pseries_basis.psbasis.PSBasis`.
        * ``degree``: see :class:`~pseries_basis.psbasis.PSBasis`.
        * ``var_name``: see :class:`~pseries_basis.psbasis.PSBasis`. In this case, the variable will represent 
          the sequence `(q^n)_n`.
        * ``q_name``: name for the `q` variable. If not in ``base``, it will be added as a rational function field.
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
        return self.OB().gens()[0]

    def q(self):
        return self.__q

    def OS(self):
        return get_double_qshift_algebra(str(self.n()), str(self.q()), "Sn", base=self.base)[0]

    def OSS(self):
        return get_qshift_algebra(str(self.n()), str(self.q()), "Sn", base=self.base)[0]

    def Q(self):
        return self.n()
    
    def is_q_hypergeometric(self, element):
        r'''
            Method to check if a symbolic expression is `q`-hypergeometric or not.

            This method checks whether ``element`` is a symbolic expression or a function
            with a parameter `n` that is hypergeometric. 

            This method returns ``True`` or ``False`` and the quotient (if the output is hypergeometric)
            or ``None`` otherwise.

            INPUT:

            * ``element``: the object that will be checked.

            TODO: Add examples and implement the method to work also with some specific sequences.
        '''
        raise NotImplementedError("The `q`-hypergeometric checker is not yet implemented")

    def valid_factor(self, element):
        r'''
            Checks whether a rational function has poles or zeros in the positive integers.

            This method overrides the method in :class:`~pseries_basis.psbasis.PSBasis`. Now, the
            variable element of the basis (see :func:`var_name`) change the meaning (from the sequence
            `(n)_n` to `(q^n)_n`) then the set of valid factors (see :func:`~pseries_basis.psbasis.PSBasis.valid_factor`)
            changes. 

            Instead of looking for zeros of the given element for positive integers, we nos need to check 
            that there is no zeros of shape `q^n` for a positive `n`.

            INPUT:

            * ``element``: rational function in `q_n` (see :func:`OB`).
            
            OUTPUT:

            This method return ``True`` if the rational function has no pole nor zero on `\{q^n\ :\ n \in \mathbb{N}\}`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = QBasis(QQ); q,q_n,_,_ = B.recurrence_vars()
                sage: B.valid_factor(q)
                True
                sage: B.valid_factor(q*q_n)
                True
                sage: B.valid_factor(1-q_n)
                False
                sage: B.valid_factor(2-q_n)
                True
                sage: B.valid_factor(q^2 - q_n)
                False
                sage: B.valid_factor((q + q_n)*(q^2 + q_n))
                True
                sage: B.valid_factor((q_n+q)/(1-q_n))
                False
                sage: B.valid_factor((q_n+q)/(q_n))
                True
        '''
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

    ## Compatibility methods
    def _compatibility_from_recurrence(self, recurrence):
        Sn = self.Sn(); Sni = self.Sni(); qn = self.n(); q = self.q()
        A = recurrence.degree(Sn); B = recurrence.degree(Sni); recurrence = recurrence.polynomial()
        alpha = ([self.OB()(recurrence.coefficient({Sn:i}))(**{str(qn) : q**(-i)*qn}) for i in range(A, 0, -1)] + 
                [self.OB()(recurrence.constant_coefficient())] + 
                [self.OB()(recurrence.coefficient({Sni:i}))(**{str(qn) : q**(i)*qn}) for i in range(1, B+1)])
        return (ZZ(A), ZZ(B), ZZ(1), lambda _, j, k: alpha[j+A](**{str(qn) : k}))

    def _recurrence_from_compatibility(self, compatibility):
        if not isinstance(compatibility, (list, tuple)) or len(compatibility) != 4:
            raise TypeError("The compatibility condition is not valid")

        A,B,m,alpha = compatibility
        ## We do the transformation
        Sn = self.Sn(); Sni = self.Sni(); q_n = self.n(); q = self.q()
        SN = lambda n : Sn**n if n > 0 else Sni**(-n) if n < 0 else 1        
        # We have to distinguish between m = 1 and m > 1
        if(m == 1): # we return an operator
            recurrence = sum(alpha(0,i,q_n*q**(i))*SN(-i) for i in range(-A,B+1))
            output = self.simplify_operator(recurrence)
        elif(m > 1):
            output = Matrix(
                [
                    [self.simplify_operator(sum(
                        alpha(j,i,q_n*q**(r-i-j)//m)*SN((r-i-j)//m)
                        for i in range(-A,B+1) if ((r-i-j)%m == 0)
                    )) for j in range(m)
                    ] for r in range(m)
                ])
        else:
            raise TypeError(f"The number of sections must be a positive integer (got {m})")
        return output

    ## Example sequences
    @cached_method
    def QPower(self, a : int = 1):
        r'''
            Method that obtains the corresponding sequence `1, q^a, q^{2a}, \ldots` for the corresponding `q`.

            This sequences make a full basis when `a \in \mathbb{N}` and the multiplication by this basis when 
            `a = 1` is the meaning of the operator obtained by :func:`Q` or :func:`n`.

            INPUT:

            * ``a``: an integer that defines the corresponding power sequence.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = QBasis(QQ)
                sage: B.QPower()
                Sequence over [Fraction Field of Univariate Polynomial Ring in q over Rational Field]: (1, q, q^2,...)
                sage: B.QPower(10)[:10] == [B.q()**(10*i) for i in range(10)]
                True

            The returned sequence will be in the corresponding universe defined from the `q`-basis::

                sage: B2 = QBasis(QQ['a'], var_name='b', q_name='c')
                Sequence over [Fraction Field of Univariate Polynomial Ring in c over Univariate Polynomial Ring in a over Rational Field]: (1, c, c^2,...)

            Check the class :class:`QPowerBasis` to see how to obtain a basis and compatibilities using these sequences.
        '''
        return LambdaSequence(lambda n : self.q()**(a*n), self.base)

    @cached_method
    def QNaturals(self):
        r'''
            Method that obtains the sequence of `q`-factorials.
            
            This method returns the sequence `([n]_q)_n`, i.e., the sequences of `q`-analog of the natural numbers.
            These analogs of the natural numbers are defined by `[n]_q = 1 + q + \ldots + q^{n-1}`. Hence, when 
            `q\rightarrow 1`, we have `[n]_q \rightarrow n`.

            This sequence does not depend on the basis defined by ``self``, but the universe of ``self`` will define
            the universe of the returned sequence.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = QBasis(QQ)
                sage: B.QNaturals()
                Sequence over [Fraction Field of Univariate Polynomial Ring in q over Rational Field]: (0, 1, q + 1,...)
                sage: B.QNaturals()[:5]
                [0, 1, q + 1, q^2 + q + 1, q^3 + q^2 + q + 1]

            By definition, we have that the forward difference of this sequence will be exactly
            the powers of `q`::

                sage: (B.QNaturals().shift(1) - B.QNaturals()).almost_equals(B.QPower(), 100)
                True

            The returned sequence will be in the corresponding universe defined from the `q`-basis::

                sage: B2 = QBasis(QQ['a'], var_name='b', q_name='c')
                sage: B2.QNaturals()
                Sequence over [Fraction Field of Univariate Polynomial Ring in c over Univariate Polynomial Ring in a over Rational Field]: (0, 1, c + 1,...)
        '''
        return LambdaSequence(lambda n : sum(self.q()**i for i in range(n)), self.base)

    @cached_method
    def QFactorial(self):
        r'''
            Method that obtains the sequence of `q`-factorials.
            
            This method returns the sequence `([n]_q!)_n`, i.e., the `n`-th element of the sequence is the 
            value of the `n`-th `q`-factorial. This is defined using the `q`-analog of the natural numbers
            where `[n]_q = 1 + q + \ldots + q^{n-1}`. Hence, the `q`-factorial is the product of all the `q`-natural
            numbers.

            This sequence does not depend on the basis defined by ``self``, but the universe of ``self`` will define
            the universe of the returned sequence.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = QBasis(QQ)
                sage: B.QFactorial()
                Sequence over [Fraction Field of Univariate Polynomial Ring in q over Rational Field]: (0, 1, q + 1,...)
                sage: B.QFactorial()[:5]
                [0, 1, q + 1, q^3 + 2*q^2 + 2*q + 1, q^6 + 3*q^5 + 5*q^4 + 6*q^3 + 5*q^2 + 3*q + 1]

            By definition, we have that the quotient of the shift and this sequence will be exactly
            the `q`-natural numbers::

                sage: (B.QFactorial().shift(1)/B.QFactorial()).almost_equals(B.QNaturals().shift(), 100)
                True

            The returned sequence will be in the corresponding universe defined from the `q`-basis::

                sage: B2 = QBasis(QQ['a'], var_name='b', q_name='c')
                sage: B2.QFactorial()
                Sequence over [Fraction Field of Univariate Polynomial Ring in c over Univariate Polynomial Ring in a over Rational Field]: (1, 1, c + 1,...)
        '''
        return LambdaSequence(lambda n : qfactorial(n, self.q()), self.base)

    @cached_method
    def QPochhammer(self, a):
        r'''
            Method that obtains the sequence of `q`-pochhammer symbols from a fixed base value.

            The general Pochhammer symbol is the rising or ascending factorial. This can be seen as the product
            of the some elements `a, a+1, a+2, \ldots`. 

            Once we have defined the `q`-naturals and the `q`-factorial numbers, it is just natural to define
            a `q`-analog for the Pochhammer symbol (usually represented by `(a;q)_k`). At the end, we can 
            write the following formula:

            .. MATH::

                (a;q)_n = \prod_{i=0}^n (1-aq^i)

            This definition is valid for any value of `a` and not only for natural numbers. 

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = QBasis(QQ)
                sage: B.QPochhammer(1)
                Sequence over [Fraction Field of Univariate Polynomial Ring in q over Rational Field]: (1, 0, 0,...)
                sage: B.QPochhammer(2)
                Sequence over [Fraction Field of Univariate Polynomial Ring in q over Rational Field]: (1, -1, 2*q - 1,...)
            
            We can even use the `q` value as input for the `q`-Pochhammer symbol::

                sage: B.QPochhammer(B.q())
                Sequence over [Fraction Field of Univariate Polynomial Ring in q over Rational Field]: (1, -q + 1, q^3 - q^2 - q + 1,...)
                
            We can check the general identity of the `q`-Pochhammer symbol relating with the `q`-factorial::

                sage: (B.QPochammer(B.q())/LambdaSequence(lambda n : (1-B.q())**n, B.base) - B.QFactorial())[:10]
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        '''
        return LambdaSequence(lambda n : qpochhammer(a, n, self.q()), self.base)
 
    def shift_in(self, shift):
        raise NotImplementedError(f"Method 'shift_in' not implemented for class {self.__class__}")

    def mult_in(self, prod):
        r'''
            TODO: Add documentation and examples to this method
        '''
        return QSequentialBasis(
            self.base, # the base ring stays the same (with `q`)
            LambdaSequence(lambda n,k : prod(k)*self.functional_seq(n,k), self.functional_seq.universe, 2), # the sequence that defines the product
            self.by_degree(), # the by_degree attribute is the same
            str(self.q()) # the name for `q` stays the same
        )

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

    def _scalar_basis(self, factor) -> "QFactorialBasis":
        r'''
            TODO: Add documentation and examples
        '''
        return QScalarBasis(self, factor)

    def _scalar_hypergeometric(self, factor, quotient) -> "QFactorialBasis":
        r'''
            TODO: Add documentation and examples
        '''
        return QScalarBasis(self, factor)

class QSFactorialBasis(SFactorialBasis, QFactorialBasis):
    def __init__(self, an, bn, q='q', X='x', init=1, base=QQ):
        QFactorialBasis.__init__(self, q, X, base)
        super().__init__(an, bn, X, init, self.base)

        ## If the compatibility by X is not set, we create it here
        if not self.has_compatibility(X):
            try:
                Sni = self.Sni(); qn = self.n(); q = self.q(); an = self.symb_an; bn = self.symb_bn
                self.set_compatibility(X, -bn(**{str(qn) : q*qn})/an(**{str(qn) : q*qn}) + (1/an)*Sni) # pylint: disable=invalid-unary-operand-type
            except (AttributeError, TypeError) as e:
                logger.info(f"Error with the compatibility with {X} -->\n\t{e}")
                pass

    def linear_coefficient(self) -> Sequence:
        q = self.q()
        return LambdaSequence(lambda n : self.symb_an(**{self.var_name : q**n}), self.base, allow_sym=False)

    def constant_coefficient(self) -> Sequence:
        q = self.q()
        return LambdaSequence(lambda n : self.symb_bn(**{self.var_name : q**n}), self.base, allow_sym=False)

    def shift_in(self, shift):
        q = self.q()
        return QSFactorialBasis(q**shift * self.symb_an, self.symb_bn, str(self.q()), str(self.n()), base = self.base)

class QRootSequenceBasis(RootSequenceBasis, QFactorialBasis):
    def __init__(self, rho, cn, q='q', X='q_n', base=QQ):
        QFactorialBasis.__init__(self, q, X, base)
        super().__init__(rho, cn, X, self.base)

class QScalarBasis(ScalarBasis, QFactorialBasis):
    r'''
        Class to represent the scaling of a `q`-factorial basis.

        Let `\mathcal{B} = (B_k(n))_k` be a `q`-factorial basis (i.e., the element `B_k(n)` is 
        a sequence that can be written as a polynomial of degree `k` in `q^n`). Then it is clear that
        for any sequence `(c_k)_k \in \mathbb{K}(q)^\mathbb{N}`, the new sequence 

        .. MATH::

            C_k(n) = c_kB_k(n)

        is again a `q`-factorial basis if:

        * `c_k \neq 0` for all `k\in \mathbb{N}`,
        * `c_k` is `q`-hypergeometric.

        Moreover, if `L` is a `(A,B)`-compatible operator with `\mathcal{B}`, then `L` is also `(A,B)`-compatible
        with the new scaled basis `\mathcal{C} = (C_k(n))_k`:

        .. MATH::

            \begin{array}{rcl}  L C_k(n) & = & c_kLB_k(n) = c_k \sum_{i=-A}^B B_{k+i}(n) = \sum_{i=-A}^B \frac{c_k}{c_{k+i}}c_{k+i}B_{k+i}(n)\\ & = & \sum_{i=-A}^B \frac{c_k}{c_{k+i}}C_{k+i}(n)\end{array}

        This class allows the user to compute the sclaed basis given the original `q`-factorial basis `\mathcal{B}` 
        and the sacling sequence `(c_k)_k`.

        INPUT:

        * ``basis``: a `q`-factorial basis to be scaled.
        * ``scale``: a hypergeometric term (see :func:`QBasis.is_q_hypergeometric`)

        EXAMPLES::

            sage: from pseries_basis import *
            sage: B = QBinomialBasis(); q = B.q(); qn = B.n()
            sage: C = QScalarBasis(B, qn)
            sage: 
    '''
    def __init__(self, basis: QFactorialBasis, scale):
        if not isinstance(basis, QFactorialBasis):
            raise TypeError(f"The given basis must be a QFactorialBasis. Got {basis.__class__}")
        is_hyper, _ = basis.is_q_hypergeometric(scale)
        if not is_hyper:
            raise TypeError(f"The given scaling sequence ([{scale}]) must be `q`-hypergeometric.")

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

            \left[\begin{array}{c}n + b\\ m\end{array}\right]_q

        This class represents precisely this basis and allow to compute compatible operators for some 
        default operators such as:

        * ``Sn``: the shift operator in `n`.
        * ``Q``: the multiplication by the sequence `(q^n)_n`.

        INPUT:

        * ``base``: (`\mathbb{Q}` by default) base ring for the `q`-basis. Changing this allow to add some parameters to the computations.
        * ``b``: (0 by default) value for the coefficient `b`.
        * ``q``: ("q" by default) the name for the `q` parameter.
        * ``X``: ("q_n" by default) the name of the polynomial variable. It represent the multiplication by :func:`QPower` (i.e., `q^n`).
        * ``E``: ("E" by default) the name for the compatible shift with this basis.
    '''
    def __init__(self, b=0, name_q='q', name_X='q_n', name_E='E', base=QQ):
        QBasis.__init__(self, base, degree=True, var_name=name_X, q_name=name_q) # initializing some default variables for using ``self.q``
        if(not b in ZZ):
            raise b("The shift of the basis must be a natural number")
        b = ZZ(b)
        self.__b = b
        self.__E_name = name_E

        qn = self.Q(); q = self.q(); Sn = self.Sn()
        super(QBinomialBasis, self).__init__(
            -(q**(b+1)/(qn*(1-qn))), 
            1/(1-qn),
            name_q,
            name_X, 
            base=self.base)

        self.set_endomorphism(name_E, qn + Sn)

    def change_base(self, base):
        return QBinomialBasis(
            self.__a,                       # the dilation value does not change
            self.__b,                       # the shift value does not change
            str(self.universe.gens()[0]),   # the name of the variable does not change
            self.__E_name,                  # the name for the shifts does not change
            base                            # the base is set to the given value
        )

    def is_quasi_func_triangular(self):
        return self.__b >= 0

    def shift_in(self, shift):
        return QBinomialBasis(self.__b + shift, str(self.q()), str(self.n()), self.__E_name, self.base)

    def __repr__(self):
        upper = ["n"]
        if self.__b != 0:
            upper.append(f"{self.__b}")

        return f"Q-Binomial Basis [n{f' + {self.__b}' if self.__b != 0 else ''}; m]_q"

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
def check_q_compatibility(basis : QBasis, operator, action, bound=100):
    r'''
        Method that checks that a compatibility formula holds for some examples.

        This method takes a :class:`QBasis`, an operator compatibility (either a tuple with the 
        compatibility data or the operator that must be compatible with the basis), an actions with the 
        map for the opertator and a bound and checks that the induced compatibility identity holds for 
        the first terms of the basis.

        INPUT:

        * ``basis``: a :class:`QBasis` to be checked.
        * ``operator``: a tuple `(A,B,m,\alpha)` with the compatibility condition (see :func:`PSBasis.compatibility`)
          or a valid input of that method.
        * ``action``: a map that takes elements in ``basis.universe`` and perform the operation of ``operator``.
        * ``bound``: positive integer with the number of cases to be checked.
    '''
    if(isinstance(operator, tuple)):
        a,b,m,alpha = operator
    else:
        a,b,m,alpha = basis.compatibility(operator)
        
    mm = int(ceil(a/m)); q = basis.q()
    return all(
        all(
            sum(basis[k*m+r+i]*basis.base(alpha(r,i,q**k)) for i in range(-a,b+1)) == action(basis[k*m+r]) 
            for r in range(m)) 
        for k in range(mm, bound))

def qpochhammer(a, n : int, q):
    '''Method for obtaining (a;q)_n'''
    return prod(1-a*q**k for k in range(n))

def qfactorial(n:int, q):
    '''Method for obtaining [n]_q! = (q;q)_n/(1-q)^n'''
    return qpochhammer(q, n, q)/(1-q)**n

def qbinomial(n:int, m:int, q):
    '''Method for obtaining [n,m]_q = [n]_q!/([m]_q![n-m]_q!)'''
    if m < 0 or m > n:
        return 0
    return qfactorial(n,q)/(qfactorial(m, q)*qfactorial(n-m, q))

__all__ = ["QBasis", "QSequentialBasis", "QBinomialBasis", "check_q_compatibility"]
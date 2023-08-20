r'''
    Module with the basic classes for implementing `q`-series.
'''
import logging
logger = logging.getLogger(__name__)

from sage.all import PolynomialRing, QQ, ZZ, cached_method, Matrix, ceil, parent #pylint: disable=no-name-in-module
from sage.categories.pushout import pushout
from sage.structure import element

from ..psbasis_old import PSBasis, PolyBasis, SequenceBasis
from ..polynomial.factorial import FactorialBasis, RootSequenceBasis
from ..misc.ore import get_qshift_algebra, get_double_qshift_algebra, get_rational_algebra, has_variable
from ..misc.qsequences import QLambdaSequence
from ..misc.sequences import LambdaSequence, Sequence

Element = element.Element

#######################################################################################################
### Private methods of this module
#######################################################################################################
def __polyvar_log(poly, var):
    r'''Compute the `n` for ``poly == var**n` for a variable ``var`` of a polynomial ring'''
    if not poly in var.parent():
        raise ValueError(f"[polyvar_log] {poly} is not a polynomial in {var}")
    poly = var.parent()(poly)
    rems = []
    while not poly.is_constant():
        rems.append(poly%var)
        poly = poly//var
    if all(el == 0 for el in rems) and poly == 1:
        return len(rems)
    elif poly != 1:
        raise ValueError(f"[polyvar_log] The leading coefficient of {poly} was not 1")
    else:
        raise ValueError(f"{poly} was not of the form {var}^n")

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
            Fraction Field of Univariate Polynomial Ring in q_n over Fraction Field of Multivariate Polynomial Ring in a, b over Rational Field
            sage: B2.universe # the Q is the variable interpreted as the `q^n` sequence
            Univariate Polynomial Ring in Q over Fraction Field of Multivariate Polynomial Ring in a, b over Rational Field
            sage: B2.n()
            q_n
            sage: B2.n().parent()
            Fraction Field of Univariate Polynomial Ring in q_n over Fraction Field of Multivariate Polynomial Ring in a, b over Rational Field
            sage: B2.q() # we have chosen that our `q` is called `b`
            b
            sage: B2.q().parent()
            Fraction Field of Multivariate Polynomial Ring in a, b over Rational Field

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
        names corresponding to the variable name and the shifts. This method also includes all variables that share ring with the `q`::

            sage: B.recurrence_vars()
            (q, q_n, Sn, Sni)
            sage: B2.recurrence_vars()
            (a, b, q_n, Sn, Sni)

        INPUT:

        * ``base``: the base ring for the `q`-basis. If this ring has no `q` (given by ``q_name``), we will include this new variable.
        * ``universe``: the universe for the `q`-basis viewed as a sequence of sequences. If ``None`` is given, we will decide the universe
          as was done in :class:`~pseries_basis.psbasis.PSBasis`.
        * ``degree``: see :class:`~pseries_basis.psbasis.PSBasis`.
        * ``var_name``: see :class:`~pseries_basis.psbasis.PSBasis`. In this case, the variable will represent 
          the sequence `(q^n)_n`.
        * ``q_name``: name for the `q` variable. If not in ``base``, it will be added as a rational function field.
    '''
    def __init__(self, base=QQ, universe=None, degree=True, var_name = None, q_name="q", power=1, **kwds):
            # if the base has no q, we add it
            with_q, self.__q = has_variable(base, q_name)
            if not with_q:
                base = PolynomialRing(base, q_name).flattening_morphism().codomain().fraction_field()
                self.__q = base(q_name)

            if not power in ZZ or power < 0:
                raise ValueError("The given power (setting the base) must be a non-negative integer")
            self.__power = ZZ(power)

            super().__init__(
                base=base, universe=universe, degree=degree, var_name=var_name, # arguments for PSBasis
                **kwds # arguments maybe used for multi-inheritance
            )

    ### Getters from the module variable as objects of the class
    def OB(self):
        return get_rational_algebra('q_n', self.base)[0]

    def n(self):
        return self.OB().gens()[0]

    def q(self):
        return self.__q

    def OS(self):
        return get_double_qshift_algebra(str(self.n()), str(self.q()), "Sn", power = self.__power, base=self.base)[0]

    def OSS(self):
        return get_qshift_algebra(str(self.n()), str(self.q()), "Sn", power=self.__power, base=self.base)[0]

    def Q(self):
        return self.n()
    
    def is_hypergeometric(self, element):
        r'''
            Method to check if a symbolic expression is `q`-hypergeometric or not.

            This method checks whether ``element`` is a symbolic expression or a function
            with a parameter `n` that is hypergeometric. 

            This method returns ``True`` or ``False`` and the quotient (if the output is hypergeometric)
            or ``None`` otherwise.

            INPUT:

            * ``element``: the object that will be checked.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = QBasis(QQ); q,q_n,_,_ = B.recurrence_vars()
                sage: B.is_hypergeometric(B.QPochhammer(B.q())) # (q;q)_n is q-hypergeometric
                True, -q*q_n + 1

            TODO: Add examples and implement the method to work also with some specific sequences.
        '''
        # Basic case of rational functions in self.OB()
        if(element in self.OB()):
            element = self.OB()(element); qn = self.n(); q = self.q()
            return True, element(**{str(qn):q*qn})/element
        elif isinstance(element, Sequence):
            from ore_algebra import guess
            shift = 5 # this is an arbitrary choice
            try:
                op = guess(element[shift:shift+10], self.OSS(), order = 1)
                if op.order() == 1:
                    op = self.remove_Sni(self.simplify_operator(self.Sni()**shift * op * self.Sn()**shift))
                    return True, -op[0]/op[1]
            except ValueError:
                pass
        else: # we assume the input is a Symbolic element
            raise NotImplementedError("The `q`-hypergeometric checker for symbolic expressions is not yet implemented")
        
        return False, None

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
        return not any(
            ( # two possible conditions to return False
                r == 1 or
                ( # the second condition is the "and" of four conditions
                  # we use "and" instead of "all" to avoid computing parts that make no sense
                    r.numerator().degree() > 0 and
                    len(r.numerator().coefficients()) == 1 and
                    r.numerator().coefficients()[0] == 1 and
                    r.denominator().is_one()
                )
            )
        for r,_ in element.denominator().roots() + element.numerator().roots())

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
            recurrence = sum(alpha(0,i,q_n*q**(-i))*SN(-i) for i in range(-A,B+1))
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
                sage: B2.QPower()
                Sequence over [Fraction Field of Multivariate Polynomial Ring in a, c over Rational Field]: (1, c, c^2,...)

            Check the class :class:`QPowerBasis` to see how to obtain a basis and compatibilities using these sequences.
        '''
        return QLambdaSequence(lambda qn : qn**a, self.base, q_name = str(self.q()))

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
                Sequence over [Fraction Field of Multivariate Polynomial Ring in a, c over Rational Field]: (0, 1, c + 1,...)
        '''
        # q = self.q()
        # return QLambdaSequence(lambda qn : (1-qn)/(1-q), self.base, q_name=str(q))
        return QNaturalSequence(self.q())

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
                Sequence over [Fraction Field of Univariate Polynomial Ring in q over Rational Field]: (1, 1, q + 1,...)
                sage: B.QFactorial()[:5]
                [1, 1, q + 1, q^3 + 2*q^2 + 2*q + 1, q^6 + 3*q^5 + 5*q^4 + 6*q^3 + 5*q^2 + 3*q + 1]

            By definition, we have that the quotient of the shift and this sequence will be exactly
            the `q`-natural numbers::

                sage: (B.QFactorial().shift(1)/B.QFactorial()).almost_equals(B.QNaturals().shift(), 100)
                True

            The returned sequence will be in the corresponding universe defined from the `q`-basis::

                sage: B2 = QBasis(QQ['a'], var_name='b', q_name='c')
                sage: B2.QFactorial()
                Sequence over [Fraction Field of Multivariate Polynomial Ring in a, c over Rational Field]: (1, 1, c + 1,...)
        '''
        # q = self.q()
        # return QLambdaSequence(lambda qn : qfactorial(qn,q), self.base, q_name = str(q))
        return QFactorialSequence(self.q())

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

                sage: (B.QPochhammer(B.q())/LambdaSequence(lambda n : (1-B.q())**n, B.base) - B.QFactorial())[:10]
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        '''
        # return QLambdaSequence(lambda qn : qpochhammer(a, qn, self.q()), self.base)
        return QPochhammerSequence(a, self.q())
 
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

class QSequentialBasis(QBasis, SequenceBasis):
    r'''
        Class for `q`-formal power series given as a 2-dimensional sequence.
    '''
    def __init__(self, base, sequence: Sequence, degree: bool = True, q_name: str = "q", **kwds):
        super().__init__(
            base=base, universe=kwds.pop("universe", None), degree=degree, var_name=kwds.pop("var_name", None), q_name=q_name, # arguments for QBasis
            sequence=sequence, **kwds# arguments for other builders (like SequenceBasis) allowing multi-inheritance
        )
        
    ## Other overwritten methods
    def change_base(self, base):
        return QSequentialBasis(base, self.functional_seq, self.by_degree(), str(self.q()))

    ## TODO: Check methods of compatibility

    ## TODO: Check method scalar

    def shift_in(self, shift):
        return QSequentialBasis(self.base, self.functional_seq.shift(0, shift), self.by_degree, str(self.q()))

    def mult_in(self, prod):
        return QSequentialBasis(self.base, LambdaSequence(lambda n,k : (prod*self[n])[k], self.base, 2), self.by_degree())

## Transforming Polynomial bases to the `q`-setting
class QPolyBasis(QBasis, PolyBasis):
    r'''
        Abstract class for a polynomial `q`-power series basis. 
        
        Their elements must be indexed by natural numbers such that the n-th
        element of the basis has degree exactly `n`.

        Since this is a `q`-basis, the variable will represent the sequence `q^n`.
        
        This class **must never** be instantiated.
    '''
    def __init__(self, base=QQ, var_name="q_n", q_name = "q", **kwds):
        super().__init__(
            base=base, universe=kwds.pop("universe", None), degree=kwds.pop("degree", True), var_name=var_name, q_name=q_name, # arguments for QBasis
            **kwds # other arguments (like PolyBasis)
        )
        
    @PSBasis.functional_seq.getter
    def functional_seq(self) -> Sequence:
        q = self.q()
        return LambdaSequence(lambda k,n : self[k](**{self.var_name : q**n}), self.base, 2, True)

    @PSBasis.evaluation_seq.getter
    def evaluation_seq(self) -> Sequence:
        raise NotImplementedError("The evaluation of `q`-series is not property defined")

class QFactorialBasis(QPolyBasis, FactorialBasis):
    def __init__(self, q_name='q', var_name='q_n', base=QQ, **kwds):
        super().__init__(
            base=base, var_name=var_name, q_name=q_name, # arguments for QPolyBasis
            **kwds # arguments for other builders (in particular FactorialBasis)
        )

    def _create_compatibility_X(self):
        if not self.has_compatibility(self.var_name):
            try:
                Sni = self.Sni(); qn = self.n(); q = self.q(); an = self.symb_an; bn = self.symb_bn
                self.set_compatibility(self.var_name, -bn(**{str(qn) : q*qn})/an(**{str(qn) : q*qn}) + (1/an)*Sni) # pylint: disable=invalid-unary-operand-type
            except (AttributeError, TypeError) as e:
                logger.info(f"Error with the compatibility with {self.var_name} -->\n\t{e}")
                pass

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

class QSFactorialBasis(QFactorialBasis, SFactorialBasis):
    def __init__(self, an=None, bn=None, q_name='q', var_name='x', init=1, base=QQ, **kwds):
        super().__init__(
            q_name=q_name, var_name=var_name, base=base, # arguments for QFactorialBasis
            an=an, bn=bn, init=init, **kwds # arguments for other constructors (such as SFactorialBasis)
        )

    def linear_coefficient(self) -> Sequence:
        q = self.q()
        return LambdaSequence(lambda n : self.symb_an(**{self.var_name : q**n}), self.base, allow_sym=False) # pylint: disable=not-callable

    def constant_coefficient(self) -> Sequence:
        q = self.q()
        return LambdaSequence(lambda n : self.symb_bn(**{self.var_name : q**n}), self.base, allow_sym=False) # pylint: disable=not-callable

    def shift_in(self, shift):
        q = self.q()
        return QSFactorialBasis(q**shift * self.symb_an, self.symb_bn, str(self.q()), str(self.n()), base = self.base)

class QRootSequenceBasis(QFactorialBasis, RootSequenceBasis):
    def __init__(self, rho, cn, q_name='q', var_name='q_n', base=QQ, **kwds):
        super().__init__(
            q_name=q_name, var_name = var_name, base=base,# arguments for QFactorialBasis
            rho=rho, cn=cn, **kwds# arguments for other builders (like RootSequenceBasis) allowing multi-inheritance
        )

class QScalarBasis(QFactorialBasis, ScalarBasis):
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

        This class allows the user to compute the scaled basis given the original `q`-factorial basis `\mathcal{B}` 
        and the scaling sequence `(c_k)_k`.

        INPUT:

        * ``basis``: a `q`-factorial basis to be scaled.
        * ``scale``: a hypergeometric term (see :func:`QBasis.is_q_hypergeometric`)

        EXAMPLES::

            sage: from pseries_basis import *; from pseries_basis.qbasis.qbasis import QScalarBasis
            sage: B = QBinomialBasis(); q = B.q(); qn = B.n()
    '''
    def __init__(self, basis: QFactorialBasis = None, scale = None, **kwds):
        if not isinstance(basis, QFactorialBasis):
            raise TypeError(f"The given basis must be a QFactorialBasis. Got {basis.__class__}")

        super().__init__(
            q_name = str(basis.q()), var_name = basis.var_name, base=basis.base, # arguments for QFactorialBasis
            basis=basis, scale=scale, **kwds # arguments
        )

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
        * ``var_name``: ("q_n" by default) the name of the polynomial variable. It represent the multiplication by :func:`QPower` (i.e., `q^n`).
        * ``E``: ("E" by default) the name for the compatible shift with this basis.
    '''
    def __init__(self, b=0, q_name='q', var_name='q_n', name_E='E', base=QQ, **kwds):        
        if(not b in ZZ):
            raise b("The shift of the basis must be a natural number")
        b = ZZ(b)
        self.__b = b
        self.__E_name = name_E

        # we need to create a dummy (simpler) basis to obtain the values for `q` and `qn`
        oself = QBasis(base, universe=None, degree=True, var_name=var_name, q_name = q_name)
        qn = oself.Q(); q = oself.q()
        # removing possible repeated arguments
        kwds.pop("an",None); kwds.pop("bn",None); kwds.pop("init",None)
        super().__init__(
            an = -(q**(b+1)/(qn*(1-qn))), bn = 1/(1-qn), q_name = q_name, var_name = var_name, init = 1, base = base, # arguments for QSFactorialBasis
            **kwds # other arguments for other builders (allow multi-inheritance)
        )
        self.set_endomorphism(name_E, qn + self.Sn(), True)

    def change_base(self, base):
        return QBinomialBasis(
            self.__b,                       # the shift value does not change
            str(self.q()),                  # the name of the variable does not change
            str(self.n()),                  # the name for the multiplication by `q^n`
            self.__E_name,                  # the name for the shifts does not change
            base                            # the base is set to the given value
        )

    def is_quasi_func_triangular(self):
        return self.__b >= 0

    def shift_in(self, shift):
        return QBinomialBasis(self.__b + shift, str(self.q()), str(self.n()), self.__E_name, self.base)

    def __repr__(self):
        return f"Q-Binomial Basis [n{f' + {self.__b}' if self.__b != 0 else ''}; m]_q"

    def _latex_(self):
        upper = f"n{f' + {self.__b}' if self.__b != 0 else ''}"
        return r"\left[\begin{array}{c} " + upper + r"\\ m \end{array}\right]_q"

#######################################################################################################
### Some useful functions and sequences
#######################################################################################################
def check_q_compatibility(basis : QBasis, operator, action, bound=100):
    r'''
        Method that checks that a compatibility formula holds for some examples.

        This method takes a :class:`QBasis`, an operator compatibility (either a tuple with the 
        compatibility data or the operator that must be compatible with the basis), an actions with the 
        map for the operator and a bound and checks that the induced compatibility identity holds for 
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

def qpochhammer(a: Element, b: Element, n: int) -> Element: # TODO: fix this to allow other `q` (or maybe not)
    r'''Recursive definition of `(a;b)_n`'''
    if n < 0: return 0
    elif n == 0: return 1
    return qpochhammer(a, b, n-1) * (1 - a*b**(n-1))

def QPochhammerSequence(a: Element, b: Element) -> Sequence:
    universe = pushout(parent(a), parent(b))
    return LambdaSequence(lambda n : qpochhammer(a, b, n), universe, 1)

def qnatural(n: int, q: Element) -> Element:
    r'''Definition of the [n]_q natural numbers'''
    return (1-q**n)/(1-q)

def QNaturalSequence(q: Element) -> Sequence:
    return LambdaSequence(lambda n : qnatural(n, q), parent(q), 1)

def qfactorial(n: int, q: Element) -> Element:
    r'''Recursive definition of `[n]_q!'''
    if n < 0: return 0
    if n == 0: return 1
    return qfactorial(n-1, q) * (1-q**n)/(1-q)

def QFactorialSequence(q: Element) -> Sequence:
    return LambdaSequence(lambda n : qfactorial(n, q), parent(q), 1)

def __polyvar_log(poly, var):
    r'''Compute the `n` for ``poly == var**n` for a variable ``var`` of a polynomial ring'''
    if not poly in var.parent():
        raise ValueError(f"[polyvar_log] {poly} is not a polynomial in {var}")
    poly = var.parent()(poly)
    rems = []
    while not poly.is_constant():
        rems.append(poly%var)
        poly = poly//var
    if all(el == 0 for el in rems) and poly == 1:
        return len(rems)
    elif poly != 1:
        raise ValueError(f"[polyvar_log] The leading coefficient of {poly} was not 1")
    else:
        raise ValueError(f"{poly} was not of the form {var}^n")

__all__ = [
    "QBasis", "QSequentialBasis", "QBinomialBasis", ## type of basis
    "check_q_compatibility", "qpochhammer", "qnatural", "qfactorial", # utility functions
    "QPochhammerSequence", "QNaturalSequence", "QFactorialSequence", # special sequences
    ]
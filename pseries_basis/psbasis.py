r'''
    Sage package for Power Series Basis.

    This module introduces the basic structures in Sage for computing with *Power
    Series Basis*. We based this work in the paper :arxiv:`2202.05550`
    by M. Petkovšek, where all definitions and proofs for the algorithms here can be found.

    A Power Series basis is defined as a sequence `\{f_n\}_{n\in\mathbb{N}} \subset \mathbb{K}[[x]]`
    that form a `\mathbb{K}`-basis of the whole ring of formal power series. We distinguish
    between two basic type of basis:

    * Polynomial basis: here `f_n \in \mathbb{K}[x]` with degree equal to `n`.
    * Order basis: here `ord(f_n) = n`, meaning that `f_n = x^n g_n(x)` such that `g(0) \neq 0`.

    Any formal power series `g(x)` can be expressed in terms of the power series basis:

    .. MATH::

        g(x) = \sum_{n \in \mathbb{N}} \alpha_n f_n.

    The main aim of this work is to understand which `\mathbb{K}`-linear operators over the
    ring of formal power series are *compatible* with a power series basis, meaning that, 
    `L\cdot g(x) = 0` if and only if the sequence `\alpha_n` is P-recursive.

    EXAMPLES::

        sage: from pseries_basis import *

    This package includes no example since all the structures it offers are abstract, so they should
    never be instantiated. For particular examples and test, look to the modules :mod:`~pseries_basis.factorial.factorial_basis`
    and :mod:`~pseries_basis.factorial.product_basis`.
'''
import logging
logger = logging.getLogger(__name__)

## Sage imports
from functools import reduce
from sage.all import (ZZ, QQ, Matrix, cached_method, latex, factorial, 
                        SR, Expression, prod, hypergeometric, lcm, cartesian_product, SR, parent,
                        block_matrix, vector, ceil)
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.symbolic.operators import add_vararg, mul_vararg
from sage.structure.element import is_Matrix # pylint: disable=no-name-in-module

# ore_algebra imports
from ore_algebra.ore_algebra import OreAlgebra_generic
from ore_algebra.ore_operator import OreOperator

from pseries_basis.misc.noncom_rings import OperatorAlgebra_generic, OperatorAlgebra_element

# imports from this package
from .misc.ore import (get_double_recurrence_algebra, is_based_field, is_recurrence_algebra, is_qshift_algebra, 
                    gens_recurrence_algebra, gens_qshift_algebra, eval_ore_operator, poly_decomposition, 
                    get_rational_algebra, get_recurrence_algebra)
from .misc.sequences import LambdaSequence, Sequence, SequenceSet


class NotCompatibleError(TypeError): pass

class PSBasis(Sequence):
    r'''
        Generic (abstract) class for a power series basis.
        
        Their elements must be indexed by natural numbers and ordered by
        *degree* or *order*.
        
        This class **must never** be instantiated, but contains all the methods that will
        have a common implementation for particular basis.

        List of abstract methods:

        * :func:`~PSBasis._element`.
        * :func:`~PSBasis._functional_matrix`.
    '''
    def __init__(self, base, universe=None, degree=True, var_name=None, **_):
        self.__degree = degree
        self.__base = base
        self.__compatibility = {}
        self.__derivations = []
        self.__endomorphisms = []
        self.__var_name = "x" if var_name is None else var_name
        
        if universe == None and degree is True:
            universe = PolynomialRing(self.__base, self.__var_name)
        elif universe is None:
            universe = LambdaSequence(lambda n,k : n+k, self.base, 2, True).parent()
        super().__init__(universe, 1)

    ### Getters from the module variable as objects of the class
    def OB(self):
        r'''
            Method to get the generic base ring for rational functions in `n`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal building, do not use in general
                sage: B.OB()
                Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return get_rational_algebra('n', base=self.base)[0]

    def n(self):
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
        return get_rational_algebra('n', base=self.base)[1]

    def OS(self):
        r'''
            Method to get the generic variable :class:`~ore_algebra.OreAlgebra` for the shift 
            and inverse shift operators over the rational functions in `n`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal building, do not use in general
                sage: B.OS()
                Multivariate Ore algebra in Sn, Sni over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return get_double_recurrence_algebra("n", "Sn", rational=True, base=self.base)[0]

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
        return get_recurrence_algebra("n", "Sn", rational=True, base=self.base)[0]

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
    
    def recurrence_vars(self):
        r'''
            Method that returns all the variables involved in the recurrence operators (see :func:`OS`).

            This method returns the variables and generators necessary to fully describe the recurrence operators obtained
            through the method :func:`recurrence`. The order of the output is "inner-out", meaning the first variables are 
            the most inner variables of the structure and the last elements are the outer most variables.

            For example, if ``self.OS()`` returns the ring `\mathbb{K}(a,b)[x]\langle S, S^{-1}\rangle`, then this method will
            return `(a,b,x,S,S^{-1})`.
        '''
        base_gens = self.base.gens()
        if 1 in base_gens: base_gens = [] # no real generator
        n, Sn, Sni = self.n(), self.Sn(), self.Sni()
        return tuple([*base_gens, n, Sn, Sni])

    def is_hypergeometric(self, element):
        r'''
            Method to check if a symbolic expression is hypergeometric or not.

            This method checks whether ``element`` is a symbolic expression or a function
            with a parameter `n` that is hypergeometric. 

            This method returns ``True`` or ``False`` and the quotient (if the output is hypergeometric)
            or ``None`` otherwise.

            INPUT:

            * ``element``: the object that will be checked.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = BinomialBasis(); n = B.n()

            Rational functions in `n` are always hypergeometric::

                sage: B.is_hypergeometric(n)
                (True, (n + 1)/n)
                sage: B.is_hypergeometric(n^2)
                (True, (n^2 + 2*n + 1)/n^2)
                sage: B.is_hypergeometric(n*(n+1))
                (True, (n + 2)/n)

            But this method accepts symbolic expressions involving the factorial or the binomial
            method of Sage and recognize the type of sequence::

                sage: B.is_hypergeometric(factorial(n))
                (True, n + 1)
                sage: B.is_hypergeometric(hypergeometric([1,2,3],[4,5,6],n))
                (True, (n^2 + 5*n + 6)/(n^3 + 15*n^2 + 74*n + 120))

            We can also recognize any polynomial expression of hypergeometric terms::

                sage: B.is_hypergeometric(n+factorial(n))
                (True, (n^2 + 2*n + 1)/n)
                sage: B.is_hypergeometric(hypergeometric([1,2],[],n)*(n^2-2) + factorial(n)*(n^4-1)/(n+1))
                (True, (2*n^6 + 6*n^5 + 2*n^4 - 6*n^3 - 7*n^2 - 9*n + 2)/(n^5 - n^4 - n^3 + n^2 - 2*n + 2))

            The argument for the :sageref:`functions/sage/functions/hypergeometric` and 
            :sageref:`functions/sage/functions/other#sage.functions.other.Function_factorial`
            has to be exactly `n` or a simple shift. Otherwise this method returns ``False``::

                sage: B.is_hypergeometric(factorial(n+1))
                (True, n + 2)
                sage: B.is_hypergeometric(factorial(n^2))
                (False, None)
                sage: B.is_hypergeometric(hypergeometric([1],[2], n+2))
                (True, 1/(n + 4))
                sage: B.is_hypergeometric(hypergeometric([1],[2], n^2))
                (False, None)

            TODO: add a global class sequence for the sequences and then allow P-finite sequences
            TODO: extend this method for further hypergeometric detection (if possible)
        '''
        from _operator import pow

        # Basic case of rational functions in self.OB()
        if(element in self.OB()):
            element = self.OB()(element); n = self.n()
            return True, element(n=n+1)/element(n)

        # We assume now it is a symbolic expression
        element = SR(element)

        operator = element.operator()
        if(operator is add_vararg):
            are_hyper = [self.is_hypergeometric(el) for el in element.operands()]
            if(any(not el[0] for el in are_hyper)):
                return (False, None)
            return (True, sum([el[1] for el in are_hyper], 0))
        elif(operator is mul_vararg):
            are_hyper = [self.is_hypergeometric(el) for el in element.operands()]
            if(any(not el[0] for el in are_hyper)):
                return (False, None)
            return (True, prod([el[1] for el in are_hyper], 1))
        elif(operator is pow):
            base,exponent = element.operands()
            if(exponent in ZZ):
                is_hyper, quotient = self.is_hypergeometric(base)
                if(is_hyper):
                    return (is_hyper, quotient**ZZ(exponent))
                return (False, None)
        elif(operator is hypergeometric):
            a, b, n = element.operands()
            # casting a and b to lists
            a = a.operands(); b = b.operands()

            if(not n in self.OB()):
                return (False, None)
            n = self.OB()(n)
            if(not self.OB()(n)-self.n() in ZZ): # the index is a rational function in `n`
                return (False, None) # TODO: check if it is extensible
            quotient = prod(n+el for el in a)/prod(n+el for el in b+[1])
            try:
                return (True, self.OB()(str(quotient)))
            except: 
                return (False, None)
            
        # The operator is not a special case: we try to check by division
        n = self.n()
        quotient = element(n=n+1)/element(n=n)
        if(isinstance(quotient, Expression)):
            quotient = quotient.simplify_full()
        
        try:
            return (True, self.OB()(str(quotient)))
        except:
            return (False, None)

    def valid_factor(self, element):
        r'''
            Checks whether a rational function has poles or zeros in the positive integers.

            When we compute a scaling of a basis for the ring of formal power series, we 
            should be careful that the factor (which is a sequence `\mathbb{K}^\mathbb{N}`)
            never vanishes and it is well defined for all possible values of `n`.

            This method perform that checking for a rational function (which we can explicitly
            compute the zeros and poles). We do not need to compute the algebraic roots of the polynomial,
            simply the rational roots (which can be done with the usual Sage algorithms).

            INPUT:

            * ``element``: rational function in `n` (see :func:`OB`).

            OUTPUT:

            This method return ``True`` if the rational function has no pole nor zero on `\mathbb{N}`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = BinomialBasis(); n = B.n()
                sage: B.valid_factor(n)
                False
                sage: B.valid_factor(n+1)
                True
                sage: B.valid_factor(n+1/2)
                True
                sage: B.valid_factor(factorial(n))
                False
                sage: B.valid_factor(5)
                True
                sage: B.valid_factor((n+1)*(n+2))
                True
                sage: B.valid_factor((n+1)/n)
                False
                sage: B.valid_factor((n+1)/(n+2))
                True

            This allow to check if a hypergeometric element is valid as a scalar product (see method :func:`is_hypergeometric`)::

                sage: hyper, quotient = B.is_hypergeometric(factorial(n))
                sage: B.valid_factor(quotient)
                True
                sage: hyper, quotient = B.is_hypergeometric(hypergeometric([2],[3],n))
                sage: B.valid_factor(quotient)
                True
                sage: hyper, quotient = B.is_hypergeometric(hypergeometric([2,6,8,4],[3,2,4,23],n))
                sage: quotient
                (n^2 + 14*n + 48)/(n^3 + 27*n^2 + 95*n + 69)
                sage: B.valid_factor(quotient)
                True
                sage: hyper, quotient = B.is_hypergeometric(hypergeometric([-2,6],[],n))
                sage: B.valid_factor(quotient)
                False
        '''
        if isinstance(element, Sequence) and element.allow_sym:
            element = element(self.n())
        if(not element in self.OB()):
            return False
        element = self.OB()(element)

        ## We check the denominator never vanishes on positive integers
        if(any((m >= 0 and m in ZZ) for m in [root[0] for root in element.denominator().roots()])):
            return False

        ## We check the numerator never vanishes on the positive integers
        if(any((m >= 0 and m in ZZ) for m in [root[0] for root in element.numerator().roots()])):
            return False
            
        return True

    def extended_quo_rem(self, n, k):
        r'''
            Extended version of quo_rem that works also for for rational functions.

            This method extends the functionality of quo_rem for rational functions and takes
            care of the different types the input may have.

            This method returns a tuple `(r,s)` such that `n = rk + s` and `s < k`.

            INPUT:

            * ``n``: value to compute quo_rem
            * ``k``: integer number for computing the quo_rem

            TODO: add examples
        '''
        ## Checking the input
        if(not k in ZZ):
            raise TypeError("The divisor must be an integer")
        k = ZZ(k)
        
        if(n in ZZ):
            return ZZ(n).quo_rem(k)
        
        elif(n in self.OB()):
            if(n.denominator() != 1):
                raise TypeError("The value of `n` can not be quo_rem by %d" %k)
            n = n.numerator().change_ring(ZZ); var = self.n()
            q = sum(n[i]//k * var**i for i in range(n.degree()+1))
            r = sum(n[i]%k * var**i for i in range(n.degree()+1))

            if(not r in ZZ):
                raise ValueError("The quo_rem procedure fail to get a valid remainder")
            r = ZZ(r)
            if(r < 0): # in case Sage does something weird and return a negative remainder
                q -= 1
                r += k
            return (q,r)
        
        raise NotImplementedError("quo_rem not implemented for %s" %type(n))

    ### BASIC METHODS
    @property
    def base(self):
        return self.__base

    @property
    def var_name(self):
        return self.__var_name

    def change_base(self, base):
        r'''
            Method to change the base ring to consider the coefficients.

            This method allows to change the base ring (see input ``base`` in :class:`PSBasis` for further information).
        '''
        raise NotImplementedError(f"Method `change_base` not implemented for {self.__class__}")

    def by_degree(self):
        r'''
            Getter for the type of ordering of the basis.
            
            Return ``True`` if the `n`-th element of the basis is a polynomial of degree `n`.
        '''
        return self.__degree
    
    def by_order(self):
        r'''
            Getter for the type of ordering of the basis.
            
            Return ``True`` if the `n`-th element of the basis is a power series of order `n`.
        '''
        return (not self.__degree)

    @property
    def functional_seq(self) -> Sequence:
        r'''
            Method to get the functional sequence of the basis.

            A :class:`PSBasis` can be seen as a sequence of functions or polynomials. However, these
            functions and polynomials are sequences by themselves. In fact, a :class:`PSBasis` is 
            a basis of the ring of formal power series which, at the same time, is a basis of the 
            ring of sequences. 

            However, the relation between the sequences and the formal power series is not unique:

            * We can consider the formal power series `f(x) = \sum_n a_n x^n`, then we have the 
              natural (also called functional) sequence `(a_n)_n`.
            * We can consider formal power series as functions `f: \mathbb{K} \rightarrow \mathbb{K},
              and (if convergent) we can define the (evaluation) sequence `(f(n))_n`.

            This method returns a bi-indexed sequence that allows to obtain the functional sequences
            of this basis.
        '''
        raise NotImplementedError("Method functional_seq must be implemented in each subclass of PSBasis")

    def functional_matrix(self, nrows, ncols=None):
        r'''
            Method to get a matrix representation of the basis.

            This method returns a matrix `\tilde{M} = (m_{i,j})` with ``nrows`` rows and
            ``ncols`` columns such that `m_{i,j} = [x^j]f_i(x)`, where `f_i(x)` is 
            the `i`-th element of ``self``.

            This is the upper-left part of the matrix `M` that represent, by rows,
            the elements of this basis in terms of the canonical basis of the formal
            power series ring (`\{1,x,x^2,\ldots\}`). Hence, if we have an element 
            `y(x) \in \mathbb{K}[[x]]` with:

            .. MATH::

                y(x) = \sum_{n\geq 0} y_n x^n = \sum_{n\geq 0} c_n f_n(x),

            then the infinite vectors `\mathbf{y}` and `\mathbf{c}` satisfies:

            .. MATH::

                \mathbf{y} = \mathbf{c} M

            INPUT:

            * ``nrows``: number of rows of the final matrix
            * ``ncols``: number of columns of the final matrix. If ``None`` is given, we
              will automatically return the square matrix with size given by ``nrows``.
        '''
        ## Checking the arguments
        if(not ((nrows in ZZ) and nrows > 0)):
            raise ValueError("The number of rows must be a positive integer")
        if(ncols is None):
            ncols = nrows
        elif(not ((ncols in ZZ) and ncols > 0)):
                raise ValueError("The number of columns must be a positive integer")

        return Matrix([[self.functional_seq(i,j) for j in range(ncols)] for i in range(nrows)])

    def is_quasi_func_triangular(self):
        r'''
            Method to check whether a basis is quasi-triangular or not as a functional basis.

            A basis `\mathcal{B} = \{f_n(x)\}_n` is a functional *quasi-triangular* if its 
            functional matrix representation `M = \left(m_{n,k}\right)_{n,k \geq 0}` (see 
            :func:`functional_matrix`) is *quasi-upper triangular*, i.e.,
            there is a strictly monotonic function `I: \mathbb{N} \rightarrow \mathbb{N}` such that

            * For all `k \in \mathbb{N}`, and `m > I(k)`, `m_{n,k} =0`, i.e., `x^k` divides `f_n(x)`.
            * For all `k \in \mathbb{N}`, `m_{I(k),k} \neq 0`, i.e., `x^k` does not divide `f_{I(k)}(x)`.

            This property will allow to transform the initial conditions from the canonical basis
            of formal power series (`\{1,x,x^2,\ldots\}`) to the initial conditions of the expansion
            over ``self``.
        '''
        return False

    def functional_to_self(self, sequence, size):
        r'''
            Matrix to convert a sequence from the canonical basis of `\mathbb{K}[[x]]` to ``self``.

            Let `y(x) = \sum_{n\geq 0} y_n x^n` be a formal power series. Since ``self`` represents another 
            basis of the formal power series ring, then `y(x)` can be expressed in terms of the elements
            of ``self``, i.e., `y(x) = \sum_{n\geq 0} c_n f_n(x)` where `f_n(x)` is the `n`-th term of this basis.

            This method allows to obtain the first terms of this expansion (as many as given in ``size``) 
            for the formal power series `y(x)` where the first elements are given by ``sequence``. 

            Using the basis matrix `M` (see :func:`functional_matrix`), this computation is straightforward, since

            .. MATH::

                y = c M.

            INPUT:

            * ``sequence``: indexable object with *enough* information to compute the result, representing the first
              terms of the sequence `(y_0, y_1, \ldots)`.
            * ``size``: number of elements of the sequence `(c_0, c_1,\ldots)` computed in this method.

            OUTPUT:

            The tuple `(c_0, \ldots, c_{k})` where `k` is given by ``size-1``.

            TODO: add Examples and tests
        '''
        if not (self.is_quasi_func_triangular()): 
            raise ValueError("We require 'functional_matrix' to be quasi-upper triangular.")

        M = self.functional_matrix(size)
        if M.is_triangular("upper"):
            return tuple([el for el in M.solve_left(vector(sequence[:size]))])
        raise NotImplementedError("The pure quasi-triangular case not implemented yet.")

    @property
    def evaluation_seq(self) -> Sequence:
        r'''
            Method to get the functional sequence of the basis.

            A :class:`PSBasis` can be seen as a sequence of functions or polynomials. However, these
            functions and polynomials are sequences by themselves. In fact, a :class:`PSBasis` is 
            a basis of the ring of formal power series which, at the same time, is a basis of the 
            ring of sequences. 

            However, the relation between the sequences and the formal power series is not unique:

            * We can consider the formal power series `f(x) = \sum_n a_n x^n`, then we have the 
              natural (also called functional) sequence `(a_n)_n`.
            * We can consider formal power series as functions `f: \mathbb{K} \rightarrow \mathbb{K},
              and (if convergent) we can define the (evaluation) sequence `(f(n))_n`.

            This method returns a bi-indexed sequence that allows to obtain the evaluation sequences
            of this basis.
        '''
        raise NotImplementedError("Method evaluation_seq must be implemented in each subclass of PSBasis")

    def evaluation_matrix(self, nrows, ncols=None):
        r'''
            Method to get a matrix representation of the basis.

            This method returns a matrix `\tilde{M} = (m_{i,j})` with ``nrows`` rows and
            ``ncols`` columns such that `m_{i,j} = f_i(j)`, where `f_i(x)` is 
            the `i`-th element of ``self``.

            This is the upper-left part of the matrix `M` that represent, by rows,
            the images of this basis in the natural numbers. This is specially useful when
            considering recurrences since, if we have an element 
            `y(x) \in \mathbb{K}[[x]]` with `y(n) = y_n` and:

            .. MATH::

                y(x) = \sum_{n\geq 0} c_n f_n(x),

            then the infinite vectors `\mathbf{y}` and `\mathbf{c}` satisfies:

            .. MATH::

                \mathbf{y} = \mathbf{c} M

            INPUT:

            * ``nrows``: number of rows of the final matrix
            * ``ncols``: number of columns of the final matrix. If ``None`` is given, we
              will automatically return the square matrix with size given by ``nrows``.
        '''
        ## Checking the arguments
        if(not ((nrows in ZZ) and nrows > 0)):
            raise ValueError("The number of rows must be a positive integer")
        if(ncols is None):
            ncols = nrows
        elif(not ((ncols in ZZ) and ncols > 0)):
                raise ValueError("The number of columns must be a positive integer")

        return Matrix([[self.evaluation_seq(i,j) for j in range(ncols)] for i in range(nrows)])

    def is_quasi_eval_triangular(self):
        r'''
            Method to check whether a basis is quasi-triangular or not as an evaluation basis.

            A basis `\mathcal{B} = \{f_n(x)\}_n` is an evaluation *quasi-triangular* if its 
            evaluation matrix representation `M = \left(m_{n,k}\right)_{n,k \geq 0}` (see 
            :func:`evaluation_matrix`) is *quasi-upper triangular*, i.e.,
            there is a strictly monotonic function `I: \mathbb{N} \rightarrow \mathbb{N}` such that

            * For all `k \in \mathbb{N}`, and `m > I(k)`, `m_{n,k} =0`, i.e., `k` is a zero of `f_n(x)`.
            * For all `k \in \mathbb{N}`, `m_{I(k),k} \neq 0`, i.e., `k` is not a zero of `f_{I(k)}(x)`.

            This property will allow to transform the initial conditions from a recurrence defined 
            by the evaluation at the natural numbers to the initial conditions of the expansion
            over ``self``.

            In :arxiv:`2202.05550`, this concept is equivalent to the definition of a *quasi-triangular* basis
            in the case of factorial bases.
        '''
        return False

    def evaluation_to_self(self, sequence, size):
        r'''
            Matrix to convert a sequence from the evaluation basis of `\mathbb{K}[[x]]` to ``self``.

            Let `y(x) = \sum_{n\geq 0} c_n f_n(x)` be a formal power series where `f_n(x)` is the `n`-th term of this basis.
            If well defined, the values `y_n = y(n)` defined a new sequence of numbers.

            This method allows to obtain the first terms of the `\mathbf{c}` expansion (as many as given in ``size``) 
            for the formal power series `y(x)` where the first evaluations `y_n` are given by ``sequence``. 

            Using the evaluation matrix `M` (see :func:`evaluation_matrix`), this computation is straightforward, since

            .. MATH::

                y = c M.

            INPUT:

            * ``sequence``: indexable object with *enough* information to compute the result, representing the first
              terms of the sequence `(y_0, y_1, \ldots)`.
            * ``size``: number of elements of the sequence `(c_0, c_1,\ldots)` computed in this method.

            OUTPUT:

            The tuple `(c_0, \ldots, c_{k})` where `k` is given by ``size-1``.

            TODO: add Examples and tests
        '''
        if not (self.is_quasi_eval_triangular()): 
            raise ValueError("We require 'evaluation_matrix' to be quasi-upper triangular.")

        M = self.evaluation_matrix(size)
        if M.is_triangular("upper"):
            return tuple([el for el in M.solve_left(vector(sequence[:size]))])
        raise NotImplementedError("The pure quasi-triangular case not implemented yet.")

    ### AUXILIARY METHODS
    def simplify_operator(self,operator):
        r'''
            Method to reduce operators with ``Sn`` and ``Sni``.

            The operators we handle will have two shifts in the variable `n`: the direct shift (`\sigma: n \mapsto n+1`)
            and the inverse shift (`\sigma^{-1}: n \mapsto n-1`). These two shifts are represented in our system with the 
            operators ``Sn`` and ``Sni`` respectively.

            However, the computations with the package ``ore_algebra`` do not take care automatically of the obvious cancellation
            between these two operators: `\sigma \sigma^{-1} = \sigma^{-1}\sigma = id`. This method performs this cancellation
            in all terms that have the two operators involved and returns a reduced version of the input.

            INPUT:

            * ``operator``: an operator involving ``Sn`` and ``Sni``.

            OUTPUT:
            
            A reduced but equivalent version of ``operator`` such that the monomials involved in the reduced version only have
            ``Sn`` or ``Sni``, but never mixed. 

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # illegal build just for examples
                sage: Sn = B.Sn(); Sni = B.Sni()
                sage: Sn*Sni
                Sn*Sni
                sage: Sni*Sn
                Sn*Sni
                sage: B.simplify_operator(Sn*Sni)
                1
                sage: B.simplify_operator(Sni*Sn)
                1
                sage: B.simplify_operator(Sni*Sn^2 - 3*Sni^2*Sn^3 + Sn)
                -Sn
        '''
        if(is_Matrix(operator)):
            base = operator.parent().base(); n = operator.nrows()
            return Matrix(base, [[self.simplify_operator(operator.coefficient((i,j))) for j in range(n)] for i in range(n)])
        
        return self._simplify_operator(operator)

    def _simplify_operator(self, operator):
        r'''
            Method that actually simplifies the operator. This removes inverses operators and ensures all
            the variables appear in desirable order.
        '''
        if isinstance(self.OS(), OreAlgebra_generic) and operator in self.OS():
            operator = self.OS()(str(operator))
            Sn = self.Sn(); Sni = self.Sni()

            poly = operator.polynomial()
            monomials = poly.monomials()
            coefficients = poly.coefficients()
            result = operator.parent().zero()

            for i in range(len(monomials)):
                d1,d2 = monomials[i].degrees()
                if(d1 > d2):
                    result += coefficients[i]*Sn**(d1-d2)
                elif(d2 > d1):
                    result += coefficients[i]*Sni**(d2-d1)
                else:
                    result += coefficients[i]
            return result
        elif isinstance(self.OS(), OperatorAlgebra_generic) and operator in self.OS():
            return operator.canonical()
        else:
            raise NotImplementedError(f"Impossible to simplify {operator} (type={operator.__class__})")

    def remove_Sni(self, operator):
        r'''
            Method to remove ``Sni`` from an operator. 

            This method allows to compute an equivalent operator but without inverse shifts. This
            can be helpful to compute a holonomic operator and apply methods from the package
            :mod:`ore_algebra` to manipulate it.

            We are usually interested in sequences such that when we apply an operator 
            `L \in \mathbb{K}(n)[\sigma,\sigma^{-1}]` we obtain zero. In this sense, we can always
            find an operator `\tilde{L} \in \mathbb{K}(n)[\sigma]` that also annihilates the same object.

            This method transform an operator with both direct and inverse shift to another operator
            only with direct shifts such that if the original operator annihilates an object, then
            the transformed operator also annihilates it.

            This elimination is the multiplication by ``Sn`` to the highest power of the simplified form
            of the input. This cancels all the appearances of ``Sni`` and only ``Sn`` remains. Since this is
            a left multiplication, the annihilator space only increases, hence obtaining the desired property.

            INPUT:

            * ``operator``: and operators involving ``Sn`` and ``Sni`` (i.e, in the ring returned by
              the method :func:`~PSBasis.OS`)

            OUTPUT:

            An operator that annihilates all the objects annihilated by ``operator`` that belong to the ring
            returned by :func:`~PSBasis.OSS`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = PSBasis(QQ) # do not do this in your code
                sage: Sn = B.Sn(); Sni = B.Sni()
                sage: B.remove_Sni(Sni)
                1
                sage: B.remove_Sni(Sni + 2 + Sn)
                Sn^2 + 2*Sn + 1
        '''
        Sni = self.Sni(); Sn = self.Sn()
        if(is_Matrix(operator)):
            d = max(max(el.degree(self.Sni()) for el in row) for row in operator)
            return Matrix(self.OSS(), [[self.simplify_operator((Sn**d)*el) for el in row] for row in operator])

        d = operator.degree(Sni)
        return self.OSS()(self.simplify_operator((Sn**d)*operator))
    
    ### COMPATIBILITY RELATED METHODS
    def _compatibility_from_recurrence(self, recurrence):
        r'''
            Method to obtain the compatibility condition from a recurrence equivalent.

            This method allows the user (for compatibilities in 1 section) to provide the 
            recurrence equation associated directly. This method can be overridden if the meaning
            of the recurrences involved changes.

            INPUT:

            * ``recurrence``: an element in ``self.OS()`` simplified. 

            OUTPUT:

            A tuple `(A,B,m,\alpha_{i,j,k})` with the compatibility condition from the recurrence
            (see the output of :func:`compatibility` for further information).
        '''
        Sn = self.Sn(); Sni = self.Sni(); n = self.n()
        A = recurrence.degree(Sn); B = recurrence.degree(Sni); recurrence = recurrence.polynomial()
        alpha = ([self.OB()(recurrence.coefficient({Sn:i}))(n=n-i) for i in range(A, 0, -1)] + 
                [self.OB()(recurrence.constant_coefficient())] + 
                [self.OB()(recurrence.coefficient({Sni:i}))(n=n+i) for i in range(1, B+1)])
        return (ZZ(A), ZZ(B), ZZ(1), lambda _, j, k: alpha[j+A](n=k))

    def set_compatibility(self, name, trans, sub=False, type=None):
        r'''
            Method to set a new compatibility operator.

            This method sets a new compatibility condition for an operator given 
            by ``name``. The compatibility condition must be given as a tuple
            `(A, B, m, \alpha_{i,j,k})` where `A` is the lower bound for the compatibility,
            `B` is the upper bound for the compatibility and `m` is the number of sections
            for the compatibility. In this way, we have tht the operator `L` defined by ``name``
            satisfies:

            .. MATH::

                L \cdot b_{km+r} = \sum_{i=-A}^B \alpha_{r, i, k} b_{km+r+j}

            See :arxiv:`2202.05550` for further information about the
            definition of a compatible operator.
            
            INPUT:
            
            * ``name``: the operator we want to set the compatibility. It can be the
              name for any generator in the *ore_algebra* package or the generator
              itself.
            * ``trans``: a tuple ``(A, B, m, alpha)`` where ``alpha`` must be a function with 
              three parameters:

                  * ``i``: a positive integer smaller than `m`.
                  * ``j``: an integer between `-A` and `B`.
                  * ``k``: an element of :func:`OB`.

              This parameter can also be an operator is :func:`OS`. Then the compatibility
              is of 1 section and we can compute explicitly the values of `A`, `B` and the
              `\alpha_{i,j,k}`.
            * ``sub`` (optional): if set to ``True``, the compatibility rule for ``name``
              will be updated even if the operator was already compatible.
            * ``type`` (optional): if set to ``"endo"`` or ``"der"``, we assume the operator given in 
              ``name`` is either a endomorphism or derivation.
        '''
        name = str(name)
        
        if(name in self.__compatibility and (not sub)):
            print(f"The operator {name} is already compatible with this basis -- no changes are done")
            return
        
        if(isinstance(trans, tuple)):
            if(len(trans) != 4):
                raise ValueError("The operator given has not the appropriate format: not a triplet")
            A, B, m, alpha = trans
            if((not A in ZZ) or A < 0):
                raise ValueError(f"The lower bound parameter is not valid: {A}")
            if((not B in ZZ) or B < 0):
                raise ValueError(f"The upper bound parameter is not valid: {B}")
            if((not m in ZZ) or m < 1):
                raise ValueError(f"The number of sections is not valid: {m}")

            # TODO: how to check the alpha?
            self.__compatibility[name] = (ZZ(A),ZZ(B),ZZ(m),alpha)
        elif(trans in self.OS()):
            self.__compatibility[name] = self._compatibility_from_recurrence(self.simplify_operator(trans))

        if type in ("endo", "der"):
            if name in self.__derivations:
                self.__derivations.remove(name)
            elif name in self.__endomorphisms:
                self.__endomorphisms.remove(name)

            if type == "endo": self.__endomorphisms.append(name)
            if type == "der": self.__derivations.append(name)

    def set_endomorphism(self, name, trans, sub=False):
        r'''
            Method to set a new compatibility operator for an endomorphism.

            This method sets a new compatibility condition for an operator given 
            by ``name`` (see :func:`set_compatibility`). We assume the name 
            is for an endomorphism, i.e., let `\varphi` be the operator represented by ``name``. 
            Then for all `a,b in \mathbb{K}[[x]]`, it holds

            .. MATH::

                \varphi(ab) = \varphi(a) \varphi(b)
            
            INPUT: see input in :func:`set_compatibility`.
        '''
        self.set_compatibility(name, trans, sub, "endo")

    def set_derivation(self, name, trans, sub=False):
        r'''
            Method to set a new compatibility operator for a derivation.

            This method sets a new compatibility condition for an operator given 
            by ``name`` (see :func:`set_compatibility`). We assume the name 
            is for a derivation, i.e., let `\partial` be the operator represented by ``name``. 
            Then for all `a,b in \mathbb{K}[[x]]`, it holds

            .. MATH::

                \partial(ab) = \partial(a)b +  a\partial(b)
            
            INPUT: see input in :func:`set_compatibility`.
        '''
        self.set_compatibility(name, trans, sub, "der")

    @cached_method
    def get_lower_bound(self, operator):
        r'''
            Method to get the lower bound compatibility for an operator.
            
            This method returns the minimal index for the compatibility property
            for a particular operator. In the notation of the paper
            :arxiv:`2202.05550`, for a `(A,B)`-compatible operator,
            this lower bound corresponds to the value of `A`.
            
            INPUT:

            * ``operator``: the operator we want to check. It can be the
              name for any generator in the ``ore_algebra`` package or the generator
              itself.
                
            WARNING:
            
            * The case when the compatibility rule is a matrix is not implemented.
        '''
        ## Case of the name
        compatibility = self.recurrence(operator)
        
        if(is_Matrix(compatibility)):
            return self.compatibility(operator)[0]
            
        return compatibility.degree(self.Sn())
    
    @cached_method
    def get_upper_bound(self, operator):
        r'''
            Method to get the upper bound compatibility for an operator.
            
            This method returns the maximal index for the compatibility property
            for a particular operator. In the notation of the paper
            :arxiv:`2202.05550`, for a `(A,B)`-compatible operator,
            this lower bound corresponds to the value of `B`.
            
            INPUT:

            * ``operator``: the operator we want to check. It can be the
              name for any generator in the ``ore_algebra`` package or the generator
              itself.
                
            WARNING:
                
            * The case when the compatibility rule is a matrix is not implemented.
        '''
        compatibility = self.recurrence(operator)
        
        if(is_Matrix(compatibility)):
            return self.compatibility(operator)[0]
            
        return compatibility.degree(self.Sni())
        
    def compatible_operators(self):
        r'''
            Method that returns a list with the compatible operators stored in the dictionary.

            This method allows the user to know the names of the basic compatible operators with this 
            basis. Any polynomial built on these operators will be valid for the method :func:`recurrence`.

            OUTPUT:

            Return the key set of the dictionary of compatibilities. This set will be composed of the names of 
            the compatible operators with ``self``.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: BinomialBasis().compatible_operators()
                dict_keys(['x', 'Et', 'E'])
                sage: PowerBasis().compatible_operators()
                dict_keys(['x', 'Id', 'Dx'])
                sage: HermiteBasis().compatible_operators()
                dict_keys(['x', 'Dx'])
                sage: B = FallingBasis(1,2,3)
                sage: B.compatible_operators()
                dict_keys(['x', 'E3'])
                
            This output gets updated when we add new compatibilities
                
                sage: B.set_compatibility('s', 1)
                sage: B.compatible_operators()
                dict_keys(['x', 'E3', 's'])
        '''
        return self.__compatibility.keys()

    def compatible_endomorphisms(self):
        r'''
            Method to get the registered endomorphisms compatible with ``self``.
        '''
        return self.__endomorphisms

    def compatible_derivations(self):
        r'''
            Method to get the registered derivations compatible with ``self``.
        '''
        return self.__derivations

    def has_compatibility(self, operator):
        r'''
            Method to know if an operator has compatibility with this basis.

            This method checks whether the operator given has a compatibility or not.

            INPUT:

            * ``operator``: the operator we want to know if it is compatible or not.
              It can be a string or an object that will be transformed into a string
              to check if the compatibility is included.

            OUTPUT:

            ``True`` if the given operator is compatible and ``False`` otherwise.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: BinomialBasis().has_compatibility('x')
                True
                sage: BinomialBasis().has_compatibility('E')
                True
                sage: BinomialBasis().has_compatibility('Id')
                False
                sage: PowerBasis().has_compatibility('Id')
                True
                sage: HermiteBasis().has_compatibility('Dx')
                True
                sage: B = FallingBasis(1,2,3)
                sage: B.has_compatibility('E3')
                True
                
            This output gets updated when we add new compatibilities::
                
                sage: B.has_compatibility('s')
                False
                sage: B.set_compatibility('s', 1)
                sage: B.has_compatibility('s')
                True
        '''
        return str(operator) in self.__compatibility

    def compatibility_type(self, operator):
        r'''
            Method to know if an operator has compatibility of a specific type.

            This method checks whether the operator given has a compatibility of
            endomorphism type or derivation type.

            INPUT:

            * ``operator``: the operator we want to know if it is compatible or not.
              It can be a string or an object that will be transformed into a string
              to check if the compatibility is included.

            OUTPUT:

            ``"endo"`` if the given operator is considered an endomorphism, ``"der"``
            if it is considered a derivation and ``None`` otherwise.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: BinomialBasis().compatibility_type('x')
                sage: BinomialBasis().compatibility_type('E')
                'endo'
                sage: BinomialBasis().compatibility_type('Id')
                Traceback (most recent call last):
                ...
                NotCompatibleError: operator Id is not compatible with this basis
                sage: PowerBasis().compatibility_type('Id')
                'endo'
                sage: HermiteBasis().compatibility_type('Dx')
                'der'
                sage: B = FallingBasis(1,2,3)
                sage: B.compatibility_type('E3')
                'endo'
                
            This output gets updated when we add new compatibilities::
                
                sage: B.compatibility_type('s')
                Traceback (most recent call last):
                ...
                NotCompatibleError: operator s is not compatible with this basis
                sage: B.set_compatibility('s', 1)
                sage: B.compatibility_type('s')
                sage: B.set_endomorphism('t', 10)
                sage: B.compatibility_type('t')
                'endo'
        '''
        operator = str(operator)
        if not self.has_compatibility(operator):
            raise NotCompatibleError(f"operator {operator} is not compatible with this basis")
        if operator in self.__derivations:
            return "der"
        elif operator in self.__endomorphisms:
            return "endo"
        else:
            return None
        
    def compatibility(self, operator):
        r'''
            Method to get the compatibility condition for an operator.

            This method returns the tuple `(A, B, m, \alpha_{i,j,k})` that defines
            the compatibility condition for the operator `L` defined by ``operator``.
            this compatibility has to be stored already (see method :func:`set_compatibility`).

            INPUT:

            * ``operator``: string or a polynomial (either a proper polynomial or an operator in an *ore_algebra*)
              that is compatible with ``self``. If it is not a string, we cast it.

            OUTPUT:

            A compatibility tuple `(A, B, m, \alpha_{i,j}(k))` such that, for all `n = km+r` it holds:

            .. MATH::

                `L \cdot b_n = \sum_{j=-A, B} \alpha_{r,j}(k) b_{n+j}`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = BinomialBasis(); n = B.n()
                sage: a,b,m,alpha = B.compatibility(x)
                sage: a,b,m
                (0, 1, 1)
                sage: alpha(0,0,n), alpha(0,1,n)
                (n, n + 1)
                sage: a,b,m,alpha = B.compatibility(x^2)
                sage: a,b,m
                (0, 2, 1)
                sage: alpha(0,0,n), alpha(0,1,n), alpha(0,2,n)
                (n^2, 2*n^2 + 3*n + 1, n^2 + 3*n + 2)

            The method :func:`~pseries_basis.psbasis.check_compatibility` can check that these tuples are
            correct for the first terms of the basis::

                sage: x = B.universe.gens()[0]
                sage: check_compatibility(B, B.compatibility(2*x^2 + 3), lambda p :(2*x^2 + 3)*p)
                True

            The Binomial basis is also compatible with the shift operator `E: x \mapsto x + 1`. We can 
            also get the compatibility of that operator by name::

                sage: a,b,m,alpha = B.compatibility('E')
                sage: a,b,m
                (1, 0, 1)
                sage: alpha(0,-1,n), alpha(0,0,n)
                (1, 1)

            But we can also use any operator in the :class:`OreAlgebra` representing the operators
            generated by `E` and `x`::

                sage: from ore_algebra import OreAlgebra
                sage: R = QQ[x]; OE.<E> = OreAlgebra(R, ('E', lambda p : p(x=x+1), lambda p : 0))
                sage: a,b,m,alpha = B.compatibility(E)
                sage: (a,b,m) == (1,0,1)
                True
                sage: alpha(0,-1,n), alpha(0,0,n)
                (1, 1)
                sage: a,b,m,alpha = B.compatibility(x*E + x^2 + 3)
                sage: a,b,m
                (1, 2, 1)
                sage: alpha(0,-1,n), alpha(0,0,n), alpha(0,1,n), alpha(0,2,n)
                (n - 1, n^2 + 2*n + 3, 2*n^2 + 4*n + 2, n^2 + 3*n + 2)
                sage: check_compatibility(B, x*E + x^2 + 3, lambda p :x*p(x=x+1)+(x^2+3)*p)
                True

            This method also allows to get compatibility in different sections::

                sage: P = ProductBasis([B,B])
                sage: a,b,m,alpha = P.compatibility('E')
                sage: a,b,m
                (2, 0, 2)
                sage: P.compatibility_matrix('E')[-1]
                [                1                 2                 1]
                [        n/(n + 1) (2*n + 1)/(n + 1)                 1]
                sage: a,b,m,alpha = P.compatibility(3)
                sage: a,b,m
                (0, 0, 1)
                sage: alpha(0,0,n)
                3
                sage: a,b,m,alpha = P.compatibility(x*E + x^2 + 3)
                sage: a,b,m
                (2, 2, 2)
                sage: P.compatibility_matrix(x*E + x^2 + 3)[-1]
                [              n - 1             3*n - 2       n^2 + 3*n + 3     2*n^2 + 3*n + 1       n^2 + 2*n + 1]
                [  (n^2 - n)/(n + 1) (3*n^2 + n)/(n + 1)       n^2 + 3*n + 4     2*n^2 + 4*n + 2       n^2 + 3*n + 2]
                sage: check_compatibility(B, x*E + x^2 + 3, lambda p :x*p(x=x+1)+(x^2+3)*p, bound=50)
                True

        '''
        if(not str(operator) in self.__compatibility):
            if(operator in self.OB().base_ring()):
                self.__compatibility[str(operator)] = (0,0,1,lambda i,j,k : self.OB()(operator) if (i==j and i==0) else self.OB().zero())
            elif(not type(operator) == str):
                if(parent(operator) is SR):
                    if(any(not operator.is_polynomial(v) for v in operator.variables())):
                        raise NotCompatibleError("The symbolic expression %s can not be casted into a polynomial" %operator)
                    operator = operator.polynomial(self.OB().base_ring())
                elif(isinstance(operator, OreOperator)):
                    operator = operator.polynomial()
                
                ## At this point, operator should be a polynomial, which have the flattening morphism
                try: 
                    operator = operator.parent().flattening_morphism()(operator) # case of iterated polynomial rings
                except AttributeError: 
                    raise NotCompatibleError("The input %s is not a polynomial" %operator)

                # now the coefficients are constants
                coeffs = operator.coefficients()
                mons = operator.monomials()

                ## NOT UNICITY IN SAGE FOR UNIVARIATE POLYNOMIALS
                ## The order of monomials and coefficients in univariate polynomials are different. That is why
                ## we need to consider that special case and treat it apart:
                from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
                if(is_PolynomialRing(operator.parent())):
                    mons.reverse() # just inverting the order of one of the list is enough

                if(len(mons) == 1): # monomial case
                    m = mons[0]; c = coeffs[0]
                    g = [g for g in operator.parent().gens() if g in operator.variables()]
                    if(len(g) == 0): # the monomial 1
                        comps = [(0,0,1,lambda i,j,k : 1)]
                    elif(len(g) == 1): # monomial with 1 variable
                        d = operator.degree()
                        if(d == 1 and c == 1): # no compatibility found
                            raise NotCompatibleError("The input %s is not compatible with this basis" %operator)
                        comps = d*[self.compatibility(g[0])]
                    else: # monomial with several variables 
                        comps = [self.compatibility(v**m.degree(v)) for v in g] 

                    A,B,m,alphas = reduce(lambda comp1, comp2 : self.__prod2_case(comp1, comp2), comps[::-1])
                    self.__compatibility[str(operator)] = (A,B,m,lambda i,j,k : c*alphas(i,j,k))
                else:
                    comps = [self.compatibility(m) for m in mons]
                    t = lcm(comps[i][2] for i in range(len(mons)))
                    comps = [self.compatibility_sections(m, t) for m in mons]
                    
                    A = max(comps[i][0] for i in range(len(mons)))
                    B = max(comps[i][1] for i in range(len(mons)))
                    def __sum_case(i,j,k):
                        return sum([coeffs[l]*comps[l][3](i,j,k) if (j >= -comps[l][0] and j <= comps[l][1]) else 0 for l in range(len(mons))])
                    self.__compatibility[str(operator)] = (A,B,t,__sum_case)
            else:
                raise NotCompatibleError("The operator %s is not compatible with %s" %(operator, self))
        return self.__compatibility[str(operator)]

    def __prod2_case(self, comp1, comp2):
        A1, B1, t1, alphas = comp1 # last one
        A2, B2, t2, betas = comp2 # second last one
        A = (A1+A2); B = (B1+B2); m = lcm(t1,t2)
        m1 = m//t1; m2 = m//t2
        def __aux_prod2_case(r,l,k):
            r0,r1 = self.extended_quo_rem(r, t1)
            r2,r3 = list(zip(*[self.extended_quo_rem(r+i, t2) for i in range(-A1,B1+1)]))

            return sum(
                alphas(r1,i,k*m1+r0)*betas(r3[i+A1],j,k*m2+r2[i+A1]) 
                for (i,j) in cartesian_product([range(-A1,B1+1),range(-A2,B2+1)])
                if(i+j == l))
        return (A,B,m,__aux_prod2_case)

    def compatibility_matrix(self, operator, sections=None):
        r'''
            Method to get the compatibility condition in matrix form

            This method is equivalent to the method :func:`compatibility`
            but instead of returning the coefficients `\alpha_{i,j}(n)` in 
            a method format, it plugs the value `n` and builds a matrix
            of size `i\times j`.

            This method requires that the compatibility condition can be written
            with a generic formula. See method :func:`compatibility` for a further
            description on compatibilities conditions and tests.

            INPUT:

            * ``operator``: operator `L` we want to compute the compatibility matrix.
            * ``sections``: optional argument (``None`` by default). If different than
              ``None``, we force that the compatibility is given in a particular number of sections.

            OUTPUT:

            A tuple `(A,B,M)` where `A` and `B` are the compatibility bounds (see output of 
            :func:`compatibility`) and `M` is a matrix of size `(m\times(A+B+1))` such that
            for all `n = km + r`:

            .. MATH::
            
            L\cdot P_n(x) = \sum_{i=-A}^B m_{r,A+i}(k)P_{n+i}

            TODO: add examples
        '''
        if(sections is None):
            a,b,m,alpha = self.compatibility(operator)
        else:
            a,b,m,alpha = self.compatibility_sections(operator, sections)
            
        return (a,b,Matrix([[alpha(i,j,self.n()) for j in range(-a,b+1)] for i in range(m)]))

    def _recurrence_from_compatibility(self, compatibility):
        r'''
            Method that returns the recurrence from a compatibility condition.

            This is the "basic" method for recurrences. Other recurrences (built from polynomials or similar) use this 
            basic method. Hence, for changing the method to compute the recurrence from the compatibility condition, 
            this is the method to change.
        '''
        if not isinstance(compatibility, (list, tuple)) or len(compatibility) != 4:
            raise TypeError("The compatibility condition is not valid")

        A,B,m,alpha = compatibility
        ## We do the transformation
        Sn = self.Sn(); Sni = self.Sni(); n = self.n()
        def SN(index):
            if(index == 0):
                return 1
            elif(index > 0):
                return Sn**index
            else:
                return Sni**(-index)
        
        # We have to distinguish between m = 1 and m > 1
        if(m == 1): # we return an operator
            recurrence = sum(alpha(0,i,n-i)*SN(-i) for i in range(-A,B+1))
            output = self.simplify_operator(recurrence)
        elif(m > 1):
            output = Matrix(
                [
                    [self.simplify_operator(sum(
                        alpha(j,i,self.n()+(r-i-j)//m)*SN((r-i-j)//m)
                        for i in range(-A,B+1) if ((r-i-j)%m == 0)
                    )) for j in range(m)
                    ] for r in range(m)
                ])
        else:
            raise TypeError("The number of sections must be a positive integer")
        return output

    def recurrence(self, operator, sections=None, cleaned=False):
        r'''
            Method to get the recurrence for a compatible operator.
            
            This method returns the recurrence equation induced for a compatible operator. 
            In :arxiv:`2202.05550` this compatibility
            is shown to be an algebra isomorphism, so we can compute the compatibility
            final sequence operator using the ``ore_algebra`` package and a plain 
            substitution.
            
            INPUT:

            * ``operator``: the operator we want to get the compatibility. It has to be the
              name for any generator in an ``ore_algebra`` package or the generator
              itself.
            * ``sections``: number of desired sections for the recurrence compatibility.
              The output will be then a square matrix of this size. If ``None`` is given,
              the default recurrence is returned.

            OUTPUT:

            An operator in the algebra returned by :func:`OS` that represents the compatibility
            condition of ``operator`` with the basis ``self``.

            If ``sections`` is a positive integer greater than 1, then a matrix of that size
            is returned.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: P = PowerBasis()
                sage: P.recurrence('x')
                Sni
                sage: P.recurrence('Dx')
                (n + 1)*Sn
                sage: P11 = PowerBasis(1,1)
                sage: P11.recurrence('x')
                Sni - 1
                sage: P11.recurrence('Id')
                1
                sage: P11.recurrence('Dx')
                (n + 1)*Sn
                sage: B = BinomialBasis()
                sage: B.recurrence('x')
                n*Sni + n
                sage: B.recurrence('E')
                Sn + 1
                sage: H = HermiteBasis()
                sage: H.recurrence('x')
                (n + 1)*Sn + 1/2*Sni
                sage: H.recurrence('Dx')
                (2*n + 2)*Sn

            We can also use the operators from :class:`ore_algebra.OreAlgebra` to get the compatibility. Here
            we see some examples extracted from Example 25 in :arxiv:`2202.05550`::

                sage: from pseries_basis.misc.ore import get_recurrence_algebra
                sage: OE, (x,E) = get_recurrence_algebra("x", "E", rational=False)
                sage: example25_1 = E - 3; B.recurrence(example25_1)
                Sn - 2
                sage: example25_2 = E^2 - 2*E + 1; B.recurrence(example25_2)
                Sn^2
                sage: example25_3 = E^2 - E - 1; B.recurrence(example25_3)
                Sn^2 + Sn - 1
                sage: example25_4 = E - (x+1); B.recurrence(example25_4)
                Sn + (-n)*Sni + (-n)
                sage: example25_5 = E^3 - (x^2+6*x+10)*E^2 + (x+2)*(2*x+5)*E-(x+1)*(x+2)
                sage: B.recurrence(example25_5)
                Sn^3 + (-n^2 - 6*n - 7)*Sn^2 + (-2*n^2 - 8*n - 7)*Sn + (-n^2 - 2*n - 1)
        '''
        if not isinstance(operator, (tuple, str)): # the input is a polynomial
            ## Trying to get a polynomial from the input
            if(operator in self.OB().base_ring()):
                return self.OS()(operator)
            elif(operator.parent() is SR): # case of symbolic expression
                if(any(not operator.is_polynomial(v) for v in operator.variables())):
                    raise NotCompatibleError("The symbolic expression %s is not a polynomial" %operator)
                operator = operator.polynomial(self.OB().base_ring())
                recurrences = {str(v): self.recurrence(str(v), sections) for v in operator.variables()}
                output = self.simplify_operator(operator(**recurrences))
            elif(isinstance(operator, OreOperator)): # case of ore_algebra operator
                mons, coeffs = poly_decomposition(operator.polynomial())
                # checking the type of coefficients and computing the final coefficients
                if operator.parent().base().gens()[0] != 1:
                    base_ring = operator.parent().base().base_ring()
                    rec_coeffs = {str(v): self.recurrence(str(v), sections) for v in operator.parent().base().gens()}
                    if is_based_field(operator.parent()):
                        if any(is_Matrix(v) for v in rec_coeffs.values()):
                            if any(not c.denominator() in self.OS().base().base_ring() for c in coeffs):
                                raise NotCompatibleError("Compatibility by sections when having denominators")
                            coeffs = [(1/self.OS().base().base_ring()(str(c.denominator()))) * c.numerator()(**rec_coeffs) for c in coeffs]
                        else:
                            coeffs = [c.numerator()(**rec_coeffs)*(1/self.OS().base()(str(c.denominator()(**rec_coeffs)))) for c in coeffs]            
                    else:
                        coeffs = [c(**rec_coeffs) for c in coeffs]
                else:
                    base_ring = operator.parent().base()
                # computing the monomials
                rec_mons = {str(v): self.recurrence(str(v), sections) for v in operator.parent().gens()}
                mons = [m.change_ring(base_ring)(**rec_mons) for m in mons]
                # computing the final recurrence operator
                output = self.simplify_operator(sum(coeffs[i]*mons[i] for i in range(len(mons))))
            elif(isinstance(operator, OperatorAlgebra_element)): # case of an Operator algebra
                rec_mons = {str(v) : self.recurrence(str(v), sections) for v in operator.variables()}
                logger.warning(f"The coefficients of {operator} are not been converted using compatibilities -- Not yet implemented")
                output = self.simplify_operator(
                    sum(
                        coeff*operator.parent()(mon)(**rec_mons).canonical() 
                        for mon, coeff in operator.monomial_coefficients().items()
                    ))
            else:
                try:
                    poly = operator.parent().flattening_morphism()(operator)
                except AttributeError: # we have no polynomial
                    raise NotCompatibleError("The input %s is not a polynomial" %operator)
                ## getting the recurrence for each generator
                recurrences = {str(v): self.recurrence(str(v), sections) for v in poly.variables()}
                output = self.simplify_operator(poly(**recurrences))
        else: # the input is a name or a compatibility condition
            if(isinstance(operator, str)): # getting the compatibility condition for the str case
                operator = self.compatibility(operator)
         
            
            ## Now we check the sections argument
            if(sections != None):
                operator = self.compatibility_sections(operator, sections)
                
            output = self._recurrence_from_compatibility(operator)

        ## Cleaning the output if required
        if cleaned:
            if is_Matrix(output): # case for several sections
                pass # TODO: implement this simplification --> remove Sni for rows and also denominators for rows
            else: # case with one operator
                output = self.remove_Sni(output) # we remove the inverse shift
                # we clean denominators
                _, coeffs = poly_decomposition(output.polynomial())
                to_mult = lcm([el.denominator() for el in coeffs])
                output = (to_mult * output).change_ring(self.OS().base().base())

        return output

    def recurrence_orig(self, operator):
        r'''
            Method to get the recurrence for a compatible operator.

            This method computes a recurrence operator associated with a compatible operator with this basis
            (see :func:`recurrence`). There are cases where the original operator was also a recurrence
            operator. In these cases, we are able to repeat the process of compatibility over and over.

            This method transforms the output of :func:`recurrence` so this iterative behavior can be done.

            INPUT:

            * ``operator``: the linear recurrence to check for compatibility.

            OUTPUT: 

            A new operator in the same ring as ``operator`` representing the associated recurrence with the 
            compatibility conditions w.r.t. ``self``.

            EXAMPLES::

                sage: from pseries_basis import *

            TODO: Add examples and tests for this method
        '''
        if is_recurrence_algebra(operator.parent()):
            gens_getter = gens_recurrence_algebra
        elif is_qshift_algebra(operator.parent()):
            gens_getter = gens_qshift_algebra
        else:
            raise TypeError(f"The iterative construction not implemented for [{operator.parent()}]")

        comp = self.recurrence(operator, cleaned=True)
        if is_Matrix(comp):
            # TODO: check whether this make sense or not
            raise NotImplementedError("The compatibility has sections. Unable to go back to original ring")

        x,E,_ = gens_getter(operator.parent())
        return eval_ore_operator(comp, operator.parent(), Sn = E, n = x, Sni = 1)

    def system(self, operator, sections=None):
        r'''
            Method to get a first order recurrence system associated with an operator.

            Using the method :func:`recurrence`, we can obtain a matrix `R(L)` of linear recurrence operators
            such that, for any solution to `L\cdot y = 0` where `y = \sum_{n\geq 0} c_n b_n` (where `b_n` are
            the elements of this basis), then:

            .. MATH::

                R(L) \begin{pmatrix}c_{km}\\c_{km+1}\\\vdots\\c_{km+m-1}\end{pmatrix} = 0.

            This is a linear system of recurrence equations involving the sections of `(c_n)_n`. Hence, 
            we ca obtain a first order recurrence equation associated with this system. This method
            computes (if possible) a matrix `A` with size `pm` such that

            .. MATH::

                A \begin{pmatrix}c_{km}\\c_{km+1}\\vdots\\c_{(k+p)m+m-1\end{pmatrix} = 
                \begin{pmatrix}c_{km+1}\\c_{km+2}\\vdots\\c_{(k+p)m+m\end{pmatrix}

            The study of this system may help understanding the final interlacing solution to the original
            equation `L\cdot y = 0`.

            INPUT:

            Same input as the method :func:`recurrence`.

            OUTPUT:

            The matrix `A` described above.

            TODO: add examples and tests.
        '''
        Sn = self.Sn(); Sni = self.Sni()
        R = self.recurrence(operator, sections)
        m = R.ncols()
        ## R is now the recursion matrix of size ``sections x sections``.
        ## We extract the matrices depending on the coefficient of the corresponding `Sn` and `Sni`
        dSn = 0
        dSni = 0
        for i in range(m):
            for j in range(m):
                el = R.coefficient((i,j))
                if(dSn < el.degree(Sn)):
                    dSn = el.degree(Sn)
                if(dSni < el.degree(Sni)):
                    dSni = el.degree(Sni)
        matrices = {}

        from sage.rings.polynomial.polydict import ETuple #pylint: disable=no-name-in-module
        for k in range(dSn+1): # getting the positive shift matrices
            matrices[k] = Matrix(self.OB(), 
            [[R.coefficient((i,j)).dict().get(ETuple((k,0)), 0) for j in range(m)] 
            for i in range(m)])
        for k in range(1, dSni+1): # getting the negative shift matrices
            matrices[-k] = Matrix(self.OB(), 
            [[R.coefficient((i,j)).dict().get(ETuple((0,k)), 0) for j in range(m)] 
            for i in range(m)])

        matrices = [matrices[i] for i in range(-dSni, dSn+1)] # putting matrices in list format

        ## Removing the Sni factor
        n = self.n()
        matrices = [Matrix(self.OB(), [[el(n=n+dSni) for el in row] for row in matrix]) for matrix in matrices]

        ## Checking invertibility of leading coefficient
        if(matrices[-1].determinant() == 0):
            raise ValueError("The leading matrix is not invertible")
        inverse_lc = matrices[-1].inverse()
        matrices = [inverse_lc*el for el in matrices]
        rows = []
        for i in range(len(matrices)-2):
            rows += [(i+1)*[0] + [1] + (len(matrices)-i-3)*[0]]
        rows += [-matrices[:-1]]
        return block_matrix(self.OB(), rows)

    @cached_method
    def compatibility_sections(self, compatibility, sections):
        r'''
            Compute an extension of a compatibility for larger amount of sections.
            
            This method takes a compatibility input (i.e., a compatible operator or the 
            tuple `(A,B,m,alpha_{i,j,k})` representing the compatibility) and returns a 
            new tuple `(A,B,M,\tilde{\alpha}_{i,j,k})` where `M` is the desired number
            of final sections.
            
            INPUT:

            * ``compatibility``: here we need either an operator (or a valid input for
              :func:`compatibility`) or a tuple with four entries `(A, B, m, \alpha_{i,j,k})`
              where the last entry is a function that takes three arguments:

                  * ``i``: an integer from `0` up to `m-1`.
                  * ``j``: an integer from `-A` up to `B`.
                  * ``k``: an element of :func:`OB` to index the coefficient.
                  
            * ``sections``: the value for the new number of sections `M`.

            OUTPUT:

            A tuple `(A,B,M,\tilde{\alpha}_{i,j,k})` representing the same compatibility
            but for a new number of sections `M`.

            TODO: add examples
        '''
        ## Considering the case of an operator
        if(not type(compatibility) in (tuple, list)):
            compatibility = self.compatibility(compatibility)

        ## Checking the input
        if(len(compatibility) != 4):
            raise TypeError("The input must a tuple with 3 elements")
        A,B,m,alpha = compatibility

        if((not sections in ZZ) or sections <= 0):
            raise ValueError("The number of sections must be a positive integer (got %s)" %sections)
        elif(sections%m != 0):
            raise ValueError("The number of sections must be a multiple of the compatibility size of the operator")
        elif((not A in ZZ) or A < 0):
            raise ValueError("The upper bound condition is not valid")
        elif((not B in ZZ) or B < 0):
            raise ValueError("The lower bound condition is not valid")

        l = sections//m # the extension factor of the compatibility
        new_alpha = lambda i,j,k : alpha(i%m, j, l*k + i//m)
        return (A, B, sections, new_alpha)
    
    @cached_method
    def compatibility_coefficient(self, operator):
        r'''
            Method to get the compatibility coefficient.
            
            Following :arxiv:`2202.05550`, an operator `L` is
            `(A,B)`-compatible if there are some `\alpha_{n,i}` such that for all `n = kr + j`

            .. MATH::

                L \cdot b_n = \sum_{i=-A}^B \alpha_{r,i}(k)b_{n+i}.
            
            This method returns, for the given operator, a function with 3 parameters
            `(i,j,n)` representing the element `\alpha_{i,j}(n)`.
            
            INPUT:

            * ``operator``: the operator we want to get the compatibility. It can be the
              name for any generator in an ``ore_algebra`` or the generator itself.
                
            OUTPUT:

            The coefficients `\alpha_{i,j}(n)` for the operator in ``operator`` as a function
            with three parameters `(i,j,n)`.
        '''
        return self.compatibility(operator)[3]

    def scalar(self, factor):
        r'''
            Method to create an equivalent basis built by multiplying by a sequence of constants.

            It is clear that if we compute the Hadamard product of a basis and a sequence
            of constants, we obtain a basis of the ring of formal power series. This 
            new basis carry over all the compatibilities of the old basis with small modifications.

            INPUT:

            * ``factor``: rational function in `n` that will be interpreted as a sequence.

            OUTPUT:

            A :class:`PSBasis` of the same type as ``self`` but representing the equivalent basis
            multiplied by ``factor``.
        '''
        hyper, quotient = self.is_hypergeometric(factor)
        if(factor in self.OB()): # rational function case
            if(not self.valid_factor(self.OB()(factor))):
                raise ValueError("The scalar factor is not valid: not well defined for all 'n'")
            new_basis = self._scalar_basis(factor)
            # we extend the compatibilities
            self.__scalar_extend_compatibilities(new_basis, factor)  
        elif(hyper): # the input is an hypergeometric expression
            new_basis = self._scalar_hypergeometric(factor, quotient)
            # we extend the compatibilities
            self.__scalar_hyper_extend_compatibilities(new_basis, factor, quotient)
        else: # we need that factor(n) is a rational function
            n = self.n(); factor_n = factor(n=n)
            if((not factor_n in self.OB()) or (not self.valid_factor(self.OB()(factor)))):
                raise ValueError("The scalar factor is not valid: the general term is not well defined for all 'n'")
            new_basis = self._scalar_basis(factor) # we create the structure for the new basis
            # we extend the compatibilities
            self.__scalar_extend_compatibilities(new_basis, factor)  
             
        return new_basis

    def _scalar_basis(self, factor):
        r'''
            Method that actually builds the structure for the new basis.

            This method build the actual structure for the new basis. This may have
            some intrinsic compatibilities that will be extended with the compatibilities that 
            are in ``self`` according with the factor.

            By default, this structure will be :class:`BruteBasis`, with the trivial method to 
            generate new elements. However, different subclasses may override this method to 
            provide a better structure to the scalar product.

            INPUT:

            * ``factor``: the scalar factor for each step.
        '''
        return BruteBasis(lambda n : self.element(n)*factor(n=n), self.by_degree())

    def _scalar_hypergeometric(self, factor, quotient): #pylint: disable=unused-argument
        r'''
            Method that actually builds the structure for the new basis.

            This method build the actual structure for the new basis in the case of 
            a hypergeometric factor. This may have
            some intrinsic compatibilities that will be extended with the compatibilities that 
            are in ``self`` according with the factor.

            By default, this structure will be :class:`BruteBasis`, with the trivial method to 
            generate new elements. However, different subclasses may override this method to 
            provide a better structure to the scalar product.

            INPUT:

            * ``factor``: the scalar factor for each step.
            * ``quotient``: the quotient that defines ``factor`` as a hypergeometric element.
        '''
        return BruteBasis(lambda n : self.element(n)*factor(n=n), self.by_degree())

    def __scalar_extend_compatibilities(self, new_basis, factor):
        r'''
            Method to extend compatibilities to ``new_basis`` with a rational function or a method
            that returns a rational function when fed by `n` (see :func:`OB`)
        '''
        compatibilities = [key for key in self.compatible_operators() if (not key in new_basis.compatible_operators())]
        for key in compatibilities:
            A, B, m, alpha = self.compatibility(key)
            new_basis.set_compatibility(key, (A, B, m, lambda i,j,k : alpha(i,j,k)*(factor(n=k*m+i)/factor(k*m+i+j))), type=self.compatibility_type(key))
            
        return

    def __scalar_hyper_extend_compatibilities(self, new_basis, factor, quotient): #pylint: disable=unused-argument
        r'''
            Method to extend compatibilities to ``new_basis`` with a rational function or a method
            that returns a rational function when fed by `n` (see :func:`OB`).

            If ``factor`` (let say `f_n`) is hypergeometric with defining quotient given by ``quotient``
            (denoted by `q_n`), then we have for all `n \in \mathbb{N}` that:

            .. MATH::

                f_{n+1} = q_nf_n

            In particular, we have that for any `m \in \mathbb{N}`:

            .. MATH::

                f_{n+m} = Q_{n,m}f_n,

            where `Q_{n,m}` is defined by:

            .. MATH::

                Q_{n,m} = \prod_{i=n}^{n+m-1}q_i
            
            This formula can be adapted for `m < 0` too.
        '''
        # defining the method for computing the jumps for ``factor`` using the quotient
        def _Q(q,n,m):
            if(m > 0):
                return prod(q(n=n+i) for i in range(m))
            elif(m < 0):
                return 1/prod(q(n=n+i) for i in range(m, 0))
            return 1

        compatibilities = [key for key in self.compatible_operators() if (not key in new_basis.compatible_operators())]
        for key in compatibilities:
            A, B, m, alpha = self.compatibility(key)
            new_basis.set_compatibility(key, (A, B, m, lambda i,j,k : alpha(i,j,k)*_Q(1/quotient, k*m+i, j)), type=self.compatibility_type(key))
            
        return

    ### MAGIC METHODS
    def __mul__(self,other):
        r'''
            See method :func:`scalar`.
        '''
        try:
            return self.scalar(other)
        except:
            return super().__mul__(other)
        
    def __rmul__(self, other):
        r'''
            See method :func:`scalar`.
        '''
        return self.__mul__(other)

    def __truediv__(self,other):
        r'''
            See method :func:`scalar`.
        '''
        try:
            return self.scalar(1/other)
        except:
            return NotImplemented
    
    ### MAGIC REPRESENTATION METHODS
    def __repr__(self):
        return f"{self.__class__.__name__} -- WARNING: this is an abstract class"
    
    ### OTHER ALIASES FOR METHODS
    A = get_lower_bound #: alias for the method :func:`get_lower_bound`, according to notation in :arxiv:`2202.05550`
    B = get_upper_bound #: alias for the method :func:`get_upper_bound`, according to notation in :arxiv:`2202.05550`
    alpha = compatibility_coefficient #: alias for the method :func:`compatibility_coefficient`, according to notation in :arxiv:`2202.05550`

class BruteBasis(PSBasis):
    r'''
        A brute type of basis where the elements are provided by a method.

        Class for representing basis where the construction does not fit into any other construction
        but can be given, element by element, via a function. These basis have no default compatibilities
        and provide no guarantee that the set compatibilities are correct.

        In order to reduce the impact of this lack of proof, we provide a method to check empirically the compatibility 
        for certain amount of elements in the basis.

        INPUT:

        * ``elements``: function or lambda method that takes one parameter `n` and return the `n`-th element
          of this basis.
        * ``base``: base domain for the sequences this basis represents.
        * ``universe``: domain where the elements of the basis will be represented.
        * ``degree``: indicates if it is a polynomial basis or an order basis.
        * ``var_name``: (only used if ``universe`` is None and ``degree`` is True) provides a name for the main variable
          of the polynomials that compose this basis.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: B = BruteBasis(lambda n : binomial(x,n), QQ, degree=True)
            sage: B2 = BinomialBasis()
            sage: all(B[i] == B2[i] for i in range(100))
            True

        **Be careful**: this method does not check that the lambda function induces a basis nor that 
        the ``degree`` argument is correct::

            sage: B = BruteBasis(lambda n : 0, ZZ, ZZ, False)
            sage: all(B[i] == 0 for i in range(100))
            True
    '''
    def __init__(self, elements, base, universe=None, degree=True, var_name=None):
        super().__init__(base, universe, degree, var_name)
        self.__get_element = elements

    def change_base(self, base):
        return BruteBasis(
            self.__get_element, 
            base, 
            self.universe, 
            self.by_degree(), 
            str(self.universe.gens()[0]) if is_PolynomialRing(self.universe) and self.by_degree() else None
        )

    @cached_method
    def _element(self, n):
        r'''
            Method to return the `n`-th element of the basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.misc.sequences.Sequence`.
            See method :func:`~pseries_basis.misc.sequences.element` for further information.
        '''
        output = self.__get_element(n)

        return output if self.universe is None else self.universe(output)

    @PSBasis.functional_seq.getter
    def functional_seq(self) -> Sequence:
        if is_PolynomialRing(self.universe):
            return LambdaSequence(lambda k,n : self(k)[n], self.base, 2, False)
        elif self.universe is SR:
            return LambdaSequence(lambda k,n: self(k).taylor(self(k).variables()[0], 0, n).polynomial(self.base)[k], self.base, 2, False)
        else:
            return LambdaSequence(lambda k,n : self(k).derivative(times=n)(0)/factorial(n), self.base, 2, False)

    @PSBasis.evaluation_seq.getter
    def evaluation_seq(self) -> Sequence:
        if is_PolynomialRing(self.universe):
            return LambdaSequence(lambda k,n : self(k)[n], self.base, 2, False)
        elif self.universe is SR:
            return LambdaSequence(lambda k,n: self(k)(**{str(self(k).variables()[0]) : n}), self.base, 2, False)
        else:
            return LambdaSequence(lambda k,n : self(k)(n), self.base, 2, False)

    def __repr__(self):
        return f"Brute basis: ({self[0]}, {self[1]}, {self[2]}, ...)"

    def _latex_(self):
        return r"Brute basis: \left(%s, %s, %s, \ldots\right)" %(latex(self[0]), latex(self[1]), latex(self[2]))

class SequenceBasis(PSBasis):
    r'''
        Abstract class for basis only viewed as sequences.

        Formal power series `f(x) \in \mathbb{K}[[x]]` can be also seen as sequences. Hence, a basis of the 
        ring of formal power series can be defined using a bi-sequence, i.e., a sequence from `\mathbb{N}^2` to 
        some field. This bi-sequence is exactly the one obtained by :func:`functional_seq`.

        This class allows to represent basis of formal power series by given this bi-sequence.

        INPUT:

        * ``base``: a SageMath structure for `\mathbb{K}`.
        * ``sequence``: a :class:`~pseries_basis.misc.sequences.Sequence` of dimension 2, whose universe can be changed to ``base``.
    '''
    def __init__(self, base, sequence : Sequence, degree : bool = True):
        if not isinstance(sequence, Sequence):
            raise TypeError("The value for a sequence must be of class :class:`Sequence`")
        if not sequence.dim == 2:
            raise ValueError(f"The sequence must have 2 variables. It has {sequence.dim}")
        self.__sequence = sequence.change_universe(base)

        super().__init__(base, SequenceSet(1, base), degree)

    @PSBasis.functional_seq.getter
    def functional_seq(self) -> Sequence:
        return self.__sequence

    def _element(self, *indices):
        return self.functional_seq.subsequence(indices[0]) #pylint: disable=no-member

    def change_base(self, base):
        return SequenceBasis(base, self.functional_seq, self.by_degree())

    def shift_in(self, shift):
        r'''
            Method to apply shift to the second layer of this basis.

            It is quite common to compute shifts for the elements of a basis for the formal power series ring. This method
            allows to obtain the corresponding :class:`SequentialBasis` after performing the given shift to all elements
            of the basis defined by ``self``.

            INPUT:

            * ``shift``: the integer defining the shift to be applied to the elements of ``self``.

            OUTPUT:

            A new :class:`SequenceBasis` whose elements are the elements of ``self`` after applying the given ``shift``.
        '''
        return SequenceBasis(self.base, self.functional_seq.shift(0,shift), self.by_degree()) #pylint: disable=no-member

    def mult_in(self, prod):
        r'''
            Method to multiply the inner sequences and obtain a new basis.

            The usual behavior or the multiplication does not allow to do this if the multiplication is also a 
            sequence. This method allows to express this other operation that is a multiplication for each of
            the sequences inside.
        '''
        return SequenceBasis(self.base, LambdaSequence(lambda n,k : (prod*self[n])[k], self.base, 2), self.by_degree())

class PolyBasis(PSBasis):
    r'''
        Abstract class for a polynomial power series basis. 
        
        Their elements must be indexed by natural numbers such that the n-th
        element of the basis has degree exactly `n`.
        
        This class **must never** be instantiated.

        List of abstract methods:

        * :func:`PSBasis.element`.
    '''
    def __init__(self, base=QQ, var_name=None, **kwds):
        super().__init__(
            base=base, universe=kwds.pop("universe", None), degree=kwds.pop("degree", True), var_name=var_name, # arguments for PSBasis
            **kwds # other arguments for other builders (allowing multi-inheritance)
        )
        super(PolyBasis,self).__init__(base, None, True, var_name)

    @PSBasis.functional_seq.getter
    def functional_seq(self) -> Sequence:
        return LambdaSequence(lambda k,n : self[k][n], self.base, 2, False)

    @PSBasis.evaluation_seq.getter
    def evaluation_seq(self) -> Sequence:
        return LambdaSequence(lambda k,n : self[k](n), self.base, 2, False)

    def __repr__(self):
        return "PolyBasis -- WARNING: this is an abstract class"

class OrderBasis(PSBasis):
    r'''
        Abstract class for a order power series basis. 
        
        Their elements must be indexed by natural numbers such that the n-th
        element of the basis has order exactly `n`.
        
        This class **must never** be instantiated.

        List of abstract methods:

        * :func:`PSBasis.element`.
    '''
    def __init__(self, base=QQ, universe=None):
        super(OrderBasis,self).__init__(base, universe, False)

    @PSBasis.functional_seq.getter
    def functional_seq(self) -> Sequence:
        if is_PolynomialRing(self.universe):
            return LambdaSequence(lambda k,n : self(k)[n], self.base, 2, False)
        elif self.universe is SR:
            return LambdaSequence(lambda k,n: self(k).taylor(self(k).variables()[0], 0, n).polynomial(self.base)[k], self.base, 2, False)
        else:
            return LambdaSequence(lambda k,n : self(k).derivative(times=n)(0)/factorial(n), self.base, 2, False)

    def is_quasi_func_triangular(self):
        return True

def check_compatibility(basis, operator, action, bound=100):
    r'''
        Method that checks that a compatibility formula holds for some examples.

        This method takes a :class:`PSBasis`, an operator compatibility (either a tuple with the 
        compatibility data or the operator that must be compatible with the basis), an actions with the 
        map for the operator and a bound and checks that the induced compatibility identity holds for 
        the first terms of the basis.

        INPUT:

        * ``basis``: a :class:`PSBasis` to be checked.
        * ``operator``: a tuple `(A,B,m,\alpha)` with the compatibility condition (see :func:`PSBasis.compatibility`)
          or a valid input of that method.
        * ``action``: a map that takes elements in ``basis.universe`` and perform the operation of ``operator``.
        * ``bound``: positive integer with the number of cases to be checked.
    '''
    if(isinstance(operator, tuple)):
        a,b,m,alpha = operator
    else:
        a,b,m,alpha = basis.compatibility(operator)
        
    mm = int(ceil(a/m))
    return all(
        all(
            sum(basis[k*m+r+i]*basis.base(alpha(r,i,k)) for i in range(-a,b+1)) == action(basis[k*m+r]) 
            for r in range(m)) 
        for k in range(mm, bound))

__all__ = ["PSBasis", "BruteBasis", "SequenceBasis", "PolyBasis", "OrderBasis", "check_compatibility"]
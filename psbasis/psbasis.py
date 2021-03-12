r'''
    Sage package for Power Series Basis.

    This module introduces the basic structures in Sage for computing with *Power
    Series Basis*. We based this work in the paper :arxiv:`1804.02964v1`
    by M. Petkovsek, where all definitions and proofs for the algorithms here can be found.

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

        sage: from psbasis import *

    This package includes no example since all the structures it offers are abstract, so they should
    never be instancaited. For particular examples and test, look to the modules :mod:`~psbasis.factorial_basis`
    and :mod:`~psbasis.product_basis`.
'''

## Sage imports
from sage.all import (FractionField, PolynomialRing, ZZ, QQ, Matrix, cached_method, latex, factorial, diff, 
                        SR, Expression, prod, hypergeometric)
from sage.symbolic.operators import add_vararg, mul_vararg
from sage.structure.element import is_Matrix # pylint: disable=no-name-in-module

# ore_algebra imports
from ore_algebra import OreAlgebra

## Private module variables (static elements)
_psbasis__OB = FractionField(PolynomialRing(QQ, ['n']))
_psbasis__n = _psbasis__OB.gens()[0]
_psbasis__OS = OreAlgebra(_psbasis__OB, ('Sn', lambda p: p(n=_psbasis__n+1), lambda p : 0), ('Sni', lambda p : p(n=_psbasis__n-1), lambda p : 0))
_psbasis__OSS = OreAlgebra(_psbasis__OB, ('Sn', lambda p: p(n=_psbasis__n+1), lambda p : 0))
_psbasis__Sn = _psbasis__OS.gens()[0]
_psbasis__Sni = _psbasis__OS.gens()[1]

class PSBasis(object):
    r'''
        Generic (abstract) class for a power series basis.
        
        Their elements must be indexed by natural numbers and ordered by
        *degree* or *order*.
        
        This class **must never** be instantiated, but contains all the methods that will
        have a common implementation for particular basis.

        List of abstract methods:

        * :func:`~PSBasis.element`.
        * :func:`~PSBasis._basis_matrix`.
    '''
    def __init__(self, degree=True):
        self.__degree = degree
        self.__compatibility = {}

    ### Getters from the module variable as objects of the class
    def OB(self):
        r'''
            Method to get the generic base ring for rational functions in `n`.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = PSBasis() # illegal building, do not use in general
                sage: B.OB()
                Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return _psbasis__OB

    def n(self):
        r'''
            Method to get the generic variable `n` for the recurrences.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = PSBasis() # illegal building, do not use in general
                sage: B.n()
                n
                sage: B.n().parent()
                Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return _psbasis__n

    def OS(self):
        r'''
            Method to get the generic variable :class:`~ore_algebra.OreAlgebra` for the shift 
            and inverse shift operators over the rational functions in `n`.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = PSBasis() # illegal building, do not use in general
                sage: B.OS()
                Multivariate Ore algebra in Sn, Sni over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return _psbasis__OS

    def OSS(self):
        r'''
            Method to get the generic variable :class:`~ore_algebra.OreAlgebra` with only the direct shift 
            over the rational functions in `n`.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = PSBasis() # illegal building, do not use in general
                sage: B.OSS()
                Univariate Ore algebra in Sn over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return _psbasis__OSS

    def Sn(self):
        r'''
            Method to get the generic variable for the direct shift operator.

            This object is in the ring :func:`~PSBasis.OS`.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = PSBasis() # illegal building, do not use in general
                sage: B.Sn()
                Sn
                sage: B.Sn().parent()
                Multivariate Ore algebra in Sn, Sni over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return _psbasis__Sn

    def Sni(self):
        r'''
            Method to get the generic variable for the inverse shift operator.

            This object is in the ring :func:`~PSBasis.OS`.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = PSBasis() # illegal building, do not use in general
                sage: B.Sni()
                Sni
                sage: B.Sni().parent()
                Multivariate Ore algebra in Sn, Sni over Fraction Field of Univariate Polynomial Ring in n over Rational Field
        '''
        return _psbasis__Sni
    
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

                sage: from psbasis import *
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
            hypers = [self.is_hypergeometric(el) for el in element.operands()]
            if(any(not el[0] for el in hypers)):
                return (False, None)
            return (True, sum([el[1] for el in hypers], 0))
        elif(operator is mul_vararg):
            hypers = [self.is_hypergeometric(el) for el in element.operands()]
            if(any(not el[0] for el in hypers)):
                return (False, None)
            return (True, prod([el[1] for el in hypers], 1))
        elif(operator is pow):
            base,exponent = element.operands()
            if(exponent in ZZ):
                hyper, quotient = self.is_hypergeometric(base)
                if(hyper):
                    return (hyper, quotient**ZZ(exponent))
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

                sage: from psbasis import *
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

            This allow to check if a hypergeometric element is valid as a scalar product (see emthod :func:`is_hypergeometric`)::

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
        if(not element in self.OB()):
            return False
        element = self.OB()(element)

        ## We check the denominator never vanishes on positive integers
        if(any((m >= 0 and m in ZZ) for m in [root[0][0] for root in element.denominator().roots()])):
            return False

        ## We check the numerator never vanishes on the positive integers
        if(any((m >= 0 and m in ZZ) for m in [root[0][0] for root in element.numerator().roots()])):
            return False
            
        return True

    ### BASIC METHODS
    def element(self, n, var_name=None):
        r'''
            Method to return the `n`-th element of the basis.

            The user can also get the `n`-th element of the sequence using the *magic* Python syntax for 
            element in a list (i.e., using the ``[]`` notation).

            This is an abstract method that has to be implemented in some subclass. 

            INPUT:

            * ``n``: the index of the element to get.
            * ``var_name``: the name of the variable of the resulting polynomial. If ``None`` is given, 
              we use the variable `x`. Otherwise we create the corresponding polynomial ring and 
              return the polynomial in that polynomial ring.
        '''
        raise NotImplementedError("Method element must be implemented in each subclass of polynomial_basis")
        
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

    def basis_matrix(self, nrows, ncols=None):
        r'''
            Method to get a matrix representation of the basis.

            This method returns a matrix `M = (m_{i,j})` with ``nrows`` rows and
            ``ncols`` columns such that `m_{i,j} = [x^j]f_i(x)`, where `f_i(x)` is 
            the `i`-th element of ``self``.

            INPUT:

            * ``nrows``: number of rows of the final matrix
            * ``ncols``: number of columns of the final matrix. If ``None`` is given, we
              will automatically return the square matrix with size given by ``nrows``.
        '''
        ## Checking the arguments
        if(not ((nrows in ZZ) and nrows > 0)):
            raise ValueError("The number of rows must be a positive integer")
        if(not ncols is None):
            if(not ((ncols in ZZ) and ncols > 0)):
                raise ValueError("The number of columns must be a positive integer")
            return self._basis_matrix(nrows, ncols)
        return self._basis_matrix(nrows, nrows)

    def _basis_matrix(self, nrows, ncols):
        r'''
            Method that actually computes the matrix for basis.

            In this method we have guaranteed that the arguments are positive integers.
        '''
        raise NotImplementedError("Method element must be implemented in each subclass of polynomial_basis")

    ### AUXILIARY METHODS
    def reduce_SnSni(self,operator):
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

                sage: from psbasis import *
                sage: B = PSBasis() # illegal build just for examples
                sage: Sn = B.Sn(); Sni = B.Sni()
                sage: Sn*Sni
                Sn*Sni
                sage: Sni*Sn
                Sn*Sni
                sage: B.reduce_SnSni(Sn*Sni)
                1
                sage: B.reduce_SnSni(Sni*Sn)
                1
                sage: B.reduce_SnSni(Sni*Sn^2 - 3*Sni^2*Sn^3 + Sn)
                -Sn
        '''
        if(is_Matrix(operator)):
            base = operator.parent().base(); n = operator.nrows()
            return Matrix(base, [[self.reduce_SnSni(operator.coefficient((i,j))) for j in range(n)] for i in range(n)])
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

    def remove_Sni(self, operator):
        r'''
            Method to remove ``Sni`` from an operator. 

            This method allows to compute an equivalent operator but without inverse shifts. This
            can be helpful to compute a holonomic operator and apply methods from the package
            :mod:`ore_algebra` to manipulate it.

            We are usually interested in sequencessuch that when we apply an operator 
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

                sage: from psbasis import *
                sage: B = PSBasis() # do not do this in your code
                sage: Sn = B.Sn(); Sni = B.Sni()
                sage: B.remove_Sni(Sni)
                1
                sage: B.remove_Sni(Sni + 2 + Sn)
                Sn^2 + 2*Sn + 1
        '''
        Sni = self.Sni(); Sn = self.Sn()
        if(is_Matrix(operator)):
            d = max(max(el.degree(self.Sni()) for el in row) for row in operator)
            return Matrix(self.OSS(), [[self.reduce_SnSni((Sn**d)*el) for el in row] for row in operator])

        d = operator.degree(Sni)
        return self.OSS()(self.reduce_SnSni((Sn**d)*operator))
    
    def polynomial_ring(self,var_name='x'):
        r'''
            Method to create a polynomial ring.

            This method creates a polynomial ring with a given variable name
            with coefficients in the field given by default by :func:`OB`.
        '''
        return PolynomialRing(self.OB().base(), [var_name])
    
    ### COMPATIBILITY RELATED METHODS
    def set_compatibility(self, name, trans, sub=False):
        r'''
            Method to set a new compatibility operator.

            This method sets a new compatibility condition for an operator given 
            by ``name``. The compatibility condition must be given as a tuple
            `(A, B, m, \alpha_{i,j,k})` where `A` is the lower bound for the compatibility,
            `B` is the upper bound for the compatibility and `m` is the number of sections
            for the compatibility. In this way, we have tht the operator `L` defind by ``name``
            satisfies:

            .. MATH::

                L \cdot b_{km+r} = \sum_{i=-A}^B \alpha_{r, i, k} b_{km+r+j}

            See :arxiv:`1804.02964v1` for further information about the
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
        '''
        name = str(name)
        
        if(name in self.__compatibility and (not sub)):
            print("The operator %s is already compatible with this basis -- no changes are done" %name)
            return
        
        if(isinstance(trans, tuple)):
            if(len(trans) != 4):
                raise ValueError("The operator given has not the appropriate format: not a triplet")
            A, B, m, alpha = trans
            if((not A in ZZ) or A < 0):
                raise ValueError("The lower bound parameter is not valid: %s" %A)
            if((not B in ZZ) or B < 0):
                raise ValueError("The upper bound parameter is not valid: %s" %B)
            if((not m in ZZ) or m < 1):
                raise ValueError("The number of sections is not valid: %s" %m)

            # TODO: how to check the alpha?
            self.__compatibility[name] = (ZZ(A),ZZ(B),ZZ(m),alpha)
        elif(trans in self.OS()):
            trans = self.reduce_SnSni(trans); Sn = self.Sn(); Sni = self.Sni(); n = self.n()
            A = trans.degree(Sn); B = trans.degree(Sni); trans = trans.polynomial()
            alpha = ([self.OB()(trans.coefficient({Sn:i}))(n=n-i) for i in range(A, 0, -1)] + 
                    [self.OB()(trans.constant_coefficient())] + 
                    [self.OB()(trans.coefficient({Sni:i}))(n=n+i) for i in range(1, B+1)])
            self.__compatibility[name] = (ZZ(A), ZZ(B), ZZ(1), lambda i, j, k: alpha[j+A](n=k))
                    
    @cached_method
    def get_lower_bound(self, operator):
        r'''
            Method to get the lower bound compatibility for an operator.
            
            This method returns the minimal index for the compatibility property
            for a particular operator. In the notation of the paper
            :arxiv:`1804.02964v1`, for a `(A,B)`-compatible operator,
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
            :arxiv:`1804.02964v1`, for a `(A,B)`-compatible operator,
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

                sage: from psbasis import *
                sage: BinomialBasis().compatible_operators()
                dict_keys(['x', 'E'])
                sage: PowerBasis().compatible_operators()
                dict_keys(['x', 'Id', 'Dx'])
                sage: HermiteBasis().compatible_operators()
                dict_keys(['x', 'Dx'])
                sage: B = FallingBasis(1,2,3)
                sage: B.compatible_operators()
                dict_keys(['x', 'E_3'])
                
            This output gets updated when we add new compatibilities
                
                sage: B.set_compatibility('s', 1)
                sage: B.compatible_operators()
                dict_keys(['x', 'E_3', 's'])
        '''
        return self.__compatibility.keys()

    def has_compatibility(self, operator):
        r'''
            Method to know if an operator has compatibility with this basis.

            This method checks whether the operator given has a compatibility or not.

            INPUT:

            * ``operator``: the operator we want to know if it is compatible or not.
              It has to be the name for any generator in an ``ore_algebra``.

            OUTPUT:

            ``True`` if the givenoperator is compatible and ``False`` otherwise.

            EXAMPLES::

                sage: from psbasis import *
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
                sage: B.has_compatibility('E')
                False
                sage: B.has_compatibility('E_3')
                True
                
            This output gets updated when we add new compatibilities
                
                sage: B.has_compatibility('s')
                False
                sage: B.set_compatibility('s', 1)
                sage: B.has_compatibility('s')
                True
        '''
        return isinstance(operator, str) and operator in self.__compatibility

    def compatibility(self, operator):
        r'''
            Method to get the compatibility condition for an operator.

            This method returns the tuple `(A, B, m, \alpha_{i,j,k})` that defines
            the compatibility condition for the operator `L` defined by ``operator``.
            this compatibility has to be stored already (see method :func:`set_compatibility`).

            INPUT:

            * ``operator``: string or generator of an *ore_algebra* that is compatible with ``self``.
              If it is not a string, we cast it.

            TODO: add examples
        '''
        return self.__compatibility[str(operator)]

    def recurrence(self, operator, sections=None):
        r'''
            Method to get the recurrence for a compatible operator.
            
            This method returns the recurrence equation induced for a compatible operator. 
            In :arxiv:`1804.02964v1` this compatibility
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

                sage: from psbasis import *
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
            we see some examples extracted from :arxiv:`1804.02964v1`::

                sage: OE.<E> = OreAlgebra(QQ[x], ('E', lambda p : p(x=x+1), lambda p : 0))
                sage: x = B[1].parent().gens()[0]
                sage: example4_1 = E - 3; B.recurrence(example4_1)
                Sn - 2
                sage: example4_2 = E^2 - 2*E + 1; B.recurrence(example4_2)
                Sn^2
                sage: example4_3 = E^2 - E - 1; B.recurrence(example4_3)
                Sn^2 + Sn - 1
                sage: example4_4 = E - (x+1); B.recurrence(example4_4)
                Sn + (-n)*Sni - n
                sage: example4_5 = E^3 - (x^2+6*x+10)*E^2 + (x+2)*(2*x+5)*E-(x+1)*(x+2)
                sage: B.recurrence(example4_5)
                Sn^3 + (-n^2 - 6*n - 7)*Sn^2 + (-2*n^2 - 8*n - 7)*Sn - n^2 - 2*n - 1
        '''
        if(not isinstance(operator, str)): # we received the operator itself
            ## Casting the operator to a polynomial
            try:
                PR = operator.polynomial().parent()
                poly = operator.polynomial()
                poly = PR.flattening_morphism()(poly)
            except TypeError:
                poly = operator
            
            ## getting the recurrence for each generator
            recurrences = {str(v): self.recurrence(str(v)) for v in poly.variables()}
            return self.reduce_SnSni(poly(**recurrences))

        ## str case: we need to get the compatibility for that operator
        ## First we get the compatibility condition
        A,B,m,alpha = self.compatibility(operator)
        
        ## Now we check the sections argument
        if(sections != None):
            A,B,m,alpha = self.compatibility_sections((A,B,m,alpha), sections)

        ## Now we do the transformation
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
            return self.reduce_SnSni(recurrence)
        elif(m > 1):
            return Matrix(
                [
                    [self.reduce_SnSni(sum(
                        alpha(j,i,self.n()+(r-i-j)//m)*SN((r-i-j)//m)
                        for i in range(-A,B+1) if ((r-i-j)%m == 0)
                    )) for j in range(m)
                    ] for r in range(m)
                ])
        else:
            raise TypeError("The number of sections must be a positive integer")

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
            
            Following :arxiv:`1804.02964v1`, an operator `L` is
            `(A,B)`-compatible if there are some `\alpha_{n,i}` such that for all `n`

            .. MATH::

                L \cdot b_n = \sum_{i=-A}^B \alpha_{n,i}b_{n+i}.
            
            This method returns, for the given operator, the value `\alpha_{n,i}`
            
            INPUT:

            * ``operator``: the operator we want to get the compatibility. It can be the
              name for any generator in an ``ore_algebra`` or the generator itself.
            * ``pos``: position of the compatibility. This is the `n` in the
              definition of compatibility. 
            * ``ind``: index of the compatibility coefficient. This is the index `i` in the 
              definition of compatibility. If it is out of range, `0` is returned.
                
            OUTPUT:

            The coefficient `\alpha_{n,i}` for the operator in ``operator`` where `n` is 
            given by ``pos`` and `i` is given by ``ind``.

            WARNING:

            * The case when the compatibility rule is a matrix is not implemented.
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

    def _scalar_hypergeometric(self, factor, quotient):
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
            that returns a rational function when feeded by `n` (see :func:`OB`)
        '''
        compatibilities = [key for key in self.compatible_operators() if (not key in new_basis.compatible_operators())]
        for key in compatibilities:
            A, B, m, alpha = self.compatibility(key)
            new_basis.set_compatibility(key, (A, B, m, lambda i,j,k : alpha(i,j,k)*(factor(n=k*m+i)/factor(k*m+i+j))))
            
        return

    def __scalar_hyper_extend_compatibilities(self, new_basis, factor, quotient):
        r'''
            Method to extend compatibilities to ``new_basis`` with a rational function or a method
            that returns a rational function when feeded by `n` (see :func:`OB`).

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
            new_basis.set_compatibility(key, (A, B, m, lambda i,j,k : alpha(i,j,k)*_Q(1/quotient, k*m+i, j)))
            
        return

    ### MAGIC METHODS
    def __getitem__(self, n):
        r'''
            See method :func:`element`
        '''
        if(isinstance(n, slice)):
            return [self[i] for i in range(n.stop)[n]]
        return self.element(n)

    def __mul__(self,other):
        r'''
            See method :func:`scalar`.
        '''
        try:
            return self.scalar(other)
        except:
            return NotImplemented
        
    def __rmul__(self, other):
        r'''
            See method :func:`scalar`.
        '''
        return self.__mul__(other)
    
    ### MAGIC REPRESENTATION METHODS
    def __repr__(self):
        return "PSBasis -- WARNING: this is an abstract class"
    
    ### OTHER ALIASES FOR METHODS
    A = get_lower_bound #: alias for the method :func:`get_lower_bound`, according to notation in :arxiv:`1804.02964v1`
    B = get_upper_bound #: alias for the method :func:`get_upper_bound`, according to notation in :arxiv:`1804.02964v1`
    alpha = compatibility_coefficient #: alias for the method :func:`compatibility_coefficient`, according to notation in :arxiv:`1804.02964v1`

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
        * ``degree``: indicates if it is a polynomial basis or an order basis.

        EXAMPLES::

            sage: from psbasis import *
            sage: B = BruteBasis(lambda n : QQ[x](binomial(x,n)), True)
            sage: B2 = BinomialBasis()
            sage: all(B[i] == B2[i] for i in range(100))
            True

        **Be careful**: this method does not check that the lambda function induces a basis nor that 
        the ``degree`` argument is correct::

            sage: B = BruteBasis(lambda n : 0, False)
            sage: all(B[i] == 0 for i in range(100))
            True
    '''
    def __init__(self, elements, degree=True):
        super().__init__(degree)
        self.__get_element = elements

    @cached_method
    def element(self, n, var_name=None):
        r'''
            Method to return the `n`-th element of the basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.element` for further information.
        '''
        if(var_name is None):
            name = 'x'
        else:
            name = var_name

        if(self.by_degree()):
            return self.polynomial_ring(name)(self.__get_element(n))
        return self.__get_element(n)

    def _basis_matrix(self, nrows, ncols):
        r'''
            Method to get a matrix representation of the basis.
            
            This method *implements* the corresponding abstract method from :class:`~psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.basis_matrix` for further information.

            EXAMPLES::

                sage: from psbasis import *
                sage: from ajpastor.dd_functions import *
                sage: B = BruteBasis(lambda n : BesselD(n), False)
                sage: B2 = BruteBasis(lambda n : bessel_J(n,x), False)
                sage: B3 = BesselBasis()
                sage: B.basis_matrix(4,5) == B2.basis_matrix(4,5)
                True
                sage: B.basis_matrix(6,7)
                [      1       0    -1/4       0    1/64       0 -1/2304]
                [      0     1/2       0   -1/16       0   1/384       0]
                [      0       0     1/8       0   -1/96       0  1/3072]
                [      0       0       0    1/48       0  -1/768       0]
                [      0       0       0       0   1/384       0 -1/7680]
                [      0       0       0       0       0  1/3840       0]
                sage: B3.basis_matrix(10) == B.basis_matrix(10)
                True
        '''
        try:
            return Matrix([[self[n].sequence(k) for k in range(ncols)] for n in range(nrows)])
        except AttributeError:
            return Matrix([[diff(self[n], k)(x=0)/factorial(k) for k in range(ncols)] for n in range(nrows)])

    def __repr__(self):
        return "Brute basis: (%s, %s, %s, ...)" %(self[0],self[1],self[2])

    def _latex_(self):
        return r"Brute basis: \left(%s, %s, %s, \ldots\right)" %(latex(self[0]), latex(self[1]), latex(self[2]))

class PolyBasis(PSBasis):
    r'''
        Abstract class for a polynomial power series basis. 
        
        Their elements must be indexed by natural numbers such that the n-th
        element of the basis has degree exactly `n`.
        
        This class **must never** be instantiated.

        List of abstract methods:

        * :func:`PSBasis.element`.
    '''
    def __init__(self):
        super(PolyBasis,self).__init__(True)

    def _basis_matrix(self, nrows, ncols):
        r'''
            Method to get a matrix representation of the basis.
            
            This method *implements* the corresponding abstract method from :class:`~psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.basis_matrix` for further information.

            In particular for a :class:`PolyBasis`, the elements `P_n(x)` have degree `n` for each
            value of `n \in \mathbb{N}`. This means that we can write:

            .. MATH::

                P_n(x) = \sum_{k=0}^n a_{n,k} x^k

            And by taking `a_{n,k} = 0` for all `k > 0`, we can build the matrix 
            `M = \left(a_{n,k}\right)_{n,k \geq 0}` with the following shape:

            .. MATH::

                \begin{pmatrix}P_0(x)\\P_1(x)\\P_2(x)\\P_3(x)\\\vdots\end{pmatrix} = 
                \begin{pmatrix}
                    a_{0,0} & 0 & 0 & 0 & \ldots\\
                    a_{1,0} & a_{1,1} & 0 & 0 & \ldots\\
                    a_{2,0} & a_{2,1} & a_{2,2} & 0 & \ldots\\
                    a_{3,0} & a_{3,1} & a_{3,2} & a_{3,3} & \ldots\\
                    \vdots&\vdots&\vdots&\vdots&\ddots
                \end{pmatrix} \begin{pmatrix}1\\x\\x^2\\x^3\\\vdots\end{pmatrix}

            EXAMPLES::

                sage: from psbasis import *
                sage: B = BinomialBasis()
                sage: B.basis_matrix(5,5)
                [    1     0     0     0     0]
                [    0     1     0     0     0]
                [    0  -1/2   1/2     0     0]
                [    0   1/3  -1/2   1/6     0]
                [    0  -1/4 11/24  -1/4  1/24]
                sage: B.basis_matrix(7,3)
                [      1       0       0]
                [      0       1       0]
                [      0    -1/2     1/2]
                [      0     1/3    -1/2]
                [      0    -1/4   11/24]
                [      0     1/5   -5/12]
                [      0    -1/6 137/360]
                sage: H = HermiteBasis()
                sage: H.basis_matrix(5,5)
                [  1   0   0   0   0]
                [  0   2   0   0   0]
                [ -2   0   4   0   0]
                [  0 -12   0   8   0]
                [ 12   0 -48   0  16]
                sage: P = PowerBasis(1,1)
                sage: P.basis_matrix(5,5)
                [1 0 0 0 0]
                [1 1 0 0 0]
                [1 2 1 0 0]
                [1 3 3 1 0]
                [1 4 6 4 1]
        '''
        return Matrix([[self[n][k] for k in range(ncols)] for n in range(nrows)])
    
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
    def __init__(self):
        super(OrderBasis,self).__init__(False)

    def _basis_matrix(self, nrows, ncols):
        r'''
            Method to get a matrix representation of the basis.
            
            This method *implements* the corresponding abstract method from :class:`~psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.basis_matrix` for further information.
            
            In an order basis, the `n`-th element of the basis is always a formal power series
            of order `n`. Hence, we can consider the infinite matrix where the coefficient
            `m_{i,j} = [x^j]f_i(x) = a_{i,j}` and the matrix have the following shape:

            .. MATH::

                \begin{pmatrix}
                    a_{0,0} & a_{0,1} & a_{0,2} & a_{0,3} & \ldots\\
                    0 & a_{1,1} & a_{1,2} & a_{1,3} & \ldots\\
                    0 & 0 & a_{2,2} & a_{2,3} & \ldots\\
                    0 & 0 & 0 & a_{3,3} & \ldots\\
                    \vdots&\vdots&\vdots&\vdots&\ddots
                \end{pmatrix}

            EXAMPLES::

                sage: from psbasis import *
                sage: B = FunctionalBasis(sin(x))
                sage: B.basis_matrix(5,5)
                [   1    0    0    0    0]
                [   0    1    0 -1/6    0]
                [   0    0    1    0 -1/3]
                [   0    0    0    1    0]
                [   0    0    0    0    1]
                sage: ExponentialBasis().basis_matrix(4,7)
                [     1      0      0      0      0      0      0]
                [     0      1    1/2    1/6   1/24  1/120  1/720]
                [     0      0      1      1   7/12    1/4 31/360]
                [     0      0      0      1    3/2    5/4    3/4]
                sage: BesselBasis().basis_matrix(3,6)
                [    1     0  -1/4     0  1/64     0]
                [    0   1/2     0 -1/16     0 1/384]
                [    0     0   1/8     0 -1/96     0]
        '''
        try:
            return Matrix([[self[n].sequence(k) for k in range(ncols)] for n in range(nrows)])
        except AttributeError:
            return Matrix([[diff(self[n], k)(x=0)/factorial(k) for k in range(ncols)] for n in range(nrows)])
    
    def __repr__(self):
        return "PolyBasis -- WARNING: this is an abstract class"

__all__ = ["PSBasis", "BruteBasis", "PolyBasis", "OrderBasis"]
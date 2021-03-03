r'''
    Sage package for Factorial Series Basis.
'''
# sage imports
from sage.all import prod, vector, binomial, ZZ, cached_method, QQ, Matrix, latex

# ore_algebra imports
from ore_algebra import OreAlgebra

# Local imports
from .psbasis import PolyBasis

class FactorialBasis(PolyBasis):
    r'''
        Abstract class for a factorial basis.

        A factorial basis is a type of polynomial basis for power series where
        the `(n+1)`-th element is build from the `n`-th element. This can be seeing
        as a two-term recurrence basis.

        It provides several functionalities and methods that all Factorial Basis
        must provide, but may differ in the implementation.

        INPUT:

        * ``X``: the name for the operator representing the multiplication by `x`.
        
        List of abstract methods:

        * :func:`psbasis.psbasis.PSBasis.element`.
        * :func:`~FactorialBasis.increasing_polynomial`.
        * :func:`~FactorialBasis.increasing_basis`.
        * :func:`~FactorialBasis.matrix_ItP`.
        * :func:`~FactorialBasis.equiv_DtC`.
        * :func:`~FactorialBasis.equiv_CtD`.
    '''
    def __init__(self, X='x'):
        super(FactorialBasis,self).__init__()

        ## Saving the name of the variable
        self.__var_name = X

        ## Creating the compatibility with the multiplication by X (if possible)
        try:
            Sni = self.Sni(); n = self.n(); an = self.an; bn = self.bn
            self.set_compatibility(X, -bn(n=n+1)/an(n=n+1) + (1/an(n=n))*Sni)
        except (AttributeError, TypeError):
            pass

    def _scalar_basis(self, factor):
        r'''
            Method that actually builds the structure for the new basis.

            This method *overrides* the corresponding abstract method from :class:`psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.scalar` for further information.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = BinomialBasis(); n = B.n()
                sage: f = (n^2+1)
                sage: B2 = B.scalar(f)
                sage: all(B[i]*f(n=i) == B2[i] for i in range(100))
                True
                sage: isinstance(B2, FactorialBasis)
                True
                
            In fact we can check that the roots are the same and the leading coefficient is scaled::

                sage: all(B.rho(n=i) == B2.rho(n=i) for i in range(100))
                True
                sage: all(B.cn(n=i)*f(n=i) == B2.cn(n=i) for i in range(100))
                True

            There are subclasses that overrides this method again and create their own structures of 
            :class:`FactorialBasis`. However, :class:`~psbasis.product_basis.ProductBasis` and 
            :class:`RootSequenceBasis` do not and always return a :class:`RootSequenceBasis`::

                sage: B = ProductBasis(BinomialBasis(), PowerBasis())
                sage: f = (n+1)
                sage: B2 = B.scalar(f)
                sage: all(B[i]*f(n=i) == B2[i] for i in range(100))
                True
                sage: isinstance(B2, RootSequenceBasis)
                True
        '''
        return RootSequenceBasis(self.rho, lambda n : self.cn(n=n)*factor(n=n), X=self.var_name())
        
    def _scalar_hypergeometric(self, factor, quotient):
        r'''
            Method that actually builds the structure for the new basis.

            This method *overrides* the corresponding abstract method from :class:`psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.scalar` for further information.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = BinomialBasis(); n = B.n()
                sage: f = (n^2+1)*factorial(n+3)
                sage: B2 = B.scalar(f)
                sage: all(B[i]*f(n=i) == B2[i] for i in range(100))
                True
                sage: isinstance(B2, FactorialBasis)
                True
                
            In fact we can check that the roots are the same and the leading coefficient is scaled::

                sage: all(B.rho(n=i) == B2.rho(n=i) for i in range(100))
                True
                sage: all(B.cn(n=i)*f(n=i) == B2.cn(n=i) for i in range(100))
                True

            There are subclasses that overrides this method again and create their own structures of 
            :class:`FactorialBasis`. However, :class:`~psbasis.product_basis.ProductBasis` and 
            :class:`RootSequenceBasis` do not and always return a :class:`RootSequenceBasis`::

                sage: B = ProductBasis(BinomialBasis(), PowerBasis())
                sage: B3 = B.scalar(factorial(n))
                sage: all(B[i]*factorial(i) == B3[i] for i in range(100))
                True
                sage: isinstance(B3, RootSequenceBasis)
                True
        '''
        return RootSequenceBasis(self.rho, lambda n : self.cn(n=n)*factor(n=n), X=self.var_name())

    ## Basic properties
    def var_name(self):
        r'''
            Getter of the name for the variable.

            This is the name of the variable that will be used to characterize the multiplication
            operator `X: \mathbb{Q}[[x]] \mapsto \mathbb{Q}[[x]]` defined by `X(f(x)) = xf(x)`.
            This method returns the name of the map `X`.
        '''
        return self.__var_name

    def root_sequence(self):
        r'''
            Method that returns the root sequence of the polynomial basis.

            Since a factorial basis satisties that `P_n(x)` divides `P_{n+1}(x)` for all
            `n`, we have that the basis forms a sequence of polynomials with a persistent
            set of roots.

            We can then define the root sequence with `\rho_n` the new root in the polynomial
            `P_{n+1}(x)`.

            OUTPUT:

            This method returns a function or lambda expression (i.e., a *callable* object)
            that takes `n` as input and returns `\rho_n`.
        '''
        def __root_fn(n):
            nth_poly = (self.element(n+1)/self.element(n)).numerator()
            # This polynomial has degree 1, hence the root is easily computable
            return -nth_poly[0]/nth_poly[1]
        return __root_fn

    def leading_coefficient(self):
        r'''
            Method that returns the sequence of leading coefficients for a factorial basis.

            This method returns the leading coefficient sequence of the polynomial basis.

            OUTPUT:

            This method returns a function or lambda expression (i.e., a *callable* object)
            that takes `n` as input and returns the leading coefficient of ``self[n]``.
        '''
        return lambda n : self[n].leading_coefficient()

    def constant_coefficient(self):
        r'''
            Getter for the constant coefficient of the factorial basis.

            This method return a sequence (in `n`) for the constant coefficient of the
            increasing polynomial for the Factorial Basis. Recall that for any Factorial
            Basis, the `n`-th element divide the next in the following way:

            .. MATH::

                P_n(x) = (a_nx + b_n)P_{n-1}(x)

            OUTPUT:
            
            This method returns the value of `b_n`.
        '''
        return lambda n : self.OB()(self[n+1]/self[n])[0]

    def linear_coefficient(self):
        r'''
            Getter for the linear coefficient of the factorial basis.

            This method return a sequence (in `n`) for the linear coefficient of the
            increasing polynomial for the Factorial Basis. Recall that for any Factorial
            Basis, the `n`-th element divide the next in the following way:

            .. MATH::

                P_n(x) = (a_nx + b_n)P_{n-1}(x)

            OUTPUT:
            
            This method returns the value of `a_n`.

            EXAMPLES::

                sage: from psbasis import *
                sage: SFactorialBasis(1,0).linear_coefficient()
                1
                sage: SFactorialBasis(2,1).linear_coefficient()
                2
                sage: SFactorialBasis(1, '(n^2 - 3)/(n+1)').linear_coefficient()
                1

            This class also allows to access this value with the property :attr:`~SFactorialBasis.an`::

                sage: SFactorialBasis(1,0).an
                1
                sage: SFactorialBasis(2,1).an
                2
                sage: SFactorialBasis(1, '(n^2 - 3)/(n+1)').an
                1
        '''
        return lambda n : self.OB()(self[n+1]/self[n])[1]

    rho = property(lambda self: self.root_sequence()) #: alias property for the root sequence (see :func:`~FactorialBasis.constant_coefficient`)
    an = property(lambda self: self.linear_coefficient()) #: alias property for the linear coefficient (see :func:`~FactorialBasis.linear_coefficient`)
    bn = property(lambda self: self.constant_coefficient()) #: alias property for the constant coefficient (see :func:`~FactorialBasis.constant_coefficient`)
    cn = property(lambda self: self.leading_coefficient()) #: alias property for the leading coefficient sequence (see :func:`~FactorialBasis.constant_coefficient`)

    ## Method related with equivalence of Proposition 1
    def increasing_polynomial(self, *args, **kwds):
        r'''
            Returns the increasing factorial for the factorial basis.

            In a Factorial Basis, the `n`-th element of the basis divides all the following.
            This means for any pair of indices `m > n`, there is a particular polynomial
            `Q_{n,m}(x) = P_m(x)/P_n(x) \in \mathbb{Q}[x]`.

            This method computes the corresponding increasing polynomial.

            This is an abstract method that has to be implemented in some subclass. The input
            may depend in each subclass.

            OUTPUT:

            A polynomial in `\mathbb{Q}[x]` representing the requested increasing polynomial.
        '''
        raise NotImplementedError("Method from FactorialBasis not implemented (Abstract method)")

    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            In a Factorial Basis, the `n`-th element of the basis divides all the following.
            This means for any pair of indices `m > n`, there is a particular polynomial
            `Q_{n,m}(x) = P_m(x)/P_n(x)` (see method :func:`increasing_polynomial` for further
            information).

            In particular, for a fixed `n` and `i \in \mathbb{N}`, the polynomials `Q_{n,n+i}(x)`
            are another Factorial Basis, since we have:

            .. MATH::

                Q_{n,n+i+1}(x) = \frac{P_{n+i+1}(x)}{P_n(x)} = \frac{(a_{n+i}x + b_{n+i})P_{n+i}(x)}{P_n(x)} = 
                (a_{n+i}x + b_{n+i})Q_{n,n+i}(x).

            This method returns the corresponding structure for the Factorial Basis.

            This is an abstract method that has to be implemented in some subclass.

            INPUT:

            * ``shift``: value for the fixed shift `n` for the increasing basis.

            OUTPUT:

            An object of type :class:`FactorialBasis` representing the corresponding 
        '''
        raise NotImplementedError("Method from FactorialBasis not implemented (Abstract method)")

    def compatible_division(self, operator, src, diff=None, dst=None):
        r'''
            Method to get the division of a polynomial by other element of the basis after an operator.

            It was proven in :arxiv:`1804.02964v1` that if `L` is an `(A,B)`-compatible operator
            with a factorial basis, then for any `n \in \mathbb{N}`, we have

            .. MATH::

                P_{n-A}(x) | L\cdot P_n(x).

            Moreover, since the basis is factorial, it is clear that for any `m < n-A`, we have
            that `P_m(x)` also divides `L\cdot P_n(x)`. See method :func:`increasing_polynomial`
            for further information.

            This method allows to compute the resulting polynomial `(L\cdot P_n(x))/P_m(x)` for 
            any valid `m \leq n-A`.

            INPUT:

            * ``operator``: the operator we want to check. See the input description
              of method :func:`get_compatibility`. This operator has to be compatible,
              so we can obtain the value for `A`.
            * ``src``: value for `n`.
            * ``diff``: difference between `n` and `m`. Must be a positive integer greater than
              the corresponding `A` value for ``operator``.
            * ``dst``: value for `m`. Only used (and required) if ``diff`` is ``None``. Must
              be smaller or equal to `n-A`.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = BinomialBasis(); n = B.n(); B.get_compatibility('x')
                n*Sni + n

            This means that for all `n`, we have that:

            .. MATH::

                xB_n(x) = (n+1)B_{n+1}(x) + nB_n(x).

            Hence, we have that `B_m(x)` divides `xB_n(x)` for any `m \leq n`::

                sage: B.compatible_division('x', 5, 2) # xB_5(x) / B_3(x)
                1/20*x^3 - 7/20*x^2 + 3/5*x
                sage: x = B[1].parent().gens()[0]
                sage: all(all(x*B[i] / B[j] == B.compatible_division('x', i, dst=j) for j in range(0,i)) for i in range(50))
                True
            
            We can also check other operators like the shift `x \mapsto x+1`::

                sage: all(all(B[i](x=x+1) / B[j] == B.compatible_division('E', i, dst=j) for j in range(0,i-1)) for i in range(50))
                True
                sage: B.compatible_division('E', n, 2)                                                                                                                                                                                      
                (1/(n^2 - n))*x^2 + ((-n + 3)/(n^2 - n))*x + (-n + 2)/(n^2 - n)

            Also we can check other types of basis::

                sage: PowerBasis().compatible_division('Dx', 10, 5)
                10*x^4
                sage: PowerBasis().compatible_division('x', 30, 10)
                x^11
                sage: n = PowerBasis().n()
                sage: PowerBasis().compatible_division('x', n, 3)
                x^4
        '''
        ## Checking the arguments
        ## Reading ``src``
        if(((src in ZZ) and src < 0) or (not src in self.OB())):
            raise ValueError("The argument `src` must be a expression involving `self.n()` or a positive integer")
        n = src

        ## Compatibility of ``operator``
        A = self.A(operator); B = self.B(operator)

        ## Reading ``diff`` or ``dst``
        if(not diff is None):
            if((not diff in ZZ) or diff < A):
                raise ValueError("The argument `diff` must be None or a positive integer bigger than %s" %A)
            else:
                d = ZZ(diff); m = n - d
        else:
            if(n in ZZ):
                if((not dst in ZZ) or dst > n-A):
                    raise ValueError("The argument `dst` must be an integer smaller or equal than `src`-%s" %A)
                m = ZZ(dst); d = n - m
            else:
                d = n-dst
                if((not d in ZZ) or d > n-A):
                    raise ValueError("The argument `dst` must be an integer smaller or equal than `src`-%s" %A)
                d = ZZ(d); m = dst

        ## Computing the polynomial
        return sum(self.alpha(operator, n, i)*self.increasing_polynomial(m, dst=n+i) for i in range(-A,B+1))

    def matrix_ItP(self, *args, **kwds):
        r'''
            Method to get the matrix for converting from the increasing basis to the power basis.

            In a Factorial Basis, the `n`-th element of the basis divides all the following.
            This means for any pair of indices `m > n`, there is a particular polynomial
            `Q_{n,m}(x) = P_m(x)/P_n(x)` (see method :func:`increasing_polynomial` for further
            information).

            In particular, for a fixed `n` and `i \in \mathbb{N}`, the polynomials `Q_{n,n+i}(x)`
            are another Factorial Basis (see method :func:`increasing_basis`). This method 
            computes a matrix that represents the identity map between polynomials of degree smaller 
            or equal to a given size from the basis $Q_{n,n+i}(x)$ and the canonical power basis.

            This is an abstract method that has to be implemented in some subclass. The input
            may depend in each subclass.

            OUTPUT:

            A matrix that convert coordinates from the increasing basis to the canonical power basis.
        '''
        raise NotImplementedError("Method from FactorialBasis not implemented (Abstract method)")

    @cached_method
    def matrix_PtI(self, *args, **kwds):
        r'''
            Method for getting the matrix from the power basis to the increasing basis.

            In a Factorial Basis, the `n`-th element of the basis divides all the following.
            This means for any pair of indices `m > n`, there is a particular polynomial
            `Q_{n,m} = P_m/P_n`.

            In particular, for a fixed `n` and `i \in \mathbb{Z}`, the polynomials `Q_{n,n+i}`
            form another Factorial Basis. This method computes a matrix that represents the
            identity between polynomials of degree smaller or equal to ``size`` from the
            power basis to the basis `Q_{n,n+i}`.

            For further information about the input, see the documentation of :func:`matrix_ItP`.

            OUTPUT:

            A matrix that convert coordinates from the canonical power basis to the requested increasing basis. 

            EXAMPLES::

                sage: from psbasis import *
                sage: B = BinomialBasis()
                sage: B.matrix_PtI(3, 5)
                [   1    3    9   27   81]
                [   0    4   28  148  700]
                [   0    0   20  240 1940]
                [   0    0    0  120 2160]
                [   0    0    0    0  840]
                sage: B.matrix_PtI(3,5).inverse() == B.matrix_ItP(3,5)
                True
                sage: x = B[1].parent().gens()[0]

            The columns are the coordinates of `\{1,x,x^2,\ldots\}` in the 3-rd increasing basis::

                sage: 4*B.increasing_polynomial(3,1) + 3*B.increasing_polynomial(3,0) == x
                True
                sage: 9*B.increasing_polynomial(3,0) + 28*B.increasing_polynomial(3,1) + 20*B.increasing_polynomial(3,2) == x^2
                True

            This is equivalent to compute the basis matrix (see method :func:`~psbasis.psbasis.PSBasis.basis_matrix`) of the 
            increasing basis (see method :func:`increasing_basis`). However, due to the different notation between these two
            methods, the relation is with the transposed matrix::

                sage: B.increasing_basis(3).basis_matrix(5).inverse().transpose() == B.matrix_PtI(3,5)
                True
        '''
        return self.matrix_ItP(*args, **kwds).inverse()

    def equiv_DtC(self, *args, **kwds):
        r'''
            Method to get the equivalence condition for a compatible operator.

            Following the notation and ideas of :arxiv:`1804.02964v1`, there is an
            equivalent condition to be a compatible operator. Namely, and operator is compatible
            by definition if it expands:

            .. MATH::

                L \cdot P_n(x) = \sum_{i=-A}^B \alpha_{n,i}P_{n+i},

            and that is equivalent to the following two conditions:

            * `deg(L\cdot P_n(x))) \leq n + B`
            * `P_{n-A}(x)` divides `L \cdot P_n(x)`.

            This method takes the list of `\alpha_{n,i}` and computes
            the division `L \cdot P_n(x)/P_{n-A}(x)` for `n \geq A` as a polynomial of degree `A+B`,
            which transforms the definition of compatible operator to the two equivalent conditions
            explicitly.

            This is an abstract method that has to be implemented in some subclass. The input
            may depend in each subclass.

            OUTPUT:
                List of coefficients of $L(P_n)/P_{n-A}$.
        '''
        raise NotImplementedError("Method from FactorialBasis not implemented (Abstract method)")

    def equiv_CtD(self, *args, **kwds):
        r'''
            Method to get the equivalence condition for a compatible operator.

            Following the notation and ideas of :arxiv:`1804.02964v1`, there is an
            equivalent condition to be a compatible operator. Namely, and operator is compatible
            by definition if it expands:

            .. MATH::

                L \cdot P_n(x) = \sum_{i=-A}^B \alpha_{n,i}P_{n+i},

            and that is equivalent to the following two conditions:

            * `deg(L\cdot P_n(x))) \leq n + B`
            * `P_{n-A}(x)` divides `L \cdot P_n(x)`.

            This method takes the division `L\cdot P_n(x)/P_{n-A}(x)` as a list of its `A+B+1` coefficients and,
            computes the compatibility coefficients for the operator `L`,
            which transforms the two equivalent conditions to the definition of compatible operator
            explicitly.

            This is an abstract method that has to be implemented in some subclass. The input
            may depend in each subclass.

            OUTPUT:
                List of coefficients of `\alpha_{n,i}`.
        '''
        raise NotImplementedError("Method from FactorialBasis not implemented (Abstract method)")

    def __repr__(self):
        return "FactorialBasis -- WARNING: this is an abstract class"

class SFactorialBasis(FactorialBasis):
    r'''
        Class for representing a simple factorial basis.

        A factorial basis is a type of polynomial basis for power series where
        the `(n+1)`-th element is build from the `n`-th element. This can be seeing
        as a two-term recurrence basis.

        This factorial nature is representing using two coefficients `a_n` and `b_n`
        such that for all `n`:

        .. MATH::

            P_n = (a_nx + b_n)P_{n-1}

        INPUT:
        
        * ``an``: the sequence of leading coefficients to build the factorial basis.
        * ``bn``: the sequence of constant coefficients to build the factorial basis.
        * ``X``: the name for the operator representing the multiplication by `x`.
        * ``init``: the value of `P_0(x)`. Must be a constant.

        TODO: add examples
    '''
    def __init__(self, an, bn, X='x',init=1):
        
        ## Cheking the first element
        init = self.OB().base()(init)
        if(init == 0):
            raise ValueError("The first polynomial must be non-zero")
        self.__init = init

        ## Adding the extra information
        n = self.n()
        an = self.OB()(an(n=n)); self.__an = an
        bn = self.OB()(bn(n=n)); self.__bn = bn

        ## Initializing the FactorialBasis structure
        super(SFactorialBasis,self).__init__(X)

        ## Extra cached variables
        self.__cached_increasing = {}

    @cached_method
    def element(self, n, var_name=None):
        r'''
            Method to return the `n`-th element of the basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.element` for further information.

            For a :class:`SFactorialBasis` the output will be a polynomial of degree `n`.

            OUTPUT:

            A polynomial with variable name given by ``var_name`` and degree ``n``.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = SFactorialBasis(1,0); B
                Factorial basis: (1, x, x^2, ...)
                sage: B.element(0)
                1
                sage: B.element(10)
                x^10
                sage: B.element(5)
                x^5
                sage: B[3]
                x^3
                sage: B.element(7, var_name='y_2')
                y_2^7
        '''
        if(var_name is None):
            name = self.var_name()
        else:
            name = var_name
        R = self.polynomial_ring(name)
        x = R.gens()[0]

        if(n > 0):
            an = self.__an; bn = self.__bn
            return self.element(n-1, var_name=var_name) * (an(n=n)*x + bn(n=n))
        elif(n == 0):
            return self.__init

    def _scalar_basis(self, factor):
        r'''
            Method that actually builds the structure for the new basis.

            This method *overrides* the corresponding method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.scalar` for further information.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = BinomialBasis(); n = B.n()
                sage: f = (n^2+1)/(n+2)
                sage: B2 = B.scalar(f)
                sage: B2
                Factorial basis: (1/2, 2/3*x, 5/8*x^2 - 5/8*x, ...)
                sage: all(B[i]*f(n=i) == B2[i] for i in range(100))
                True
                sage: B.compatible_operators() == B2.compatible_operators()
                True
                sage: B2.get_compatibility('E')
                ((n^3 + 4*n^2 + 6*n + 4)/(n^3 + 3*n^2 + n + 3))*Sn + 1

            This scalar product also work with the other subclasses of :class:`SFactorialBasis`::

                sage: P = PowerBasis()
                sage: P2 = P.scalar(n^4+1)
                sage: P2
                Factorial basis: (1, 2*x, 17*x^2, ...)
                sage: P2.compatible_operators()
                dict_keys(['x', 'Id', 'Dx'])
                sage: all(P[i]*(i^4+1) == P2[i] for i in range(100))
                True

            It is interesting to remark that the output class is :class:`SFactorialBasis`, loosing the 
            information of how the basic basis was created. However, as seen before, we keep all
            compatibilities::

                sage: P2.__class__ == P.__class__
                False
        '''
        factor = self.OB()(factor); n = self.n()
        to_mult = factor(n=n)/factor(n=self.n()-1)
        return SFactorialBasis(self.__an*to_mult, self.__bn*to_mult, X=self.var_name(), init=self[0]*factor(n=0))
        
    def _scalar_hypergeometric(self, factor, quotient):
        r'''
            Method that actually builds the structure for the new basis.

            This method *overrides* the corresponding abstract method from :class:`psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.scalar` for further information.

            TODO: add examples
        '''
        return SFactorialBasis(self.__an*quotient(n=self.n()-1), self.__bn*quotient(n=self.n()-1), X=self.var_name(), init=self[0]*factor(n=0))

    def __repr__(self):
        return "Factorial basis: (%s, %s, %s, ...)" %(self[0],self[1],self[2])

    def _latex_(self):
        return r"Factorial basis \left(%s,%s\right): \left\{%s,%s,%s,\ldots\right\}" %(latex(self.__an), latex(self.__bn), latex(self[0]), latex(self[1]), latex(self[2]))

    def root_sequence(self):
        r'''
            Method that returns the root sequence of the polynomial basis.

            This method *overrides* the implementation from class :class:`FactorialBasis`. See :func:`FactorialBasis.root_sequence`
            for a description on the output.

            In this case, as the basis is built from the first order recurrence:
            
            .. MATH::
            
                P_n(x) = (a_nx + b_n)P_{n-1}(x),

            we can explicitly build the new root added in `P_{n+1}(x)` by simply taking
            `-b_{n+1}/a_{n+1}`.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = SFactorialBasis(1,0); roots = B.root_sequence()
                sage: all(roots(i) == 0 for i in range(100))
                True
                sage: n = B.n()
                sage: B2 = SFactorialBasis(n+1, n-1); roots = B2.root_sequence()
                sage: [roots(i) for i in range(7)]
                [0, -1/3, -1/2, -3/5, -2/3, -5/7, -3/4]
        '''
        return lambda n : -self.__bn(n=n+1)/self.__an(n=n+1)

    def constant_coefficient(self):
        r'''
            Getter for the constant coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.constant_coefficient` for further information 
            in the description or the output.

            EXAMPLES::

                sage: from psbasis import *
                sage: SFactorialBasis(1,0).constant_coefficient()
                0
                sage: SFactorialBasis(2,1).constant_coefficient()
                1
                sage: SFactorialBasis(1, '(n^2 - 3)/(n+1)').constant_coefficient()
                (n^2 - 3)/(n + 1)

            This class also allows to access this value with the property :attr:`~SFactorialBasis.bn`::

                sage: SFactorialBasis(1,0).bn
                0
                sage: SFactorialBasis(2,1).bn
                1
                sage: SFactorialBasis(1, '(n^2 - 3)/(n+1)').bn
                (n^2 - 3)/(n + 1)
        '''
        return self.__bn

    def linear_coefficient(self):
        r'''
            Getter for the linear coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.linear_coefficient` for further information 
            in the description or the output.
            
            This method returns the value of `a_n`.

            EXAMPLES::

                sage: from psbasis import *
                sage: SFactorialBasis(1,0).linear_coefficient()
                1
                sage: SFactorialBasis(2,1).linear_coefficient()
                2
                sage: SFactorialBasis(1, '(n^2 - 3)/(n+1)').linear_coefficient()
                1

            This class also allows to access this value with the property :attr:`~SFactorialBasis.an`::

                sage: SFactorialBasis(1,0).an
                1
                sage: SFactorialBasis(2,1).an
                2
                sage: SFactorialBasis(1, '(n^2 - 3)/(n+1)').an
                1
        '''
        return self.__an

    def increasing_polynomial(self, src, diff=None, dst=None):
        r'''
            Returns the increasing factorial for the factorial basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.increasing_polynomial` for further information 
            in the description or the output.

            INPUT:

            * ``src``: value for lowest index, `n`.
            * ``diff``: difference between `n` and the largest index, `m`. Must be a positive integer.
            * ``dst``: value for `m`. Only used (and required) if ``diff`` is ``None``. Must
              be bigger than `n`.

            TODO: add examples
        '''
        ## Checking the arguments
        if(((src in ZZ) and src < 0) or (not src in self.OB())):
            raise ValueError("The argument `src` must be a expression involving `self.n()` or a positive integer")
        n = src

        if(not diff is None):
            if((not diff in ZZ) or diff < 0):
                raise ValueError("The argument `diff` must be None or a positive integer")
            else:

                d = ZZ(diff); m = n + d
        else:
            if(n in ZZ):
                if((not dst in ZZ) or dst < n):
                    raise ValueError("The argument `dst` must be an integer bigger than `src`")
                m = ZZ(dst); d = m - n
            else:
                d = dst-n
                if((not d in ZZ) or d < 0):
                    raise ValueError("The difference between `dst` and `src` must be a positive integer")
                d = ZZ(d); m = dst

        ## Building the polynomial
        PR = self.polynomial_ring(self.var_name()); x = PR.gens()[0]
        if(d == 0):
            return PR.one()

        if(not (n,d) in self.__cached_increasing):
            n_name = str(self.n())

            self.__cached_increasing[(n,d)] = prod(self.bn(**{n_name : n+i}) + x*self.an(**{n_name : n+i}) for i in range(1,d+1))

        return self.__cached_increasing[(n,d)]

    @cached_method
    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            TODO: add examples
        '''
        ## Checking the arguments
        if((shift in ZZ) and shift < 0):
            raise ValueError("The argument `shift` must be a positive integer")
        n = self.n()
        return SFactorialBasis(self.an(n=n+shift),self.bn(n=n+shift), X=self.var_name())

    # FactorialBasis abstract method
    @cached_method
    def matrix_ItP(self, src, size):
        r'''
            Method to get the matrix for converting from the increasing basis to the power basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.matrix_ItP`.
            
            INPUT:
                - ``src``: value for `n`.
                - ``size``: bound on the degree for computing the matrix.
        '''
        if(((src in ZZ) and src < 0) or (not src in self.OB())):
            raise ValueError("The argument `src` must be a expression involving `self.n()` or a positive integer")
        n = src
        if(n in ZZ):
            n =  ZZ(n); dest = QQ
        else:
            dest = self.OB()

        if((not size in ZZ) or size <= 0):
            raise ValueError("The argument `size` must be a positive integer")

        ## Computing the matrix
        polys = [self.increasing_polynomial(n,diff=i) for i in range(size)]
        return Matrix(dest, [[polys[j][i] for j in range(size)] for i in range(size)])

    # FactorialBasis abstract method
    def equiv_DtC(self, bound, *coeffs):
        r'''
            Method to get the equivalence condition for a compatible operator.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.equiv_DtC`.

            INPUT:

            * ``bound``: value for the lower bound for the compatibility condition (i.e., `A`).
            * ``coeffs``: list of coefficients in ``self.OB()`` representing the coefficients
              `\alpha_{n,i}`, i.e., `coeffs[j] = \alpha_{n,j-A}`.

            TODO: add examples
        '''
        ## Checking the input parameters
        if((not bound in ZZ) or (bound < 0)):
            raise ValueError("The argument `bound` must be a positive integer")
        if(len(coeffs) ==  1 and (type(coeffs) in (tuple, list))):
            coeffs = coeffs[0]
        A = ZZ(bound); B = len(coeffs) - A - 1; n = self.n()

        ## At this point we have that `coeffs` is the list of coefficients of
        ## L(P_n)/P_{n-A} in the increasing basis starting with $n-A$.
        ## We only need to change the basis to the Power Basis
        new_alpha = self.matrix_ItP(n-A, A+B+1)*vector(coeffs)

        return [el for el in new_alpha]

    # FactorialBasis abstract method
    def equiv_CtD(self, bound, *coeffs):
        r'''
            Method to get the equivalence condition for a compatible operator.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.equiv_CtD`.

            INPUT:

            * ``bound``: value for the lower bound for the compatibility (i.e., `A`).
            * ``coeffs``: list representing the coefficients of the polynomial `L \cdot P_n(x)/P_{n-A}(x)` in the canonical 
              power basis.

            TODO: add examples
        '''
        ## Checking the input parameters
        if((not bound in ZZ) or (bound < 0)):
            raise ValueError("The argument `bound` must be a positive integer")
        if(len(coeffs) ==  1 and (type(coeffs) in (tuple, list))):
            coeffs = coeffs[0]
        A = ZZ(bound); B = len(coeffs) - A - 1; n = self.n()

        ## At this point we have that `coeffs` is the list of coefficients of
        ## L(P_n)/P_{n-A} in the power basis. If we change to the increasing
        ## basis starting in $n-A$ then we have the $alpha_{n,i}$.
        new_alpha = self.matrix_PtI(n-A, A+B+1)*vector(coeffs)

        return [el for el in new_alpha]

class RootSequenceBasis(FactorialBasis):
    r'''
        Class representing a factorial basis from the root sequence and sequence of coefficients.

        A factorial basis is a type of polynomial basis for power series where
        the `(n+1)`-th element is build from the `n`-th element. This is equivalent to provide
        a sequence of roots `\rho_n` (each polynomial add the corresponding root)
        and the sequence of leading coefficients `c_n`:

        .. MATH::

            P_n(x) = c_n\prod_{i=1}^n (x-\rho_n)

        INPUT:

        * ``rho``: the sequence of roots for the factorial basis.
        * ``cn``: the sequence of leading coefficients for the factorial basis.
        * ``X``: the name for the operator representing the multiplication by `x`.

        TODO: add examples.
    '''
    def __init__(self, rho, cn, X='x'):
        ## Checking the input of leading coefficient
        if(cn in self.OB() and self.valid_factor(self.OB()(cn))):
            self.__cn = self.OB()(cn)
        elif(cn in self.OB()):
            raise ValueError("The leading coefficient sequence must be well defined and never take the value zero")
        else:
            self.__cn = cn

        ## Saving the sequence of roots
        self.__rho = rho
        
        ## Initializing the PolyBasis structure
        super(RootSequenceBasis,self).__init__(X)   

        ## Other cached elements
        self.__cached_increasing = {}

    @cached_method
    def element(self, n, var_name=None):
        r'''
            Method to return the `n`-th element of the basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.element` for further information.

            For a :class:`RootSequenceBasis` the output will be a polynomial of degree `n`.

            OUTPUT:

            A polynomial with variable name given by ``var_name`` and degree ``n``.

            TODO: add examples
        '''
        if(var_name is None):
            name = self.var_name()
        else:
            name = var_name
        R = self.polynomial_ring(name)
        x = R.gens()[0]

        if(n > 0):
            rho = self.rho; cn = self.cn
            return cn(n=n)/cn(n=n-1)*self.element(n-1, var_name=var_name) * (x - rho(n=n-1))
        elif(n == 0):
            return self.cn(n=0)

    def __repr__(self):
        return "Factorial basis: (%s, %s, %s, ...)" %(self[0],self[1],self[2])

    def _latex_(self):
        return r"Factorial basis \left(r:%s,lc:%s\right): \left\{%s,%s,%s,\ldots\right\}" %(latex(self.__rho), latex(self.__cn), latex(self[0]), latex(self[1]), latex(self[2]))

    def root_sequence(self):
        r'''
            Method that returns the root sequence of the polynomial basis.

            This method *overrides* the implementation from class :class:`FactorialBasis`. See :func:`FactorialBasis.root_sequence`
            for a description on the output.

            TODO: add examples
        '''
        return self.__rho

    def leading_coefficient(self):
        r'''
            Getter for the constant coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.leading_coefficient` for further information 
            in the description or the output.

            TODO: add examples
        '''
        return self.__cn

    def constant_coefficient(self):
        r'''
            Getter for the constant coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.constant_coefficient` for further information 
            in the description or the output.

            TODO: add examples
        '''
        cn = self.cn; rho = self.rho
        return lambda n : cn(n=n)/cn(n=n-1)*rho(n=n)

    def linear_coefficient(self):
        r'''
            Getter for the linear coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.linear_coefficient` for further information 
            in the description or the output.
            
            This method returns the value of `a_n`.

            TODO: add examples
        '''
        cn = self.cn
        return lambda n : cn(n=n)/cn(n=n-1)

    def increasing_polynomial(self, src, diff=None, dst=None):
        r'''
            Returns the increasing factorial for the factorial basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.increasing_polynomial` for further information 
            in the description or the output.

            INPUT:

            * ``src``: value for lowest index, `n`.
            * ``diff``: difference between `n` and the largest index, `m`. Must be a positive integer.
            * ``dst``: value for `m`. Only used (and required) if ``diff`` is ``None``. Must
              be bigger than `n`.

            TODO: add examples
        '''
        ## Checking the arguments
        if(((src in ZZ) and src < 0) or (not src in self.OB())):
            raise ValueError("The argument `src` must be a expression involving `self.n()` or a positive integer")
        n = src

        if(not diff is None):
            if((not diff in ZZ) or diff < 0):
                raise ValueError("The argument `diff` must be None or a positive integer")
            else:

                d = ZZ(diff); m = n + d
        else:
            if(n in ZZ):
                if((not dst in ZZ) or dst < n):
                    raise ValueError("The argument `dst` must be an integer bigger than `src`")
                m = ZZ(dst); d = m - n
            else:
                d = dst-n
                if((not d in ZZ) or d < 0):
                    raise ValueError("The difference between `dst` and `src` must be a positive integer")
                d = ZZ(d); m = dst

        ## Building the polynomial
        PR = self.polynomial_ring(self.var_name()); x = PR.gens()[0]
        if(d == 0):
            return PR.one()

        if(not (n,d) in self.__cached_increasing):
            self.__cached_increasing[(n,d)] = prod(self.bn(n=n+i) + x*self.an(n=n+i) for i in range(1,d+1))

        return self.__cached_increasing[(n,d)]

    @cached_method
    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            TODO: add examples
        '''
        ## Checking the arguments
        if((shift in ZZ) and shift < 0):
            raise ValueError("The argument `shift` must be a positive integer")
        return RootSequenceBasis(lambda n : self.rho(n=n+shift), lambda n : self.cn(n=n+shift), self.var_name())

    # FactorialBasis abstract method
    @cached_method
    def matrix_ItP(self, src, size):
        r'''
            Method to get the matrix for converting from the increasing basis to the power basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.matrix_ItP`.
            
            INPUT:

            * ``src``: value for `n`.
            * ``size``: bound on the degree for computing the matrix.

            TODO: add example
        '''
        if(((src in ZZ) and src < 0) or (not src in self.OB())):
            raise ValueError("The argument `src` must be a expression involving `self.n()` or a positive integer")
        n = src
        if(n in ZZ):
            n =  ZZ(n); dest = QQ
        else:
            dest = self.OB()

        if((not size in ZZ) or size <= 0):
            raise ValueError("The argument `size` must be a positive integer")

        ## Computing the matrix
        polys = [self.increasing_polynomial(n,diff=i) for i in range(size)]
        return Matrix(dest, [[polys[j][i] for j in range(size)] for i in range(size)])

    # FactorialBasis abstract method
    def equiv_DtC(self, bound, *coeffs):
        r'''
            Method to get the equivalence condition for a compatible operator.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.equiv_DtC`.

            INPUT:

            * ``bound``: value for the lower bound for the compatibility condition (i.e., `A`).
            * ``coeffs``: list of coefficients in ``self.OB()`` representing the coefficients
              `\alpha_{n,i}`, i.e., `coeffs[j] = \alpha_{n,j-A}`.

            TODO: add examples
        '''
        ## Checking the input parameters
        if((not bound in ZZ) or (bound < 0)):
            raise ValueError("The argument `bound` must be a positive integer")
        if(len(coeffs) ==  1 and (type(coeffs) in (tuple, list))):
            coeffs = coeffs[0]
        A = ZZ(bound); B = len(coeffs) - A - 1; n = self.n()

        ## At this point we have that `coeffs` is the list of coefficients of
        ## L(P_n)/P_{n-A} in the increasing basis starting with $n-A$.
        ## We only need to change the basis to the Power Basis
        new_alpha = self.matrix_ItP(n-A, A+B+1)*vector(coeffs)

        return [el for el in new_alpha]

    # FactorialBasis abstract method
    def equiv_CtD(self, bound, *coeffs):
        r'''
            Method to get the equivalence condition for a compatible operator.

            This method *implements* the corresponding abstract method from :class:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.equiv_CtD`.

            INPUT:

            * ``bound``: value for the lower bound for the compatibility (i.e., `A`).
            * ``coeffs``: list representing the coefficients of the polynomial `L \cdot P_n(x)/P_{n-A}(x)` in the canonical 
              power basis.

            TODO: add examples
        '''
        ## Checking the input parameters
        if((not bound in ZZ) or (bound < 0)):
            raise ValueError("The argument `bound` must be a positive integer")
        if(len(coeffs) ==  1 and (type(coeffs) in (tuple, list))):
            coeffs = coeffs[0]
        A = ZZ(bound); B = len(coeffs) - A - 1; n = self.n()

        ## At this point we have that `coeffs` is the list of coefficients of
        ## L(P_n)/P_{n-A} in the power basis. If we change to the increasing
        ## basis starting in $n-A$ then we have the $alpha_{n,i}$.
        new_alpha = self.matrix_PtI(n-A, A+B+1)*vector(coeffs)

        return [el for el in new_alpha]

###############################################################
### EXAMPLES OF PARTICULAR FACTORIAL BASIS
###############################################################
class FallingBasis(SFactorialBasis):
    r'''
        Class for the Falling factorial Basis `(1, (ax+b), (ax+b)(ax+b-c), (ax+b)(ax+b-c)(ax+b-2c), \dots))`.

        This class represent the FactorialBasis formed by the falling factorial basis
        for the power series ring `\mathbb{Q}[[x]]` with two extra paramenters `a` and `b`:

        .. MATH::

            1,\quad (ax+b),\quad (ax+b)(ax+b-c),\quad (ax+b)(ax+b-c)(ax+b-2c),\dots

        In the case of `a = 1`, `b = 0` and `c = 0`, we have the usual power basis (see class
        :class:`PowerBasis`) and in the case of `a=1`, `b = 0` and `c = \pm 1` we have the falling (or
        raising) factorial basis.

        Following the notation in :arxiv:`1804.02964v1`, these basis
        have compatibilities with the multiplication by `x` and with the isomorphism
        `E_c: x \mapsto x+c`.

        INPUT:

        * ``dilation``: the natural number corresponding to the parameter `a`.
        * ``shift``: the shift corresponding to the value `b`.
        * ``decay``: the value for `c`
        * ``X``: the name for the operator representing the multiplication by `x`. If not given, we will
          consider `x` as default.
        * ``E``: the name for the operator representing the shift of `x` by `c`. If not given, we will consider
          "Id" if `c = 0`, "E" if `c = 1` and "E_c" otherwise by default.

        TODO add examples
    '''
    def __init__(self, dilation, shift, decay, X='x', E=None):
        if(not dilation in ZZ or dilation <= 0):
            raise ValueError("The dilation of the basis ust be a natural number")
        dilation = ZZ(dilation); shift = self.OB().base()(shift); decay = self.OB().base()(decay)
        self.__a = dilation; a = self.__a
        self.__b = shift; b = self.__b
        self.__c = decay; c = self.__c

        if(E is None):
            if(c == 0):
                self.__E_name = "Id"
            if(c == 1):
                self.__E_name = "E"
            else:
                self.__E_name = "E_%s" %abs(QQ(c))
        else:
            self.__E_name = E

        n = self.n()
        super(FallingBasis, self).__init__(a, b-c*(n-1), X)

        Sn = self.Sn()
        aux_PR = self.polynomial_ring(X); x = aux_PR.gens()[0]
        aux_OE = OreAlgebra(aux_PR, (self.__E_name, lambda p : p(x=x+b), lambda p: 0))
        P = aux_OE(prod(a*x+b-c*i for i in range(-a,0)))
        self.set_compatibility(self.__E_name, self.get_compatibility(P)*(Sn**a), True)

    @cached_method
    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            This method *overrides* the corresponding method from :class:`~psbasis.factorial_basis.SFactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            In the particular case of a :class:`FallingBasis`, we had parameters `a`, `b` and `c` that defines
            the leading coefficients, the shift and the dilation of the falling basis. Hence the `N`-th increasing
            basis is a new :class:`FallingBasis` with parameters `a`, `b-Nc` and `c`.

            TODO: add examples
        '''
        if((shift in ZZ) and shift < 0):
            raise ValueError("The argument `shift` must be a positive integer")

        return FallingBasis(self.__a, self.__b - shift*self.__c, self.__c, self.var_name(),self.__E_name)

    def __repr__(self):
        a = self.__a; b = self.__b; c = self.__c
        if(c == -1):
            return "Raising Factorial Basis (1, %s, %s(%s+1),...)" %(self.element(1), self.element(1), self.element(1))
        elif(c == 1):
            return "Falling Factorial Basis (1, %s, %s(%s-1),...)" %(self.element(1), self.element(1), self.element(1))
        return "General (%s,%s,%s)-Falling Factorial Basis (%s, %s,%s,...)" %(a,b,c,self.element(0), self.element(1), self.element(2))

    def _latex_(self):
        a = self.__a; b = self.__b; c = self.__c
        x = self.polynomial_ring(self.var_name()).gens()[0]
        if(c == -1):
            return r"\left\{(%s)^{\overline{n}}\right\}_{n \geq 0}" %self.element(1)
        elif(c == 1):
            return r"\left\{(%s)^{\underline{n}}\right\}_{n \geq 0}" %self.element(1)
        return r"\left\{(%s)^{\underline{n}_%s}\right\}_{n \geq 0}" %(a*x, b)

class PowerBasis(FallingBasis):
    r'''
        Class for the Power Basis `(1,x,x^2,\dots)`.

        This class represents the :class:`FactorialBasis` formed by the simplest basis
        for the power series: `1`, `(ax+b)`, `(ax+b)^2`, etc.

        Following the notation in :arxiv:`1804.02964v1`, this basis
        corresponds with $\mathfrak{P}_{a,b}$. In that paper we can find that these basis
        have compatibilities with the multiplication by `x` and with the derivation
        with respect to `x`.

        INPUT:

        * ``dilation``: the natural number corresponding to the value `a`.
        * ``shift``: the shift corresponding to the value `b`.
        * ``X``: the name for the operator representing the multiplication by `x`. If not given, we will
          consider `x` as default.
        * ``Dx``: the name for the operator representing the derivation by `x`. If not given, we will
          consider `Dx` as default.

        TODO: add examples
    '''
    def __init__(self, dilation=1, shift=0, X='x', Dx='Dx'):
        super(PowerBasis, self).__init__(dilation,shift,0,X,'Id')

        n = self.n(); Sn = self.Sn(); a = self.linear_coefficient()
        self.set_compatibility(Dx, a*(n+1)*Sn)

    @cached_method
    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            This method *overrides* the corresponding method from :class:`~psbasis.factorial_basis.SFactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            In the particular case of :class:`PowerBasis`, the `n`-th increasing basis is always 
            equal to itself.

            TODO: add examples
        '''
        if((shift in ZZ) and shift < 0):
            raise ValueError("The argument `shift` must be a positive integer")

        return self

    def __repr__(self):
        a = self.linear_coefficient(); b = self.constant_coefficient()
        if(a == 1 and b == 0):
            return "Power Basis %s^n" %self.element(1)
        else:
            return "(%s,%s)-Power Basis (%s)^n" %(a,b,self.element(1))

    def _latex_(self):
        a = self.linear_coefficient(); b = self.constant_coefficient()
        if(a == 1 and b == 0):
            return r"\left\{%s^n\right\}_{n \geq 0}" %self.element(1)
        else:
            return r"\left\{(%s)^n\right\}_{n \geq 0}" %self.element(1)

class BinomialBasis(SFactorialBasis):
    r'''
        Class for the generic binomial basis.

        This class represents a binomial basis with a shift and dilation effect on the
        top variable. Namely, a basis of the form

        .. MATH::

            \binom{ax+b}{n},

        where `a` is a natural number and `b` is a rational number.

        In :arxiv:`1804.02964v1` this corresponds to $\mathfrak{C}_{a,b}$
        and it is compatible with the multiplication by `x` and by the shift operator
        `E: x \rightarrow x+1`.

        INPUT:

        * ``dilation``: the natural number corresponding to the value `a`.
        * ``shift``: the shift corresponding to the value `b`.
        * ``X``: the name for the operator representing the multiplication by $x$. If not given, we will
          consider `x` as default.
        * ``E``: the name for the operator representing the shift of $x$ by `1`. If not given, we will
          consider `E` as default.
    '''
    def __init__(self, dilation=1, shift=0, X='x', E='E'):
        if(not dilation in ZZ or dilation <= 0):
            raise ValueError("The dilation of the basis ust be a natural number")
        dilation = ZZ(dilation); shift = self.OB().base()(shift)
        self.__a = dilation; a = self.__a
        self.__b = shift; b = self.__b

        n = self.n()
        super(BinomialBasis, self).__init__(a/n, (b-n + 1)/n, X)

        Sn = self.Sn()
        self.set_compatibility(E, sum(binomial(a, i)*Sn**i for i in range(a+1)))

    def __repr__(self):
        x = self.polynomial_ring(self.var_name()).gens()[0]
        return "Binomial basis (%s) choose n" %(self.__a*x + self.__b)

    def _latex_(self):
        x = self.polynomial_ring(self.var_name()).gens()[0]
        return r"\left\{\binom{%s}{n}\right\}_{n\geq 0}" %(self.__a*x+self.__b)

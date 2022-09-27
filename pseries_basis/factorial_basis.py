r'''
    Sage package for Factorial Series Basis.
'''
# sage imports
from sage.all import prod, vector, ZZ, cached_method, QQ, Matrix, latex

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

        * :func:`pseries_basis.psbasis.PSBasis.element`.
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

            This method *overrides* the corresponding abstract method from :class:`pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.scalar` for further information.

            EXAMPLES::

                sage: from pseries_basis import *
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
            :class:`FactorialBasis`. However, if this method is not overridden, then the standard output
            is a :class:`ScalarBasis`::

                sage: B = ProductBasis([BinomialBasis(), PowerBasis()])
                sage: f = (n+1)
                sage: B2 = B.scalar(f)
                sage: all(B[i]*f(n=i) == B2[i] for i in range(100))
                True
                sage: isinstance(B2, ScalarBasis)
                True
        '''
        return ScalarBasis(self, factor)
        
    def _scalar_hypergeometric(self, factor, quotient):
        r'''
            Method that actually builds the structure for the new basis.

            This method *overrides* the corresponding abstract method from :class:`pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.scalar` for further information.

            EXAMPLES::

                sage: from pseries_basis import *
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
            :class:`FactorialBasis`. However, if this method is not overridden, then the standard output
            is a :class:`ScalarBasis`::

                sage: B = ProductBasis([BinomialBasis(), PowerBasis()])
                sage: B3 = B.scalar(factorial(n))
                sage: all(B[i]*factorial(i) == B3[i] for i in range(100))
                True
                sage: isinstance(B3, ScalarBasis)
                True
        '''
        return ScalarBasis(self, factor)

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

                sage: from pseries_basis import *
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

    def compatible_division(self, operator):
        r'''
            Method to get the division of a polynomial by other element of the basis after an operator.

            It was proven in :arxiv:`2202.05550` that if `L` is an `(A,B)`-compatible operator
            with a factorial basis, then for any `n \in \mathbb{N}`, we have

            .. MATH::

                P_{n-A}(x) | L\cdot P_n(x).

            Moreover, since the basis is factorial, it is clear that for any `m < n-A`, we have
            that `P_m(x)` also divides `L\cdot P_n(x)`. See method :func:`increasing_polynomial`
            for further information.

            Let `m` be the number of sections defining the compatibility of `L` with ``self``. 
            This method returns a function `D_{r,s}(n)` with 3 arguments such that

            .. MATH::

                D_{r,s}(n) = \frac{L\cdot P_{nm+r}(x)}{P_{nm+r-A-s}}

            INPUT:

            * ``operator``: the operator we want to check. See the input description
              of method :func:`recurrence`. This operator has to be compatible,
              so we can obtain the value for `A`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = BinomialBasis(); n = B.n(); B.recurrence('x')
                n*Sni + n

            This means that for all `n`, we have that:

            .. MATH::

                xB_n(x) = (n+1)B_{n+1}(x) + nB_n(x).

            Hence, we have that `A = 0` and `B_m(x)` divides `xB_n(x)` for any `m \leq n`::

                sage: B.compatible_division('x')[2](0, 2, 5) # xB_5(x) / B_3(x)
                1/20*x^3 - 7/20*x^2 + 3/5*x
                sage: x = B[1].parent().gens()[0]
                sage: all(all(x*B[i] / B[j] == B.compatible_division('x')[2](0,i-j,i) for j in range(0,i)) for i in range(50))
                True
            
            We can also check other operators like the shift `x \mapsto x+1`. Here `A = 1`::

                sage: A = B.A('E')
                sage: all(all(B[i](x=x+1) / B[j] == B.compatible_division('E')[2](0,i-A-j,i) for j in range(0,i-1)) for i in range(50))
                True
                sage: B.compatible_division('E')[2](0,1,n)
                (1/(n^2 - n))*x^2 + ((-n + 3)/(n^2 - n))*x + (-n + 2)/(n^2 - n)

            Also we can check other types of basis::

                sage: PowerBasis().compatible_division('Dx')[2](0,4,10)
                10*x^4
                sage: PowerBasis().compatible_division('x')[2](0,10,30)
                x^11
                sage: n = PowerBasis().n()
                sage: PowerBasis().compatible_division('x')[2](0,3,n)
                x^4

            TODO: add more examples with some sectioned compatibility
        '''
        ## Compatibility of ``operator``
        A, B, m, alpha = self.compatibility(operator)
        
        return (A, m, lambda r,s,n : sum(alpha(r,i,n)*self.increasing_polynomial(n*m+r-A-s,dst=n*m+r+i) for i in range(-A,B+1)))

    def matrix_ItP(self, src, size):
        r'''
            Method to get the matrix for converting from the increasing basis to the power basis.

            In a Factorial Basis, the `n`-th element of the basis divides all the following.
            This means for any pair of indices `m > n`, there is a particular polynomial
            `Q_{n,m}(x) = P_m(x)/P_n(x)` (see method :func:`increasing_polynomial` for further
            information).

            In particular, for a fixed `n` and `i \in \mathbb{N}`, the polynomials `Q_{n,n+i}(x)`
            are another Factorial Basis (see method :func:`increasing_basis`). This method 
            computes a matrix that represents the identity map between polynomials of degree smaller 
            or equal to a given size from the basis `Q_{n,n+i}(x)` and the canonical power basis.

            INPUT:

            * ``src``: value for `n`.
            * ``size``: bound on the degree for computing the matrix.

            OUTPUT:

            A matrix that convert coordinates from the increasing basis to the canonical power basis.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: n = PSBasis.n(None); B = SFactorialBasis(n+1, 1/n)
                sage: BinomialBasis().matrix_ItP(n,3)
                [                         1                 -n/(n + 1)                  n/(n + 2)]
                [                         0                  1/(n + 1) (-2*n - 1)/(n^2 + 3*n + 2)]
                [                         0                          0          1/(n^2 + 3*n + 2)]
                sage: B.matrix_ItP(2,6)
                [        1       1/3      1/12      1/60     1/360    1/2520]
                [        0         4       8/3     31/30     13/45      4/63]
                [        0         0        20        20    317/30  2407/630]
                [        0         0         0       120       160 11276/105]
                [        0         0         0         0       840      1400]
                [        0         0         0         0         0      6720]

            The increasing polynomials have as coordinates in the canonical basis the columns of this matrix::

                sage: B.increasing_polynomial(2,3)
                120*x^3 + 20*x^2 + 31/30*x + 1/60
                sage: B.increasing_polynomial(2,5)
                6720*x^5 + 1400*x^4 + 11276/105*x^3 + 2407/630*x^2 + 4/63*x + 1/2520
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

    @cached_method
    def matrix_PtI(self, src, size):
        r'''
            Method for getting the matrix from the power basis to the increasing basis.

            In a Factorial Basis, the `n`-th element of the basis divides all the following.
            This means for any pair of indices `m > n`, there is a particular polynomial
            `Q_{n,m} = P_m/P_n`.

            In particular, for a fixed `n` and `i \in \mathbb{Z}`, the polynomials `Q_{n,n+i}`
            form another Factorial Basis. This method computes a matrix that represents the
            identity between polynomials of degree smaller or equal to ``size`` from the
            power basis to the basis `Q_{n,n+i}`.

            INPUT:

            * ``src``: value for `n`.
            * ``size``: bound on the degree for computing the matrix.

            OUTPUT:

            A matrix that convert coordinates from the canonical power basis to the requested increasing basis. 

            EXAMPLES::

                sage: from pseries_basis import *
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

            This is equivalent to compute the basis matrix (see method :func:`~pseries_basis.psbasis.PSBasis.functional_matrix`) of the 
            increasing basis (see method :func:`increasing_basis`). However, due to the different notation between these two
            methods, the relation is with the transposed matrix::

                sage: B.increasing_basis(3).functional_matrix(5).inverse().transpose() == B.matrix_PtI(3,5)
                True
        '''
        return self.matrix_ItP(src,size).inverse()

    def equiv_DtC(self, compatibility):
        r'''
            Method to get the equivalence condition for a compatible operator.

            Following the notation and ideas of :arxiv:`2202.05550`, there is an
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

            If we name `I_{n-A,m}(x) = P_{n-A+m}/P_{n-A}`,i.e., the `n-A`-th increasing basis of 
            this basis (see method :func:`increasing_basis` and :func:`increasing_polynomial`) then
            it is clear that we can write:

            .. MATH::

                \frac{L\cdot P_{n}(x)}{P_{n-A}(x)} = \sum_{i = -A}^B \alpha_{i}(n) \frac{P_{n+j}(x)}{P_{n-A}(x)} = 
                \sum_{i=0}^{A+B+1} \alpha_{i-A}(n)I_{n-A,i}.

            Hence the output of this method can be computed after changing coordinates from the increasing
            basis to the canonical basis (see method :func:`matrix_ItP`).

            INPUT:
            
            * ``compatibility``: the compatibility condition as a tuple `(A,B,m,\alpha_{i,j,k})` where
              `A` and `B` are the compatibility bounds, `m` is the number of sections in which the 
              compatibility is considered and `\alpha_{i,j,k} \in \mathbb{K}(k)` for `i \in \{0,\ldots,m-1\}
              and `j \in \{-A,\ldots,B\}`. We also allow here a compatible operator with ``self``.

            OUTPUT:

            A tuple of the form `(A, B, m, c_{i,j}(n))` where `A` is the lower bound for the compatibility
            condition, `B` is the upper bound, `m` is the number of sections considered and `c_{i,j}(k)` 
            is a function with three arguments `i \in \{0,\ldots,m-1\}`, `j \in \{0,\ldots,A+B+1}` 
            and `k` such that, for `n = km+r`:

            .. MATH::

                \frac{L\cdot P_n(x)}{P_{n-A}(x)} = \sum_{j=0}^{A+B+1} c_{r,j}(k)x^j

            TODO: add examples
        '''
        if(not type(compatibility) in (tuple, list)):
            compatibility = self.compatibility(compatibility)

        A,B,m,alpha = compatibility; n = self.n()
        ItP = [self.matrix_ItP(m*n+i-A, A+B+1) for i in range(m)]
        coeffs = [ItP[i]*vector([alpha(i,j,n) for j in range(-A,B+1)])for i in range(m)]

        return (A, B, m, lambda i,j,k: coeffs[i][j](n=k))

    def equiv_CtD(self, division):
        r'''
            Method to get the equivalence condition for a compatible operator.

            Following the notation and ideas of :arxiv:`2202.05550`, there is an
            equivalent condition to be a compatible operator. Namely, and operator is compatible
            by definition if it expands:

            .. MATH::

                L \cdot P_n(x) = \sum_{i=-A}^B \alpha_{n,i}P_{n+i},

            and that is equivalent to the following two conditions:

            * `deg(L\cdot P_n(x))) \leq n + B`
            * `P_{n-A}(x)` divides `L \cdot P_n(x)`.

            This method takes coefficients of the division `\frac{L\cdot P_n(x)}{P_{n-A}(x)}` and computes
            the compatibility condition for the corresponding operator `L`.

            If we name `I_{n-A,m}(x) = P_{n-A+m}/P_{n-A}`,i.e., the `n-A`-th increasing basis of 
            this basis (see method :func:`increasing_basis` and :func:`increasing_polynomial`) then
            it is clear that we can write:

            .. MATH::

                \frac{L\cdot P_{n}(x)}{P_{n-A}(x)} = \sum_{i = -A}^B \alpha_{i}(n) \frac{P_{n+j}(x)}{P_{n-A}(x)} = 
                \sum_{i=0}^{A+B+1} \alpha_{i-A}(n)I_{n-A,i}.

            Hence the output of this method can be computed after changing coordinates from the canonical
            basis to the increasing basis (see method :func:`matrix_PtI`).

            INPUT:
            
            * ``division``: the input representing the division `\frac{L\cdot P_n(x)}{P_{n-A}(x)}`. This must 
              be a tuple or list with at least 4 elements `(A, B, m, c_{i,j}(n))` where `A, B, m\in \mathbb{N}`,
              and `c_{i,j}(n)` isa method with three arguments `i,j,n`.

            OUTPUT:

            A tuple of the form `(A, B, m, \alpha_{i,j}(n))` where `A` is the lower bound for the compatibility
            condition, `B` is the upper bound, `m` is the number of sections considered and `\alpha_{i,j}(k)` 
            is a function with three arguments `i \in \{0,\ldots,m-1\}`, `j \in \{0,\ldots,A+B+1}` 
            and `k` such that, for `n = km+r`:

            .. MATH::

                L\cdot P_n(x) = \sum_{j=-A}^{B} \alpha_{r,j}(k)P_{n+j}(x)

            TODO: add examples
        '''
        if(not type(division) in (tuple, list)):
            raise TypeError("The division condition must be given in the proper format")

        A,B,m,c = division; n = self.n()
        PtI = [self.matrix_PtI(m*n+i-A, A+B+1) for i in range(m)]
        coeffs = [PtI[i]*vector([c(i,j,n) for j in range(0,A+B+1)])for i in range(m)]

        return (A, B, m, lambda i,j,k: coeffs[i][j+A](n=k))

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

            P_n(x) = (a_nx + b_n)P_{n-1}(x)

        INPUT:
        
        * ``an``: the sequence of leading coefficients to build the factorial basis.
        * ``bn``: the sequence of constant coefficients to build the factorial basis.
        * ``X``: the name for the operator representing the multiplication by `x`.
        * ``init``: the value of `P_0(x)`. Must be a constant.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: B = BinomialBasis(); n = B.n()
            sage: B2 = SFactorialBasis(1/n, (-n+1)/n)
            sage: all(B[i] == B2[i] for i in range(50))
            True
            sage: B = SFactorialBasis(n, n); x = B[1].parent().gens()[0]
            sage: B[0]
            1
            sage: all(B[i] == B[i-1]*(i*x+i) for i in range(1,50))
            True
            sage: all(B.scalar(1/factorial(n))[i] == SFactorialBasis(1,1)[i] for i in range(10))
            True
            sage: SFactorialBasis(n^2/3, 1/n)
            Factorial basis: (1, 1/3*x + 1, 4/9*x^2 + 3/2*x + 1/2, ...)

        The first argument must never be zero for `n \geq 0`. Otherwise a :class:`ValueError` is raised::

            sage: SFactorialBasis(n^2 - 1, 1)
            Traceback (most recent call last):
            ...
            ValueError: The leading coefficient for the extra factor must always be non-zero
            sage: SFactorialBasis(1/(n-1), 1)
            Traceback (most recent call last):
            ...
            ValueError: The leading coefficient for the extra factor must always be non-zero
    '''
    def __init__(self, an, bn, X='x',init=1):
        
        ## Cheking the first element
        init = self.OB().base()(init)
        if(init == 0):
            raise ValueError("The first polynomial must be non-zero")
        self.__init = init

        n = self.n()

        ## Checking the argument `an`
        try:
            an = self.OB()(an(n=n))
        except TypeError: # an is not callable
            an = self.OB()(an) # original code for rational functions
        if(not self.valid_factor(an(n=n+1))): # an can not be zero for n >= 1
            raise ValueError("The leading coefficient for the extra factor must always be non-zero")
        self.__an = an

        ## Checking the argument `bn`
        try:
            bn = self.OB()(bn(n=n)); self.__bn = bn
        except TypeError: # bn is not callable
            bn = self.OB()(bn); self.__bn = bn # original code for rational functions

        ## Initializing the FactorialBasis structure
        super(SFactorialBasis,self).__init__(X)

        ## Extra cached variables
        self.__cached_increasing = {}

    @cached_method
    def element(self, n, var_name=None):
        r'''
            Method to return the `n`-th element of the basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.element` for further information.

            For a :class:`SFactorialBasis` the output will be a polynomial of degree `n`.

            OUTPUT:

            A polynomial with variable name given by ``var_name`` and degree ``n``.

            EXAMPLES::

                sage: from pseries_basis import *
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
        if(not n in ZZ):
            raise TypeError("Elements in SFactorialBasis can only be computed for integers 'n'")
        n = ZZ(n)
        if(n < 0):
            raise IndexError("A SFactorialBasis is only defined for non-negative integers")

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
        else:
            raise IndexError("The index must be a non-negative integer")

    def _scalar_basis(self, factor):
        r'''
            Method that actually builds the structure for the new basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.scalar` for further information.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: B = BinomialBasis(); n = B.n()
                sage: f = (n^2+1)/(n+2)
                sage: B2 = B.scalar(f)
                sage: B2
                Factorial basis: (1/2, 2/3*x, 5/8*x^2 - 5/8*x, ...)
                sage: all(B[i]*f(n=i) == B2[i] for i in range(100))
                True
                sage: B.compatible_operators() == B2.compatible_operators()
                True
                sage: B2.recurrence('E')
                (n^3 + 4*n^2 + 6*n + 4)/(n^3 + 3*n^2 + n + 3)*Sn + 1

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

            This method *overrides* the corresponding abstract method from :class:`pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.scalar` for further information.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: n = PSBasis.n(None); B = SFactorialBasis(n+1, 1/n)
                sage: B
                Factorial basis: (1, 2*x + 1, 6*x^2 + 4*x + 1/2, ...)
                sage: B2 = B*factorial(n)
                sage: B2[:10] == [B[i]*factorial(i) for i in range(10)]
                True
                
            In the case of a :class:`SFactorialBasis`, we always get a new :class:`SFactorialBasis` but with the
            values of :func:`~pseries_basis.factorial_basis.FactorialBasis.an` and :func:`~pseries_basis.factorial_basis.FactorialBasis.bn`
            slightly changed::

                sage: _,quot = B.is_hypergeometric(factorial(n))
                sage: B2.an == B.an*quot(n=n-1)
                True
                sage: B2.bn == B.bn*quot(n=n-1)
                True
        '''
        return SFactorialBasis(self.__an*quotient(n=self.n()-1), self.__bn*quotient(n=self.n()-1), X=self.var_name(), init=self[0]*factor(n=0))

    def __repr__(self):
        return "Factorial basis: (%s, %s, %s, ...)" %(self[0],self[1],self[2])

    def _latex_(self):
        return r"\text{Factorial basis }\left(%s,%s\right): \left\{%s,%s,%s,\ldots\right\}" %(latex(self.__an), latex(self.__bn), latex(self[0]), latex(self[1]), latex(self[2]))

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

                sage: from pseries_basis import *
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

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.constant_coefficient` for further information 
            in the description or the output.

            EXAMPLES::

                sage: from pseries_basis import *
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

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.linear_coefficient` for further information 
            in the description or the output.
            
            This method returns the value of `a_n`.

            EXAMPLES::

                sage: from pseries_basis import *
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

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_polynomial` for further information 
            in the description or the output.

            INPUT:

            * ``src``: value for lowest index, `n`.
            * ``diff``: difference between `n` and the largest index, `m`. Must be a positive integer.
            * ``dst``: value for `m`. Only used (and required) if ``diff`` is ``None``. Must
              be bigger than `n`.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: n = PSBasis.n(None)
                sage: B = SFactorialBasis(n, 1)
                sage: B.increasing_polynomial(2, 3)
                60*x^3 + 47*x^2 + 12*x + 1
                sage: B.increasing_polynomial(2, 3) == B.increasing_polynomial(2, dst=5)
                True
                sage: B.increasing_polynomial(3,0)
                1
                sage: B.increasing_polynomial(5, 2)
                42*x^2 + 13*x + 1

            The ``diff`` argument must be non-negative, or ``dst`` must be greater or equal then ``src``::

                sage: B.increasing_polynomial(2,-1)
                Traceback (most recent call last):
                ...
                ValueError: The argument `diff` must be None or a positive integer
                sage: B.increasing_polynomial(5, dst=4)
                Traceback (most recent call last):
                ...
                ValueError: The argument `dst` must be an integer bigger than `src`

            This method also accepts rational expressions for ``src`` and ``dst``, however their 
            difference must be a non-negative integer::

                sage: B.increasing_polynomial(n, 1)
                (n + 1)*x + 1
                sage: B.increasing_polynomial(n^2-3, dst=n^2)
                (n^6 - 3*n^4 + 2*n^2)*x^3 + (3*n^4 - 6*n^2 + 2)*x^2 + (3*n^2 - 3)*x + 1
                sage: B.increasing_polynomial(n/(n-1), dst=n)
                Traceback (most recent call last):
                ...
                ValueError: The difference between `dst` and `src` must be a positive integer
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

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            EXAMPLES::

                sage: from pseries_basis import *
                sage: n = PSBasis.n(None); B = SFactorialBasis(n+1, 1/n)
                sage: B2 = B.increasing_basis(5)
                sage: isinstance(B2, SFactorialBasis)
                True
                sage: B2[:10] == [B[i+5]/B[5] for i in range(10)]
                True
                sage: B2[:10] == [B.increasing_polynomial(5, i) for i in range(10)]
                True
        '''
        ## Checking the arguments
        if((shift in ZZ) and shift < 0):
            raise ValueError("The argument `shift` must be a positive integer")
        n = self.n()
        return SFactorialBasis(self.an(n=n+shift),self.bn(n=n+shift), X=self.var_name())

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

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.element` for further information.

            For a :class:`RootSequenceBasis` the output will be a polynomial of degree `n`.

            OUTPUT:

            A polynomial with variable name given by ``var_name`` and degree ``n``.

            TODO: add examples
        '''
        if(not n in ZZ):
            raise TypeError("Elements in RootSequenceBasis can only be computed for integers 'n'")
        n = ZZ(n)
        if(n < 0):
            raise IndexError("A RootSequenceBasis is only defined for non-negative integers")
        if(var_name is None):
            name = self.var_name()
        else:
            name = var_name
        R = self.polynomial_ring(name)
        x = R.gens()[0]

        if(n > 0):
            rho = self.rho; cn = self.cn
            return R(cn(n=n)/cn(n=n-1)*self.element(n-1, var_name=var_name) * (x - rho(n=n-1)))
        elif(n == 0):
            return R(self.cn(n=0))
        else:
            raise IndexError("The index must be a non-negative integer")

    def __repr__(self):
        return "Factorial basis: (%s, %s, %s, ...)" %(self[0],self[1],self[2])

    def _latex_(self):
        return r"\text{Factorial basis }\left(r:%s,lc:%s\right): \left\{%s,%s,%s,\ldots\right\}" %(latex(self.__rho), latex(self.__cn), latex(self[0]), latex(self[1]), latex(self[2]))

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

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.leading_coefficient` for further information 
            in the description or the output.

            TODO: add examples
        '''
        return self.__cn

    def constant_coefficient(self):
        r'''
            Getter for the constant coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.constant_coefficient` for further information 
            in the description or the output.

            TODO: add examples
        '''
        cn = self.cn; rho = self.rho
        return lambda n : cn(n=n)/cn(n=n-1)*rho(n=n)

    def linear_coefficient(self):
        r'''
            Getter for the linear coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.linear_coefficient` for further information 
            in the description or the output.
            
            This method returns the value of `a_n`.

            TODO: add examples
        '''
        cn = self.cn
        return lambda n : cn(n=n)/cn(n=n-1)

    def increasing_polynomial(self, src, diff=None, dst=None):
        r'''
            Returns the increasing factorial for the factorial basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_polynomial` for further information 
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

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            TODO: add examples
        '''
        ## Checking the arguments
        if((shift in ZZ) and shift < 0):
            raise ValueError("The argument `shift` must be a positive integer")
        return RootSequenceBasis(lambda n : self.rho(n=n+shift), lambda n : self.cn(n=n+shift), self.var_name())

class ScalarBasis(FactorialBasis):
    r'''
        This class represents trivially the scalar product of a Factorial basis.

        This class allows the user to trivially extend the functionality of a 
        :class:`FactorialBasis` after a scalar product with a sequence of coefficients.
        This class can be seen as a wrapper, since it will invoke the methods of 
        the original basis as often as possible adjusting the result with the corresponding
        element of the sequence for leading coefficients.

        This is also the default class obtained by a scalar product in :class:`FactorialBasis`.

        INPUT:

        * ``basis``: the original basis.
        * ``scale``: a sequence of coefficients `(c_n)` valid as a scalar factor. It has to be 
          a hypergeometric sequence.

        TODO: add examples
    '''
    def __init__(self, basis, scale):
        ## Checking the scaling factor
        hyper, quot = self.is_hypergeometric(scale)
        if(not (hyper and self.valid_factor(quot))):
            raise TypeError("The scaling factor is not valid")
        self.__scale = scale
        self.__quot = quot
        
        if(not isinstance(basis, FactorialBasis)):
            raise TypeError("The basis to scale must be a Factorial basis")

        super().__init__(basis.var_name())
        self.__basis = basis
        self.__scale = scale
        self.__cached_increasing = {}

    basis = property(lambda self: self.__basis) #: Property for getting the original basis
    scale = property(lambda self: self.__scale) #: Property for getting the scaling factor
    quot = property(lambda self: self.__quot) #: Property for getting the quotient of the scaling factor

    @cached_method
    def element(self, n, var_name=None):
        r'''
            Method to return the `n`-th element of the basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.element` for further information.

            For a :class:`ScalarBasis` the output will be a polynomial of degree `n` and can be directly
            construct from the original basis.

            OUTPUT:

            A polynomial with variable name given by ``var_name`` and degree ``n``.

            TODO: add examples
        '''
        return self.basis.element(n,var_name)*self.OB().base_ring()(self.scale(n=n))

    def __repr__(self):
        return "Scalar product of [%s] by [%s] (%s, %s, %s,...)" %(self.basis, self.scale, self[0], self[1], self[2])

    def _latex_(self):
        return r"\left(%s\right)_n \cdot \left(%s\right)" %(latex(self.scale), latex(self.basis))

    def root_sequence(self):
        r'''
            Method that returns the root sequence of the polynomial basis.

            This method *overrides* the implementation from class :class:`FactorialBasis`. See :func:`FactorialBasis.root_sequence`
            for a description on the output.

            In this case, as the basis is built from the original basis, since the root sequence does
            not change.

            TODO: add examples
        '''
        return self.basis.rho

    def leading_coefficient(self):
        r'''
            Method that returns the sequence of leading coefficients for a factorial basis.

            This method *overrides* the implementation from class :class:`FactorialBasis`. See :func:`FactorialBasis.leading_coefficient`
            for a description on the output.

            In this case, the leading coefficient is the original times the element in the scaling sequence.

            TODO: add examples
        '''
        return lambda n : self.basis.cn(n=n)*self.scale(n=n)

    def constant_coefficient(self):
        r'''
            Getter for the constant coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.constant_coefficient` for further information 
            in the description or the output.

            In a :class:`ScalarBasis` this value is proportional to the original value, since:

            .. MATH::

                P_{n}(x) = c_{n}Q_{n}(x) = c_nQ_{n-1}(x)(a_nx + b_n) = 
                P_{n-1}(x)\left(\frac{c_na_n}{c_{n+1}}x + \frac{c_nb_n}{c_{n+1}}\right)

        '''
        return lambda n : self.basis.bn(n=n)*self.quot(n=n-1)

    def linear_coefficient(self):
        r'''
            Getter for the linear coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.linear_coefficient` for further information 
            in the description or the output.
            
            In a :class:`ScalarBasis` this value is proportional to the original value, since:

            .. MATH::

                P_{n}(x) = c_{n}Q_{n}(x) = c_nQ_{n-1}(x)(a_nx + b_n) = 
                P_{n-1}(x)\left(\frac{c_na_n}{c_{n+1}}x + \frac{c_nb_n}{c_{n+1}}\right)

            TODO: add examples
        '''
        return lambda n : self.basis.an(n=n)*self.quot(n=n-1)

    def increasing_polynomial(self, src, diff=None, dst=None):
        r'''
            Returns the increasing factorial for the factorial basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_polynomial` for further information 
            in the description or the output.

            For a :class:`ScalarBasis`, this polynomial can be computed using the increasing polynomial
            of the original basis with an extra factor that depends on the space between ``scr`` and ``dst``
            (i.e., the value of ``diff``). Namely:

            .. MATH::

                I_{n,d}^{(P)}(x) = \frac{P_{n+d}(x)}{P_{n}(x)} = \frac{c_{n+d}}{c_n}I_{n,d}^{(Q)}(x)

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
        PR = self.polynomial_ring(self.var_name())
        if(d == 0):
            return PR.one()

        if(not (n,d) in self.__cached_increasing):
            if(self.valid_factor(self.scale)): # rational case
                self.__cached_increasing[(n,d)] = self.basis.increasing_polynomial(n,diff=d)*self.scale(n=n+d)/self.scale(n=n)
            else: # hypergeometric case
                self.__cached_increasing[(n,d)] = self.basis.increasing_polynomial(n,diff=d)*prod(self.quot(n=n+i) for i in range(d))

        return self.__cached_increasing[(n,d)]
        
    @cached_method
    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            For a :class:`ScalarBasis`, this is again a :class:`ScalarBasis` where the scaling factor is the shifted 
            version of the original scaling divided by the element of the original shift:

            .. MATH::

                \mathcal{I}_m^{(P)} = \left(\frac{c_{m+n}}{c_m}\right)_n * \mathcal{I}_m^{(Q)}


            TODO: add examples
        '''
        new_scale = lambda n : self.scale(shift+n)/self.scale(shift)

        return ScalarBasis(self.basis.increasing_basis(shift), new_scale)

    def is_quasi_eval_triangular(self):
        return self.basis.is_quasi_eval_triangular()

    def is_quasi_func_triangular(self):
        return self.basis.is_quasi_func_triangular()   
###############################################################
### EXAMPLES OF PARTICULAR FACTORIAL BASIS
###############################################################
class FallingBasis(SFactorialBasis):
    r'''
        Class for a Falling factorial Basis.

        This class represent the FactorialBasis formed by the falling factorial basis
        for the power series ring `\mathbb{Q}[[x]]` with two extra paramenters `a` and `b`:

        .. MATH::

            1,\quad (ax+b),\quad (ax+b)(ax+b-c),\quad (ax+b)(ax+b-c)(ax+b-2c),\dots

        In the case of `a = 1`, `b = 0` and `c = 0`, we have the usual power basis (see class
        :class:`PowerBasis`) and in the case of `a=1`, `b = 0` and `c = \pm 1` we have the falling (or
        raising) factorial basis.

        Following the notation in :arxiv:`2202.05550`, these basis
        have compatibilities with the multiplication by `x` and with the isomorphism
        `E: x \mapsto x+\frac{c}{a}`. All other compatible shifts (i.e., 
        maps `E_{\alpha}: x \mapsto x+\alpha)` are just powers of `E`.

        INPUT:

        * ``dilation``: the natural number corresponding to the parameter `a`.
        * ``shift``: the shift corresponding to the value `b`.
        * ``decay``: the value for `c`
        * ``X``: the name for the operator representing the multiplication by `x`. If not given, we will
          consider `x` as default.
        * ``E``: the name for the operator representing the shift of `x` by `c/a`. If not given, we will 
          consider "E" as default.

        TODO check the compatibility with shifts using the roots. Is there a generator of all compatibilities?
    '''
    def __init__(self, dilation, shift, decay, X='x', E=None):
        if(not dilation in ZZ or dilation <= 0):
            raise ValueError("The dilation of the basis ust be a natural number")
        dilation = ZZ(dilation); shift = self.OB().base()(shift); decay = self.OB().base()(decay)
        self.__a = dilation; a = self.__a
        self.__b = shift; b = self.__b
        self.__c = decay; c = self.__c

        self.__E_name = E if E != None else 'E'

        n = self.n()
        super(FallingBasis, self).__init__(a, b-c*(n-1), X)

        Sn = self.Sn(); x = self[1].parent().gens()[0]
        P = a*x+b+c
        self.set_compatibility(self.__E_name, self.recurrence(P)*Sn, True)

    @cached_method
    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.SFactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            In the particular case of a :class:`FallingBasis`, we had parameters `a`, `b` and `c` that defines
            the leading coefficients, the shift and the dilation of the falling basis. Hence the `N`-th increasing
            basis is a new :class:`FallingBasis` with parameters `a`, `b-Nc` and `c`.

            TODO: add examples
        '''
        if((shift in ZZ) and shift < 0):
            raise ValueError("The argument `shift` must be a positive integer")

        return FallingBasis(self.__a, self.__b - shift*self.__c, self.__c, self.var_name(),self.__E_name)

    def shift(self):
        r'''
            Method that returns the action of the compatible shift over the main variable.

            Any :class:`FallingFactorial` is compatible with a shift `E` that maps `x \mapsto x+ \alpha`
            and generates all possible compatible shifts with ``self``. This method returns the 
            action of the value `x+\alpha` associated with this basis.

            OUTPUT:

            The action of the main shift over the main variable.
        '''
        x = self[1].parent().gens()[0]
        return x+self.__c/self.__a

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
        return r"\left\{(%s)^{\underline{n}_{(%s)}}\right\}_{n \geq 0}" %(a*x+b, c)

class PowerBasis(FallingBasis):
    r'''
        Class for the Power Basis `(1,x,x^2,\dots)`.

        This class represents the :class:`FactorialBasis` formed by the simplest basis
        for the power series: `1`, `(ax+b)`, `(ax+b)^2`, etc.

        Following the notation in :arxiv:`2202.05550`, this basis
        corresponds with `\mathfrak{P}_{a,b}`. In that paper we can find that these basis
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

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.SFactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_basis` for further information.

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

    def is_quasi_func_triangular(self):
        return self.shift == 0

class BinomialBasis(SFactorialBasis):
    r'''
        Class for the generic binomial basis.

        This class represents a binomial basis with a shift and dilation effect on the
        top variable. Namely, a basis of the form

        .. MATH::

            \binom{ax+b}{n},

        where `a` is a natural number and `b` is a rational number.

        In :arxiv:`2202.05550` this corresponds to `\mathfrak{C}_{a,b}`
        and it is compatible with the multiplication by `x` and by the shift operator
        `E: x \rightarrow x+1`.

        INPUT:

        * ``dilation``: the natural number corresponding to the value `a`.
        * ``shift``: the shift corresponding to the value `b`.
        * ``X``: the name for the operator representing the multiplication by `x`. If not given, we will
          consider `x` as default.
        * ``E``: the name for the operator representing the shift of `x` by `1`. If not given, we will
          consider `E` as default. The operator of shift by `1/a` will be named by adding a `t` to the name.
    '''
    def __init__(self, dilation=1, shift=0, X='x', E='E'):
        if(not dilation in ZZ or dilation <= 0):
            raise ValueError("The dilation of the basis must be a natural number")
        dilation = ZZ(dilation); shift = self.OB().base()(shift)
        self.__a = dilation; a = self.__a
        self.__b = shift; b = self.__b

        n = self.n()
        super(BinomialBasis, self).__init__(a/n, (b-n + 1)/n, X)

        Sn = self.Sn()
        ## Adding the compatibility by $x \mapsto x+1/a$:
        self.set_compatibility(E+'t', Sn+1)
        ## Adding the compatibility by $x \mapsto x+1$ (simply powering the previous compatibility)
        self.set_compatibility(E, (Sn+1)**a)

    def __repr__(self):
        x = self.polynomial_ring(self.var_name()).gens()[0]
        return "Binomial basis (%s) choose n" %(self.__a*x + self.__b)

    def _latex_(self):
        x = self.polynomial_ring(self.var_name()).gens()[0]
        return r"\left\{\binom{%s}{n}\right\}_{n\geq 0}" %(self.__a*x+self.__b)

    def is_quasi_eval_triangular(self):
        return self.__b in ZZ and self.__b >= 0

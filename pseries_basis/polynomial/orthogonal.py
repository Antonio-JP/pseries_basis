r'''
    Sage package for Orthogonal Series Basis.

    TODO: review this file to check the compatibility with the derivative in general.
'''
# Sage imports
from sage.all import cached_method, Matrix, QQ, ZZ, lcm

# Local imports
from ..misc.ore import poly_decomposition
from ..psbasis_old import PSBasis, PolyBasis

class OrthogonalBasis(PolyBasis):
    r'''
        Class for representing a basis of orthogonal polynomials.

        A basis of orthogonal polynomials is a type of polynomial basis for power series
        where the `(n+2)`-th element is built from the `n`-th and `(n+1)`-th elements. This can be
        seen as a three-term recurrence basis. See :dlmf:`18` for further
        information and formulas for orthogonal polynomials.

        The first element in the sequence is given by input and the second element in the sequence 
        is such that the recurrence still holds when taking `P_{-1}(x) = 0`.

        We represent a orthogonal polynomial basis with the three coefficients required for
        the three term recurrence:

        .. MATH::

            P_{n+1}(x) = (a_n x + b_n)P_n(x) - c_nP_{n-1}(x).

        INPUT:

        * ``an``: the first coefficient of the three term recurrence. It can be either a rational
          element in `n` or a function or lambda expression that takes 1 argument.
        * ``bn``: the second coefficient of the three term recurrence. It can be either a rational
          element in `n` or a function or lambda expression that takes 1 argument.
        * ``cn``: the third coefficient of the three term recurrence. It can be either a rational
          element in `n` or a function or lambda expression that takes 1 argument.
        * ``var_name``: the name for the operator representing the multiplication by `x`. By default, this 
          takes the value "x".
        * ``Dx``: the name for the operator representing the derivation w.r.t. `x`. By default, this 
          takes the value "Dx".
        * ``init``: the first element of the basis. By default, this takes the value `1`.
        * ``base``: base ring where the coefficients `a_n`, `b_n`, `c_n` and ``init`` must belong.

        TODO: add examples

        WARNING: the compatibility with "Dx" is not standard, since not all orthogonal basis are compatible
        with the derivation w.r.t. `x`. However, there is always a derivation with which the basis is compatible.
        This derivation is of the form `p(x)\partial_x`, for some `p(x)` that is determined by the three 
        term recurrence (see method :func:`get_differential_equation`).

        List of abstract methods:

        * :func:`~OrthogonalBasis.get_mixed_equation`.
        * :func:`~OrthogonalBasis._first_compatibility`.
    '''
    def __init__(self, an, bn, cn, var_name='x', Dx='Dx', init=1, base=QQ):
        ## Initializing the PolyBasis structure
        super().__init__(base, var_name)

        ## Checking the first element
        init = self.base(init)
        if(init == 0):
            raise ValueError("The first polynomial must be non-zero")
        self.__init = init

        ## Adding the extra information
        n = self.n()
        try:
            an = self.OB()(an); self.__an = an
        except:
            if(an(n=n) in self.OB()): self.__an = an
        try:
            bn = self.OB()(bn); self.__bn = bn
        except:
            if(bn(n=n) in self.OB()): self.__bn = bn
        try:
            cn = self.OB()(cn); self.__cn = cn
        except:
            if(cn(n=n) in self.OB()): self.__cn = cn

        ## The multiplication by var_name compatibility is given
        Sni = self.Sni(); Sn = self.Sn()
        self.set_compatibility(var_name, (cn(n=n+1)/an(n=n+1))*Sn - bn(n=n)/an(n=n) + (1/an(n=n-1))*Sni)

        self.__der_name = Dx
        try:
            self.set_derivation(self.derivation_name(), self._first_compatibility())
        except NotImplementedError: # the class has no specific compatibility with the derivative
            try: # we try to get the mixed relation for the orthogonal basis
                _, a,b,c = self.get_mixed_equation()
                N = self.n(); Sn = self.Sn()
                self.set_derivation(Dx, self.simplify_operator(a*self.recurrence(var_name) + b + c(n=N+1)*Sn))
            except NotImplementedError:
                pass # there is nothing we can do

    # Static elements
    @staticmethod
    def _poly_coeff_by_dict(poly, dict):
        r'''
            Static method for getting the coefficient of a monomial from a dictionary.

            This method unifies the interface between univariate and multivariate polynomials
            in Sage.
        '''
        from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing as isMPoly
        from sage.rings.polynomial.polynomial_ring import is_PolynomialRing as isUPoly

        if(isUPoly(poly.parent())):
            if(len(dict) > 1):
                return 0
            elif(len(dict) == 0):
                return poly
            else:
                return poly[dict.items()[0][1]]
        elif(isMPoly(poly.parent())):
            return poly.coefficient(dict)
        raise TypeError("The input is not a polynomial")

    @staticmethod
    def _poly_degree(poly, var=None):
        r'''
            Static method for getting the degree of polynomial.

            This method unifies the interface between univariate and multivariate polynomials
            in Sage for the method ``degree``.
        '''
        from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing as isMPoly
        from sage.rings.polynomial.polynomial_ring import is_PolynomialRing as isUPoly
        if(var is None or isUPoly(poly.parent())):
            return poly.degree()
        elif(isMPoly(poly.parent())):
            return poly.degree(var)
        raise TypeError("The input is not a polynomial")

    def _element(self, n):
        r'''
            Method to return the `n`-th element of the basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.element` for further information.

            For a :class:`OrthogonalBasis` the output will be a polynomial of degree `n`.

            OUTPUT:

            A polynomial with variable name given by ``var_name`` and degree ``n``.

            TODO: add examples
        '''
        R = self.universe
        x = R.gens()[0]
        an = self.__an; bn = self.__bn; cn = self.__cn

        ## Basic cases
        if(not n in ZZ):
            raise TypeError("The index must be an integer")
        n = ZZ(n)
        if(n < 0):
            raise IndexError("The index must be a non-negative integer")
        elif(n == 0):
            return self.__init
        elif(n == 1):
            return (an(n=0)*x + bn(n=0))*self.__init
        else: # General (recursive) case
            return (an(n=n-1)*x + bn(n=n-1))*self.element(n-1) - cn(n=n-1)*self.element(n-2)

    def _scalar_basis(self, factor):
        r'''
            Method that actually builds the structure for the new basis.

            This method *overrides* the corresponding abstract method from :class:`pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.scalar` for further information.

            TODO: add examples
        '''
        factor = self.OB()(factor)
        n = self.n()
        return OrthogonalBasis(
            self.__an*factor(n=n+1)/factor, # an = an*f(n+1)/f(n) 
            self.__bn*factor(n=n+1)/factor, # bn = bn*f(n+1)/f(n)
            self.__cn*factor(n=n+1)/factor(n=n-1), # cn = cn*f(n+1)/f(n-1)
            str(self.universe.gens()[0]), # the name of the main variable does not change
            self.__der_name,    # the name for derivation does not change
            init = self.__init * factor(n=0) # the first value is altered with the first factor
        )

    def _scalar_hypergeometric(self, factor, quotient):
        r'''
            Method that actually builds the structure for the new basis.

            This method *overrides* the corresponding abstract method from :class:`pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.scalar` for further information.

            TODO: add examples
        '''
        n = self.n(); quotient = self.OB()(quotient)
        return OrthogonalBasis(
            self.__an*quotient, # an = an*f(n+1)/f(n) 
            self.__bn*quotient, # bn = bn*f(n+1)/f(n)
            self.__cn*quotient*quotient(n=n-1), # cn = cn*f(n+1)/f(n-1)
            str(self.universe.gens()[0]), # the name of the main variable does not change
            self.__der_name,    # the name for derivation does not change
            init = self.__init * factor(n=0) # the first value is altered with the first factor
        )

    @cached_method
    def get_differential_equation(self):
        r'''
            Method to get the second order differential equation for a Orthogonal basis.

            By definition, a set of Orthogonal polynomials satisfies a three term recurrence
            of the form  

            .. MATH::

                P_{n+1}(x) = (a_n x + b_n)P_n(x) - c_nP_{n-1}(x).

            This implies that the set also satisfies a second order differential equation. In fact,
            both representation are equivalent. This method computes the second order differential
            equation for the current Orthogonal basis.

            OUTPUT: 

            A triplet `(A(n),B(n),C(n)) \in \mathbb{Q}(n)[x]` such that, for any element `P_n(x)` of the basis, we have

            .. MATH::

                A(n)P_n(x)'' + B(n)P_n(x)' + C(n)P_n(x) = 0.

            TODO: add examples
        '''
        R = self.universe
        x = R.gens()[0]

        rows = []; n = 0
        M = Matrix(rows)
        while(M.rank() < 4 and n < 10):
            n += 1
            rows += [[(k*(k-1) - n*(n-1))*self[n][k], (k+1)*k*self[n][k+1], (k+2)*(k+1)*self[n][k+2], (k-n)*self[n][k], (k+1)*self[n][k+1]] for k in range(n+1)]
            M = Matrix(rows)
        eigen = M.right_kernel()

        a,b,c,d,e = eigen.an_element()*eigen.denominator()
        n = self.n()

        return (a*x**2 + b*x + c),(d*x+e),-n*((n-1)*a + d)

    def get_mixed_equation(self):
        r'''
            Method to get a mixed relation between the shift and differential operators.

            By definition, a set of Orthogonal polynomials satisfies a three term recurrence
            of the form  

            .. MATH::

                P_{n+1}(x) = (a_n x + b_n)P_n(x) - c_nP_{n-1}(x).

            This implies that the set also satisfies a mixed difference-differential equation. In fact,
            both representation are equivalent. This method computes the mixed relation for the current 
            Orthogonal basis.

            OUTPUT: 

            A tuple `(A(n),B(n),C(n),D(n)) \in \mathbb{Q}(n)` such that, for any element `P_n(x)` of the basis, we have

            .. MATH::

                A(n)P_{n}(x)' = (B(n)x+C(n))P_n(x) + D(n)P_{n-1}(x).

            TODO: add examples

            WARNING: **this method is currently not implemented.**
        '''
        raise NotImplementedError("The mixed relation is not (yet) implemented in general")

    @cached_method
    def recurrence(self, operator, cleaned=False):
        r'''
            Method to get the compatibility for an operator.

            This method *overrides* the corresponding abstract method from :class:`pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.recurrence` for further information.

            In a first glance, this method tries the classical compatibility using the compatibility dictionary.
            However, the derivation is not usually compatible with ``self``, but we may need a pre-factor
            `Q(x)` (see method :func:`derivation_name`) to make the derivation compatible.

            Since for any `L \in \mathbb{Q}[x][\partial]` we have that `L \cdot f(x) = 0` if and only
            if `(Q(x)^{\deg(L)}L) \cdot f(x)= 0`, and that last operator **is** compatible, we adapted
            this method to return the compatibility condition related with **that** extended operator.

            TODO: add examples
        '''
        try:
            return super(OrthogonalBasis, self).recurrence(operator, cleaned=cleaned)
        except:
            try:
                poly = operator.polynomial()
            except TypeError:
                poly = operator

            if(self.__der_name in [str(el) for el in poly.variables()]):
                R = poly.parent(); variable = R(self.__der_name); m = OrthogonalBasis._poly_degree(poly,variable)

                coefficients = [self.recurrence(OrthogonalBasis._poly_coeff_by_dict(poly,{variable: i})) for i in range(m+1)]
                monomials = [self.__compatibility_derivation(m,i) for i in range(m+1)]

                output = self.simplify_operator(sum(coefficients[i]*monomials[i] for i in range(m+1)))
                if cleaned:
                    output = self.remove_Sni(output) # we remove the inverse shift
                    # we clean denominators
                    _, coeffs = poly_decomposition(output.polynomial())
                    to_mult = lcm([el.denominator() for el in coeffs])
                    output = (to_mult * output).change_ring(self.OS().base().base())
                
                return output

            raise TypeError("The operator %s is not compatible with %s" %(operator, self))

    @cached_method
    def derivation_name(self):
        r'''
            Name of the compatible derivation with this basis.

            This method returns the name of the 
            associated derivation with this basis. By definition, a set of Orthogonal 
            polynomials satisfies a three term recurrence of the form  

            .. MATH::

                P_{n+1}(x) = (a_n x + b_n)P_n(x) - c_nP_{n-1}(x).

            That leads to a second order differential equation (see method :func:`get_differential_equation`)
            of the form

            .. MATH::

                Q(x)P_n''(x) + R(x)P_n'(x) + S(n)P_n(x) = 0.

            This means that the operator `Q(x)\partial_x` is directly compatible with this basis. This method
            returns the string for this derivation.

            OUTPUT:

            Name of the associated derivation.

            TODO: add examples
        '''
        Q, _, _ = self.get_differential_equation()
        return "("+str(Q)+")*"+self.__der_name

    def _first_compatibility(self):
        r'''
            Method to get compatibility with the associated derivation.

            This method returns the compatibility of the Orthogonal basis with the 
            associated derivation with this basis. By definition, a set of Orthogonal 
            polynomials satisfies a three term recurrence of the form  

            .. MATH::

                P_{n+1}(x) = (a_n x + b_n)P_n(x) - c_nP_{n-1}(x).

            That leads to a second order differential equation (see method :func:`get_differential_equation`)
            of the form

            .. MATH::

                Q(x)P_n''(x) + R(x)P_n'(x) + S(n)P_n(x) = 0.

            This means that the operator `Q(x)\partial_x` is directly compatible with this basis. This method
            computes the compatibility with this operator.

            This method is abstract and may be implemented in all subclasses. If not 
            provided, the compatibility with the derivation will not be set, but no
            error will be raised. See also :func:`get_mixed_relation`.

            OUTPUT:

            An difference operator (see :func:`~pseries_basis.psbasis.PSBasis.OS`) that represents
            the compatibility of the derivation with this basis.
        '''
        raise NotImplementedError("The general first compatibility with derivation is not implemented")

    @cached_method
    def __compatibility_derivation(self, pow_Q, pow_D):
        r'''
            Private method to get the compatibility of some derivative.

            This method computes recursively the compatibility representation of 
            some derivative of the form `Q(x)^d\partial^k` where `Q(x)` is the 
            polynomial that makes `Q(x)\partial` compatible (see method
            :func:`get_differential_equation`) and `\partial` is 
            the standard derivation w.r.t. `x`.

            This only make sense when `d \geq k`, since, otherwise, it is not possible
            to express the operator in terms of `Q(x)\partial`.

            INPUT:

            * ``pow_Q``: value for the power `d`.
            * ``pow_D``: value for the power `k`.

            OUTPUT:

            A difference operator (see :func:`~pseries_basis.psbasis.PSBasis.OS`) that represents the compatibility
            of `Q(x)^d\partial^k` with the orthogonal basis.

            TODO: add examples
        '''
        Q,_,_ = self.get_differential_equation()
        if(pow_Q < pow_D):
            raise ValueError("Incompatibility found because of not valid exponents")
        if(pow_Q > pow_D):
            return self.simplify_operator(self.recurrence(Q**(pow_Q-pow_D))*self.__compatibility_derivation(pow_D, pow_D))
        n = pow_D
        if(n > 1):
            return self.simplify_operator((self.__compatibility_derivation(1,1) - (n-1)*self.recurrence(Q.derivative()))*self.__compatibility_derivation(n-1,n-1))
        if(n == 1):
            return self.recurrence(self.__der_name)
        else: # last case is pow_Q == pow_D == 0 --> no operator
            return self.Sn().parent().one()

###############################################################
### EXAMPLES OF PARTICULAR ORTHOGONAL BASIS
###############################################################
class JacobiBasis(OrthogonalBasis):
    r'''
        Class for the Jacobi Basis with parameters `\alpha` and `\beta`.

        This class represents the :class:`OrthogonalBasis` formed by the set of Jacobi polynomials
        with some fixed parameters `\alpha, \beta`, which are a class of orthogonal polynomials
        with weight function `(1-x)^\alpha (1+x)^\beta`.

        Following the notation in :arxiv:`2202.05550`, we can find that
        this basis has compatibilities with the multiplication by `x` and the derivation
        `(1-x^2)\partial_x`.

        INPUT:

        * ``alpha``: a rational number greater than -1
        * ``beta``: a rational number greater than -1
        * ``var_name``: the name for the operator representing the multiplication by `x`.By default, this 
          takes the value "x".
        * ``Dx``: the name for the operator representing the derivation w.r.t. `x`. By default, this 
          takes the value "Dx".
        * ``base``: base ring where the coefficients `\alpha` and `\beta` must belong.

        TODO: add examples

        List of abstract methods:

        * :func:`pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.get_mixed_equation`.
    '''
    def __init__(self, alpha, beta, var_name='x', Dx='Dx', base=QQ):
        PSBasis.__init__(self, base, var_name=var_name) # needed for getting ``self.n()``
        if(not alpha in base or alpha <= -1):
            raise TypeError("The argument `alpha` must be a rational number greater than -1")
        self.__alpha = base(alpha); alpha = self.__alpha
        if(not beta in base or beta <= -1):
            raise TypeError("The argument `beta` must be a rational number greater than -1")
        self.__beta = base(beta); beta = self.__beta
        n = self.n()

        self.__special_zero = (alpha + beta == 0) or (alpha + beta == -1)

        an = (2*n + alpha + beta + 1)*(2*n + alpha + beta + 2)/(2*(n + 1)*(n + alpha + beta + 1))
        bn = (alpha**2 - beta**2)*(2*n + alpha + beta + 1)/(2*(n + 1)*(n + alpha + beta + 1)*(2*n + alpha + beta))
        cn = (n + alpha)*(n + beta)*(2*n + alpha + beta + 2)/((n + 1)*(n + alpha + beta + 1)*(2*n + alpha + beta))

        super().__init__(an,bn,cn,var_name,Dx,base=base)

    def _element(self, n):
        r'''
            Method to return the `n`-th element of the basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis`.
            See method :func:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.element` for further information.

            This method takes into consideration special cases of the Jacobi polynomial basis, when the 
            coefficients create non-standard behaviors (namely, for `n = 1` when `\alpha + \beta = 0` or 
            `\alpha + \beta == -1`).

            TODO: add examples
        '''
        if(self.__special_zero and n == 1):
            R = self.universe; x = R.gens()[0]
            a0 = (self.__alpha + self.__beta)/2 + 1; b0 = (self.__alpha - self.__beta)/2
            return a0*x + b0

        return super(JacobiBasis, self).element(n)

    def _first_compatibility(self):
        r'''
            Method to get compatibility with the associated derivation.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis`.
            See method :func:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis._first_compatibility` for further information.
        '''
        Sni = self.Sni(); n = self.n(); Sn = self.Sn()
        a = self.__alpha; b = self.__beta

        # operator got using the package ortho_poly_reduction
        op = (2*(b + n + 1)*(a + n + 1)*(a + b + n + 2)/(a + b + 2*n + 2)**2)*Sn
        op += 2*n*(a - b)*(a + b + n + 1)*(a + b + 2*n + 1)/((a + b + 2*n)**2 * (a + b + 2*n + 2))
        op += ((-2)*n*(n - 1)*(a + b + n)/((a + b + 2*n)*(a + b + 2*n - 2)))*Sni

        # the '-' here comes from the factor (1-x^2) of the compatibility
        # since self.get_differential_equation gives Q = x^2-1, we have
        # to change signs in the usual recurrences gotten for Jacobi
        # polynomials
        return -op

    def __repr__(self):
        return f"Jacobi ({self.__alpha},{self.__beta})-Basis ({self(0)}, {self(1)}, {self(2)},...)"

    def _latex_(self):
        return r"\left\{P_n^{(%s,%s)}(%s)\right\}_{n \geq 0}" %(self.__alpha, self.__beta,str(self.universe.gens()[0]))

class GegenbauerBasis(OrthogonalBasis):
    r'''
        Class for the Gegenbauer Basis with parameter `\lambda`.

        This class represents the :class:`OrthogonalBasis` formed by the set of Gegenbauer polynomials
        with some fixed parameter `\lambda`, which are a class of orthogonal polynomials
        with weight function `(1-x^2)^{\lambda - 1/2}`.

        Gegenbauer polynomials are (up to scale) a special case of Jacobi polynomials
        with parameters `\alpha = \beta = \lambda - 1/2`.

        Following the notation in :arxiv:`2202.05550`, we can find that
        this basis has compatibilities with the multiplication by `x` and the 
        differential operator `(1-x^2)\partial_x`.

        INPUT:
            
        * ``L``: a rational number greater than `-1/2` different from `0` representing the parameter `\lambda`
        * ``var_name``: the name for the operator representing the multiplication by `x`. By default, this 
          takes the value "x".
        * ``Dx``: the name for the operator representing the derivation w.r.t. `x`. By default, this 
          takes the value "Dx".
        * ``base``: base ring where the coefficients `\lambda` must belong.

        TODO: add examples

        List of abstract methods:

        * :func:`pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.get_mixed_equation`.
    '''
    def __init__(self, L, var_name='x', Dx='Dx', base=QQ):
        PSBasis.__init__(self, base, var_name=var_name) # needed for getting ``self.n()``
        if(not L in base or L <= -1/2 or L == 0):
            raise TypeError("The argument `alpha` must be a rational number greater than -1/2 different from 0")
        self.__lambda = base(L); L = self.__lambda
        n = self.n()

        an = 2*(n+L)/(n+1)
        cn = (n+2*L-1)/(n+1)

        super().__init__(an,0,cn,var_name,Dx,base=base)

    def _first_compatibility(self):
        r'''
            Method to get compatibility with the associated derivation.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis`.
            See method :func:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis._first_compatibility` for further information.
        '''
        Sni = self.Sni(); n = self.n(); Sn = self.Sn()
        a = self.__lambda

        # operator got using the package ortho_poly_reduction
        op = ((2*a + n)*(2*a + n + 1)/(2*(a + n + 1)))*Sn
        op += (n*(n - 1)/(2*(a + n - 1)))*Sni

        # the '-' here comes from the factor (1-x^2) of the compatibility
        # since self.get_differential_equation gives Q = x^2-1, we have
        # to change signs in the usual recurrences gotten for Jacobi
        # polynomials
        return -op

    def __repr__(self):
        return f"Gegenbauer ({self.__lambda})-Basis ({self(0)}, {self(1)}, {self(2)},...)"

    def _latex_(self):
        return r"\left\{C_n^{(%s)}(%s)\right\}_{n \geq 0}" %(self.__lambda, str(self.universe.gens()[0]))

class LegendreBasis(JacobiBasis):
    r'''
        Class for the Legendre Basis.

        This class represents the :class:`OrthogonalBasis` formed by the set of Legendre polynomials
        which are a class of orthogonal polynomials with weight function `1`.

        Legendre polynomials are a special case of Jacobi polynomials
        with parameters `\alpha = \beta = 0`.

        Following the notation in :arxiv:`2202.05550`, we can find that
        this basis has compatibilities with the multiplication by `x` and 
        the differential operator `(1-x^2)\partial_x`.

        INPUT:
        
        * ``var_name``: the name for the operator representing the multiplication by `x`. By default, this 
          takes the value "x".
        * ``Dx``: the name for the operator representing the derivation w.r.t. `x`. By default, this 
          takes the value "Dx".
        * ``base``: base ring where the coefficients of the polynomial will belong.
        
        TODO: add examples

        List of abstract methods:

        * :func:`pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.get_mixed_equation`.
    '''
    def __init__(self, var_name='x', Dx='Dx', base=QQ):
        super().__init__(0,0,var_name,Dx,base)

    def _first_compatibility(self):
        r'''
            Method to get compatibility with the associated derivation.

            This method *overrides* the corresponding method from :class:`~pseries_basis.orthogonal.ortho_basis.JacobiBasis`.
            See method :func:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis._first_compatibility` for further information.
        '''
        Sni = self.Sni(); n = self.n(); Sn = self.Sn()
        return self.simplify_operator((n*(n-1)/(2*n-1))*Sni - ((n+1)*(n+2)/(2*n+3))*Sn)

    def __repr__(self):
        return f"Legendre Basis ({self(0)}, {self(1)}, {self(2)},...)"

    def _latex_(self):
        return r"\left\{P_n(%s)\right\}_{n \geq 0}" %(str(self.universe.gens()[0]))

class TChebyshevBasis(OrthogonalBasis):
    r'''
        Class for the T-Chebyshev Basis.

        This class represents the T-Chebyshev Basis formed by the set of Chebyshev polynomials
        of the first kind. These polynomials are easily defined with the three term recurrence:

        .. MATH::

            `T_{n+1} = (2-\delta_{n,0})xT_n - T_{n-1}`

        Following the notation in :arxiv:`2202.05550`, we can find that
        this basis has compatibilities with the multiplication by `x` and with the
        differential operator `(1-x^2)\partial_x`.

        INPUT:
        
        * ``var_name``: the name for the operator representing the multiplication by `x`. By default, this 
          takes the value "x".
        * ``Dx``: the name for the operator representing the derivation w.r.t. `x`. By default, this 
          takes the value "Dx".

        TODO: add examples

        List of abstract methods:

        * :func:`pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.get_mixed_equation`.
    '''
    def __init__(self, var_name='x', Dx='Dx'):
        super(TChebyshevBasis, self).__init__(lambda n: 1 if n == 0 else 2,0,1,var_name,Dx,base=QQ)

    def _first_compatibility(self):
        r'''
            Method to get compatibility with the associated derivation.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis`.
            See method :func:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis._first_compatibility` for further information.
        '''
        Sni = self.Sni(); n = self.n(); Sn = self.Sn()
        return self.simplify_operator(((n-1)/2)*Sni - ((n+1)/2)*Sn)

    def __repr__(self):
        return f"T-Chebyshev Basis ({self(0)}, {self(1)}, {self(2)},...)"

    def _latex_(self):
        return r"\left\{T_n(%s)\right\}_{n \geq 0}" %(str(self.universe.gens()[0]))

class UChebyshevBasis(OrthogonalBasis):
    r'''
        Class for the U-Chebyshev Basis.

        This class represents the U-Chebyshev Basis formed by the set of Chebyshev polynomials
        of the second kind. These polynomials are easily defined with the three term recurrence:

        .. MATH::

            `U_{n+1} = 2xU_n - U_{n-1}`

        Following the notation in :arxiv:`2202.05550`, we can find that
        this basis has compatibilities with the multiplication by `x` and with the
        differential operator `(1-x^2)\partial_x`.

        INPUT:
        
        * ``var_name``: the name for the operator representing the multiplication by `x`. By default, this 
          takes the value "x".
        * ``Dx``: the name for the operator representing the derivation w.r.t. `x`. By default, this 
          takes the value "Dx".

        TODO: add examples

        List of abstract methods:

        * :func:`pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.get_mixed_equation`.
    '''
    def __init__(self, var_name='x', Dx='Dx'):
        super(UChebyshevBasis, self).__init__(2,0,1,var_name,Dx,base=QQ)

    def _first_compatibility(self):
        r'''
            Method to get compatibility with the associated derivation.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis`.
            See method :func:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis._first_compatibility` for further information.
        '''
        Sni = self.Sni(); n = self.n(); Sn = self.Sn()
        return self.simplify_operator(((3*n-1)/2)*Sni + ((n+1)/2)*Sn)

    def __repr__(self):
        return f"U-Chebyshev Basis ({self(0)}, {self(1)}, {self(2)},...)"

    def _latex_(self):
        return r"\left\{U_n(%s)\right\}_{n \geq 0}" %(str(self.universe.gens()[0]))

class LaguerreBasis(OrthogonalBasis):
    r'''
        Class for the Laguerre Basis.

        This class represents the :class:`OrthogonalBasis` formed by the set of Laguerre polynomials
        parametrized by `\alpha`, which are a class of orthogonal polynomials with weight function `e^{-x}x^{\alpha}`.

        Following the notation in :arxiv:`2202.05550`, we can find that
        this basis has compatibilities with the multiplication by `x` and with the Euler
        differential operator (i.e., `x\partial_x`).

        INPUT:
        
        * ``alpha``: rational value for parameter `\alpha`.
        * ``var_name``: the name for the operator representing the multiplication by `x`. By default, this 
          takes the value "x".
        * ``Dx``: the name for the operator representing the derivation w.r.t. `x`. By default, this 
          takes the value "Dx".

        TODO: add examples

        List of abstract methods:

        * :func:`pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.get_mixed_equation`.
    '''
    def __init__(self, alpha, var_name='x', Dx='Dx'):
        PSBasis.__init__(self, QQ, var_name=var_name) # needed for getting ``self.n()``
        if(alpha < -1):
            raise ValueError("Laguerre polynomials require an alpha parameter of at least -1")
        self.alpha = alpha

        n = self.n()
        super(LaguerreBasis, self).__init__(-1/(n+1),(2*n+alpha+1)/(n+1),(n+alpha)/(n+1),var_name,Dx,base=QQ)

    def _first_compatibility(self):
        r'''
            Method to get compatibility with the associated derivation.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis`.
            See method :func:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis._first_compatibility` for further information.
        '''
        Sni = self.Sni(); n = self.n()
        return self.simplify_operator(n*Sni - (n+self.alpha+1))

    def __repr__(self):
        return f"{self.alpha}-Laguerre Basis ({self(0)}, {self(1)}, {self(2)},...)"

    def _latex_(self):
        return r"\left\{L_n^{(%s)}(%s)\right\}_{n \geq 0}" %(self.alpha,str(self.universe.gens()[0]))

class HermiteBasis(OrthogonalBasis):
    r'''
        Class for the Hermite Basis `(1,2x,4x^2-2,\dots)`.

        This class represents the :class:`OrthogonalBasis` formed by the set of Hermite polynomials
        which are a class of orthogonal polynomials with weight function `e^{-x^2}`.

        Following the notation in :arxiv:`2202.05550`, we can find that
        this basis has compatibilities with the multiplication by `x` and with the derivation
        w.r.t. `x`.

        INPUT:
        
        * ``var_name``: the name for the operator representing the multiplication by `x`. By default, this 
          takes the value "x".
        * ``Dx``: the name for the operator representing the derivation w.r.t. `x`. By default, this 
          takes the value "Dx".

        TODO: add examples

        List of abstract methods:

        * :func:`pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.get_mixed_equation`.
        * :func:`~OrthogonalBasis._first_compatibility`.
    '''
    def __init__(self, var_name='x', Dx='Dx'):
        PSBasis.__init__(self, QQ, var_name=var_name) # needed for getting ``self.n()``
        super(HermiteBasis, self).__init__(2,0,2*self.n(),var_name,Dx,base=QQ)

        n = self.n(); Sn = self.Sn()
        self.set_derivation(Dx, 2*(n+1)*Sn, True)

    def derivation_name(self):
        r'''
            Name of the compatible derivation with this basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis`.
            See method :func:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.derivation_name` for further information.
        '''
        return super().derivation_name().split(")")[-1]

    def __repr__(self):
        return f"Hermite Basis ({self(0)}, {self(1)}, {self(2)},...)"

    def _latex_(self):
        return r"\left\{H_n(%s)\right\}_{n \geq 0}" %str(self.universe.gens()[0])

class HermitePBasis(OrthogonalBasis):
    r'''
        Class for the Probabilistic Hermite Basis `(1,x,x^2-1,\dots)`.

        This class represents the :class:`OrthogonalBasis` formed by the set of probabilistic Hermite polynomials
        which are a class of orthogonal polynomials with weight function `e^{-x^2/2}`.

        Following the notation in :arxiv:`2202.05550`, we can find that
        this basis has compatibilities with the multiplication by `x` and with the derivation
        w.r.t. `x`.

        INPUT:

        * ``var_name``: the name for the operator representing the multiplication by `x`. By default, this 
          takes the value "x".
        * ``Dx``: the name for the operator representing the derivation w.r.t. `x`. By default, this 
          takes the value "Dx".

        TODO: add examples

        List of abstract methods:

        * :func:`pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.get_mixed_equation`.
        * :func:`~OrthogonalBasis._first_compatibility`.
    '''
    def __init__(self, var_name='x', Dx='Dx'):
        PSBasis.__init__(self, QQ, var_name=var_name) # needed for getting ``self.n()``
        super(HermitePBasis, self).__init__(1,0,self.n(),var_name,Dx,base=QQ)

        n = self.n(); Sn = self.Sn()
        self.set_derivation(Dx, (n+1)*Sn, True)

    def derivation_name(self):
        r'''
            Name of the compatible derivation with this basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis`.
            See method :func:`~pseries_basis.orthogonal.ortho_basis.OrthogonalBasis.derivation_name` for further information.
        '''
        return super().derivation_name().split(")")[-1]

    def __repr__(self):
        return f"Probabilistic Hermite Basis ({self(0)}, {self(1)}, {self(2)},...)"

    def _latex_(self):
        return r"\left\{He_n(%s)\right\}_{n \geq 0}" %str(self.universe.gens()[0])

## Auxiliary methods to manage the dictionaries to handle the reductions
from sage.all import PolynomialRing, FractionField
OrthoRed_R = PolynomialRing(QQ, ['a','b','n'])
OrthoRed_F = FractionField(OrthoRed_R)
OrthoRed_RR = PolynomialRing(OrthoRed_F, ['x'])

OrthoRed_a,OrthoRed_b,OrthoRed_n = OrthoRed_R.gens(); OrthoRed_x = OrthoRed_RR.gens()[0]

def apply_op_dict(op, default=0, *dicts):
    if(len(dicts) < 1 or any((not isinstance(el, dict)) for el in dicts)):
        raise TypeError("Invalid arguments")
    key_set = set(dicts[0].keys())
    for i in range(1,len(dicts)):
        key_set = key_set.union(dicts[i].keys())
    return {key:op(*[el.get(key,default) for el in dicts]) for key in key_set}

def OrthoRed_ssum(*summand):
    return sum(summand)

def OrthoRed_add_dicts(*dicts):
    return apply_op_dict(OrthoRed_ssum, 0, *dicts)

def OrthoRed_sub_dicts(d1, d2):
    return OrthoRed_add_dicts(d1, apply_op_dict(lambda p: -p, 0, d2))

def OrthoRed_scalar_dict(d1, factor):
    return apply_op_dict(lambda p : p*factor, 0, d1)

def OrthoRed_get_shift(dict):
    return {k : v(**{str(OrthoRed_n):2*OrthoRed_n-k}).factor() for (k,v) in dict.items()}

## Methods to reduce Jacobi polynomials
def OrthoRed_jacobi_reduce_beta(n):
    return {
        n : 2*(n+OrthoRed_b+1)/(2*n+OrthoRed_a+OrthoRed_b+2), 
        n+1 : 2*(n+1)/(2*n+OrthoRed_a+OrthoRed_b+2)
    }, (1+OrthoRed_x)

def OrthoRed_jacobi_reduce_alpha(n):
    db,_ = OrthoRed_jacobi_reduce_beta(n)
    da = {n : 2}
    return OrthoRed_sub_dicts(da, db), (1-OrthoRed_x)

def OrthoRed_jacobi_reduce_derivation(n):
    da, _ = OrthoRed_jacobi_reduce_alpha(n-1)
    d_ab = OrthoRed_add_dicts(*[OrthoRed_scalar_dict(OrthoRed_jacobi_reduce_beta(k)[0], v) for (k,v) in da.items()])

    dd = OrthoRed_scalar_dict(d_ab, ((n+OrthoRed_a+OrthoRed_b+1)/2))
    return dd, (1-OrthoRed_x**2)

## Methods to reduce Gegenbauer polynomials
def OrthoRed_gegenbauer_reduce_lambda(n):
    return {
        n: (n+2*OrthoRed_a)*(n+2*OrthoRed_a+1)/(4*OrthoRed_a*(n+OrthoRed_a+1)),
        n+2: -(n+1)*(n+2)/(4*OrthoRed_a*(n+OrthoRed_a+1))
    }, (1-OrthoRed_x**2)

def OrthoRed_gegenbauer_reduce_derivation(n):
    da, fa = OrthoRed_gegenbauer_reduce_lambda(n-1)

    dd = OrthoRed_scalar_dict(da, 2*OrthoRed_a)
    return dd, fa

__all__ = ["OrthogonalBasis", "JacobiBasis", "GegenbauerBasis", "LegendreBasis", "TChebyshevBasis", "UChebyshevBasis", "LaguerreBasis", "HermiteBasis", "HermitePBasis"]
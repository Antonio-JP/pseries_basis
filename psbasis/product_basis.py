r'''
    Sage package for Product of Factorial Series Basis.
'''
# Sage imports
from sage.all import cached_method, prod, ZZ, QQ, Matrix, vector

# Local imports
from .psbasis import PolyBasis
from .factorial_basis import FactorialBasis


class ProductBasis(FactorialBasis):
    r'''
        Class for Product Basis.

        This class represent basis built using the pieces of other polynomial basis.
        Namely, the $n=km+j$ element of the product of $m$ basis, is the product of
            $$Q_n = \prod_{i=1}^{j}P_{k+1}^{(j)}\prod_{i=j+1}^{m}P_{k}^{(j)}.$$

        See the paper :arxiv:`1804.02964v1` for further information.

        INPUT:
            * args: list of Polynomial Basis
            * kwds: optional parameters. The following are allowed:
                * ``X`` or ``name``: name for the variable. 'x' is the default.
                * ``Dx`` or ``ders`` or ``der`` or ``derivations`` or ``derivation``: name or list of names
                  for the derivations compatible with the basis.
                * ``E`` or ``shift`` or ``sh`` or ``endo`` or ``endomorphism``: name or list of names for
                  the endomorphisms compatible with the basis.

        REMARK::
            * If any factor of the basis is a ProductBasis, we will consider its factors as normal
              factors of this basis.
    '''
    def __init__(self, *args, **kwds):
        ## Processing the optional parameters
        X = kwds.get('name', kwds.get('X', 'x'))
        ders = kwds.get('Dx', kwds.get('ders', kwds.get('der', kwds.get('derivations', kwds.get('derivation', [])))))
        endos = kwds.get('E', kwds.get('shift', kwds.get('sh', kwds.get('endo', kwds.get('endomorphism', [])))))
        if(not (type(ders) in (list,tuple))):
            ders = [ders]
        if(not (type(endos) in (list,tuple))):
            endos = [endos]

        ## Calling the super constructor
        super(ProductBasis, self).__init__(X)

        ##Checking the input
        if(len(args) == 1 and type(args[0]) == list):
            args = args[0]
        if(not all(isinstance(el, PolyBasis) for el in args)):
            raise TypeError("All the elements in args must be PolyBasis")

        ## Unrolling the inner Product Basis
        final_args = []
        for el in args:
            if(isinstance(el, ProductBasis)):
                final_args += el.factors()
            elif(isinstance(el, FactorialBasis)):
                final_args += [el]
            else:
                raise TypeError("All the elements in args must be either FactorialBasis or ProductBasis")

        ## Setting the final list factors to the inner variable
        self.__factors = final_args

        ## Extra cached elements
        self.__cached_increasing = {}

        ## Computing the new compatibility operators for this basis
        self.set_compatibility(X, self.__compute_operator_for_X(X))
        for der in ders:
            self.add_derivation(der)
        for endo in endos:
            self.add_endomorphism(endo)


    # PSBasis abstract method
    @cached_method
    def element(self, n, var_name=None):
        r'''
            Method to return the `n`-th element of the basis.

            This method *implements* the corresponding abstract method from :class:`~psbasis.psbasis.PSBasis`.
            See method :func:`~psbasis.psbasis.PSBasis.element` for further information.

            For a :class:`ProductBasis` the output will be a polynomial of degree `n`.

            OUTPUT:

            A polynomial with variable name given by ``var_name`` and degree ``n``.

            TODO: add examples
        '''
        if(var_name is None):
            name = self.var_name()
        else:
            name = var_name

        F = self.nfactors(); factors = self.factors()
        k = n//F; j = n%F

        return prod(factors[i].element(k+1,name) for i in range(j))*prod(factors[i].element(k,name) for i in range(j,F))

    def factors(self):
        r'''
            Getter for the factor of this basis

            This method returns the factor basis that compose the Product Basis
            represented by ``self``.
        '''
        return self.__factors

    @cached_method
    def nfactors(self):
        r'''
            Getter for the number of factors of this basis.
        '''
        return len(self.__factors)

    ## METHODS FOR COMPUTING THE COMPATIBILITY FROM THE BASIS FACTORS
    def __compute_operator_for_X(self, name):
        r'''
            Get the matrix operator for the multiplication by the variable.

            This method computes the matrix of operators that represent the compatibility
            of the multiplication by the variable in the Product Basis. This matrix has
            as many columns and rows as factors the basis.

            INPUT:
                * ``name``: name for the variable. If there is any factor which variable
                  is called differently, this method will raise an error.
        '''
        ## Checking all factors are compatible with the basis
        for el in self.factors():
            el.get_compatibility(name)

        ## Building the elements required for the matrix
        m = self.nfactors()
        ccoeff = [el.constant_coefficient() for el in self.factors()]
        lcoeff = [el.linear_coefficient() for el in self.factors()]

        def alpha(k,j,i):
            if(i==0):
                return -ccoeff[j](n=k+1)/lcoeff[j](n=k+1)
            else:
                return 1/lcoeff[j](n=k+1)

        return self.get_compatibility_sections(m, (0,1,alpha))

    def add_derivation(self, name):
        r'''
            Method to get the compatibility of a derivation.

            This method computes the compatibility of a derivation L over
            the ring K[x]. Such derivation must be compatible with all the
            factors on the basis.

            Following the results of :arxiv:`1804.02964v1`, if $L$
            is $(A_i,B_i)$-compatible with the $i$th factor of ``self``, then
            $L$ is $(mA, B)$-compatible where $A = max(A_i)$ and $B = min(B_i)$.

            This method computes the matrix operator representing the compatibility
            of $L$ in as many sections as factors the basis have and add it to the
            compatibility dictionary of this basis.

            Then it returns this matrix.

            INPUT:
                - ``name``: name of the derivation or a generator of a *ore_algebra*
                  ring of operators.

            WARNING::
                This method do not check whether the operator given is an derivation
                or not. That remains as a user responsability.
        '''
        if(not (type(name) is str)):
            name = str(name)

        if(not self.has_compatibility(name)):
            self.set_compatibility(name, self.__compute_operator_for_derivation(name))

        return self.get_compatibility(name)

    def __compute_operator_for_derivation(self, name):
        r'''
            Method to get the compatibility of a derivation.

            This method computes the compatibility of a derivation L over
            the ring K[x]. Such derivation must be compatible with all the
            factors on the basis.

            Following the results of :arxiv:`1804.02964v1`, if $L$
            is $(A_i,B_i)$-compatible with the $i$th factor of ``self``, then
            $L$ is $(mA, B)$-compatible where $A = max(A_i)$ and $B = min(B_i)$.

            This method computes the matrix operator representing the compatibility
            of $L$ in as many sections as factors the basis have and add it to the
            compatibility dictionary of this basis.

            Then it returns this matrix.

            INPUT:
                - ``name``: name of the derivation or a generator of a *ore_algebra*
                  ring of operators.

            WARNING::
                This method do not check whether the operator given is an derivation
                or not. That remains as a user responsability.
        '''
        n = self.n()
        A = max(factor.A(name) for factor in self.factors()); B = max(factor.B(name) for factor in self.factors()); mA = self.nfactors()*A

        alphas = []
        for j in range(self.nfactors()):
            P = [self.OB()(self.derivation_division_polynomial(name, n, j, mA)[i]) for i in range(mA + B + 1)]
            alphas += [self.equiv_CtD(mA, j, P)]

        def alpha(k,j,i):
            return alphas[j][i+mA](n=k)
        return self.get_compatibility_sections(self.nfactors(), (mA,B,alpha))

    def add_endomorphism(self, name):
        r'''
            Method to get the compatibility of a endomorphism.

            This method computes the compatibility of an endomorphism L, i.e.,
            a map $L: K[x] \rightarrow K[x]$ that is a ring homomorphism. Such
            endomorphism must be compatible with all the factors on the basis.

            Following the results of :arxiv:`1804.02964v1`, if $L$
            is $(A_i,B_i)$-compatible with the $i$th factor of ``self``, then
            $L$ is $(mA, B)$-compatible where $A = max(A_i)$ and $B = min(B_i)$.

            This method computes the matrix operator representing the compatibility
            of $L$ in as many sections as factors the basis have and add it to the
            compatibility dictionary of this basis.

            Then it returns this matrix.

            INPUT:
                - ``name``: name of the endomorphism or a generator of a *ore_algebra*
                  ring of operators.

            WARNING::
                This method do not check whether the operator given is an endomorphism
                or not. That remains as a user responsability.
        '''
        if(not (type(name) is str)):
            name = str(name)

        if(not self.has_compatibility(name)):
            self.set_compatibility(name, self.__compute_operator_for_endomorphism(name))

        return self.get_compatibility(name)

    def __compute_operator_for_endomorphism(self, name):
        r'''
            Method to get the compatibility of a endomorphism.

            This method computes the compatibility of an endomorphism L, i.e.,
            a map $L: K[x] \rightarrow K[x]$ that is a ring homomorphism. Such
            endomorphism must be compatible with all the factors on the basis.

            Following the results of :arxiv:`1804.02964v1`, if $L$
            is $(A_i,B_i)$-compatible with the $i$th factor of ``self``, then
            $L$ is $(mA, B)$-compatible where $A = max(A_i)$ and $B = min(B_i)$.

            This method returns a matrix operator representing the compatibility
            of $L$ in as many sections as factors the basis have.

            INPUT:
                - ``name``: name of the endomorphism or a generator of a *ore_algebra*
                  ring of operators

            WARNING::
                This method do not check whether the operator given is an endomorphism
                or not. That remains as a user responsability.
        '''
        n = self.n()
        A = max(factor.A(name) for factor in self.factors()); B = max(factor.B(name) for factor in self.factors()); mA = self.nfactors()*A

        alphas = []
        for j in range(self.nfactors()):
            P = [self.OB()(self.endomorphism_division_polynomial(name, n, j, mA)[i]) for i in range(mA + B + 1)]
            alphas += [self.equiv_CtD(mA, j, P)]

        def alpha(k,j,i):
            return alphas[j][i+mA](n=k)
        return self.get_compatibility_sections(self.nfactors(), (mA,B,alpha))

    def increasing_polynomial(self, src, shift, diff=None, dst=None):
        r'''
            Returns the increasing factorial for the factorial basis.

            This method *implements* the corresponding abstract method from :func:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.increasing_polynomial` for further information 
            in the description or the output.

            As a :class:`ProductBasis` have several factors, we consider its elements always indexed
            by two elements: `(k,j) \mapsto n = kF + j`, where $F$ is the number of factors.

            INPUT:

            * ``src``: value for `k`.
            * ``shift``: value for `j`, it has to be a value in `\{0,\dots,m-1\}`.
            * ``diff``: difference between the index `n` and the largest index, `m`. Must be a positive integer.
            * ``dst``: value for `m`. Only used (and required) if ``diff`` is ``None``. Must be bigger than `n`.

            TODO: add examples
        '''
        ## Checking the arguments
        if(((src in ZZ) and src < 0) or (not src in self.OB())):
            raise ValueError("The argument `src` must be a expression involving `self.n()` or a positive integer")
        k = src

        if((not shift in ZZ) or (shift < 0) or (shift > self.nfactors()-1)):
            raise ValueError("The argument `shift` must be an integer between 0 and %s" %self.nfactors())
        j = ZZ(shift)
        n = k*self.nfactors() + j

        if(not diff is None):
            if((not diff in ZZ) or diff < 0):
                raise ValueError("The argument `diff` must be None or a positive integer")
            else:
                d = ZZ(diff); p = k + (d+j)//self.nfactors(); q = (d+j)%self.nfactors(); m = p*self.nfactors() + q
        else:
            if(n in ZZ):
                if((not dst in ZZ) or dst < n):
                    raise ValueError("The argument `dst` must be an integer bigger than `n`")
                m = ZZ(dst); d = n - m; p = m//self.nfactors(); q = m%self.nfactors()
            else:
                d = dst-n
                if((not d in ZZ) or d < 0):
                    raise ValueError("The difference between `dst` and `n` must be a positive integer")
                d = ZZ(d); p = k + (d+j)//self.nfactors(); q = (d+j)%self.nfactors(); m = dst

        ## Building the polynomial
        if(not (k,j,d) in self.__cached_increasing):
            def decide_index(k,j,i):
                if(i < j):
                    return k+1
                return k

            self.__cached_increasing[(k,j,d)] = prod(
                self.factors()[i].increasing_polynomial(decide_index(k,j,i),dst=decide_index(p,q,i))
                for i in range(self.nfactors()))

        return self.__cached_increasing[(k,j,d)]

    @cached_method
    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            This method *implements* the corresponding abstract method from :func:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            For a :class:`ProductBasis`, the increasing basis is again a :class:`ProductBasis` of the increasing basis
            of its factors. Depending on the actual shift, the increasing basis may differ. Namely, if the shift is 
            `N = kF+j` where `F` is the number of factors of ``self`` and `B_i` are those factors, then the `N`-th 
            increasing basis is the product:

            .. MATH::

                I(B_{j}, k)\cdots I(B_{F-1},k) I(B_0, k+1) \cdots I(B_{j-1}, k+1),

            where `I(\cdot, k)` is the `k`-th increasing basis of `\cdot`.

            TODO: add examples
        '''
        ## Checking the arguments
        if((shift in ZZ) and shift < 0):
            raise ValueError("The argument `shift` must be a positive integer")
        F = self.nfactors(); factors = self.factors()

        k = shift//F; j = shift%F
        return ProductBasis(*[factors[i].increasing_basis(k) for i in range(j, F)], *[factors[i].increasing_basis(k+1) for i in range(j)])

    @cached_method
    def matrix_ItP(self, src, shift, size):
        r'''
            Method to get the matrix for converting from the increasing basis to the power basis.

            This method *implements* the corresponding abstract method from :func:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.matrix_ItP`.

            For a :class:`ProductBasis`, it is convenient to take the index `n = kF + j` where `F` is
            the number of factors.

            INPUT:

            * ``src``: value for `k`.
            * ``shift``: value for `j`.
            * ``size``: bound on the degree for computing the matrix.
        '''
        ## Checking the arguments
        if(((src in ZZ) and src < 0) or (not src in self.OB())):
            raise ValueError("The argument `src` must be a expression involving `self.n()` or a positive integer")
        k = src
        if(k in ZZ):
            k =  ZZ(k); dest = QQ
        else:
            dest = self.OB()

        if((not shift in ZZ) or (shift < 0) or (shift > self.nfactors()-1)):
            raise ValueError("The argument `shift` must be an integer between 0 and %s" %self.nfactors())
        j = ZZ(shift)

        if((not size in ZZ) or size <= 0):
            raise ValueError("The argument `size` must be a positive integer")

        ## Computing the matrix
        polys = [self.increasing_polynomial(k,j,diff=i) for i in range(size)]
        return Matrix(dest, [[polys[j][i] for j in range(size)] for i in range(size)])

    def derivation_division_polynomial(self, operator, src, shift, diff=None, dst=None):
        r'''
            Method to get the division of a polynomial by other element of the basis after an operator.

            In a Factorial Basis, the $n$th element of the basis divides all the following.
            This means for any pair of indices $r > n$, there is a particular polynomial
            $Q_{n,r} = P_r/P_n$.

            Moreover, by Proposition 1 of :arxiv:`1804.02964v1`, for a fixed operator $L$
            that is $(A,B)$-compatible, we know that $P_{n-A}$ divides $L(P_n)$.

            For a ProductBasis, it is convenient to take the index $n = km + j$ where $m$ is
            the number of factors.

            This method computes the division $L(P_n)/P_m$ where $L$ is a derivation,
            for $m < n-A$.

            INPUT:
                - ``operator``: the operator we want to check. It can be the
                  name for any generator in the *ore_algebra* package or the generator
                  itself.
                - ``src``: value for $k$.
                - ``shift``: the value for $j$.
                - ``diff``: difference between $n$ and $m$. Must be a positive integer.
                - ``dst``: value for $m$. Only used (and required) if ``diff`` is None. Must
                  be smaller than $n-A$.
        '''
        ## Checking the arguments
        ## Reading ``src``
        if(((src in ZZ) and src < 0) or (not src in self.OB())):
            raise ValueError("The argument `src` must be a expression involving `self.n()` or a positive integer")
        k = src

        ## Reading ``shift``
        if((not shift in ZZ) or (shift < 0) or (shift > self.nfactors()-1)):
            raise ValueError("The argument `shift` must be an integer between 0 and %s" %self.nfactors())
        j = ZZ(shift)
        n = k*self.nfactors() + j

        ## Compatibility of ``operator``
        A = [factor.A(operator) for factor in self.factors()]; B = [factor.B(operator) for factor in self.factors()]
        A = max(A); B = min(B); mA = self.nfactors()*A

        ## Reading ``diff`` or ``dst``
        if(not diff is None):
            if((not diff in ZZ) or diff < mA):
                raise ValueError("The argument `diff` must be None or a positive integer")
            else:
                d = ZZ(diff); p = k + (d+j)//self.nfactors(); q = (d+j)%self.nfactors(); r = p*self.nfactors() + q
        else:
            if(n in ZZ):
                if((not dst in ZZ) or dst < n):
                    raise ValueError("The argument `dst` must be an integer bigger than `n`")
                r = ZZ(dst); d = n - r; p = r//self.nfactors(); q = r%self.nfactors()
            else:
                d = dst-n
                if((not d in ZZ) or d < 0):
                    raise ValueError("The difference between `dst` and `n` must be a positive integer")
                d = ZZ(d); p = k + (d+j)//self.nfactors(); q = (d+j)%self.nfactors(); r = dst

        ## Computing the polynomial
        def decide_index(k,j,i):
            if(i < j):
                return k+1
            return k

        # We apply Leibniz rule to the product and combine it using increasing basis and
        # the method `compatible_division` for each factor of the basis.
        return sum( # Leibniz rule
            prod(
                [self.factors()[p].increasing_polynomial(decide_index(k,j,p)-A, A) # Increasing basis
                for p in range(self.nfactors) if p != i],
                self.factors()[i].compatible_division(operator, decide_index(k,j,i), A)) # Starting with the derivative
            for i in range(self.nfactors())
            )

    def endomorphism_division_polynomial(self, operator, src, shift, diff=None, dst=None):
        r'''
            Method to get the division of a polynomial by other element of the basis after an operator.

            In a Factorial Basis, the $n$th element of the basis divides all the following.
            This means for any pair of indices $r > n$, there is a particular polynomial
            $Q_{n,r} = P_r/P_n$.

            Moreover, by Proposition 1 of :arxiv:`1804.02964v1`, for a fixed operator $L$
            that is $(A,B)$-compatible, we know that $P_{n-A}$ divides $L(P_n)$.

            For a ProductBasis, it is convenient to take the index $n = km + j$ where $m$ is
            the number of factors.

            This method computes the division $L(P_n)/P_m$ where $L$ is an endomorphism,
            for $m < n-A$.

            INPUT:
                - ``operator``: the operator we want to check. It can be the
                  name for any generator in the *ore_algebra* package or the generator
                  itself.
                - ``src``: value for $k$.
                - ``shift``: the value for $j$.
                - ``diff``: difference between $n$ and $m$. Must be a positive integer.
                - ``dst``: value for $m$. Only used (and required) if ``diff`` is None. Must
                  be smaller than $n-A$.
        '''
        ## Checking the arguments
        ## Reading ``src``
        if(((src in ZZ) and src < 0) or (not src in self.OB())):
            raise ValueError("The argument `src` must be a expression involving `self.n()` or a positive integer")
        k = src

        ## Reading ``shift``
        if((not shift in ZZ) or (shift < 0) or (shift > self.nfactors()-1)):
            raise ValueError("The argument `shift` must be an integer between 0 and %s" %self.nfactors())
        j = ZZ(shift)
        n = k*self.nfactors() + j

        ## Compatibility of ``operator``
        A = [factor.A(operator) for factor in self.factors()]; B = [factor.B(operator) for factor in self.factors()]
        A = max(A); B = min(B); mA = self.nfactors()*A

        ## Reading ``diff`` or ``dst``
        if(not diff is None):
            if((not diff in ZZ) or diff < mA):
                raise ValueError("The argument `diff` must be None or a positive integer")
            else:
                d = ZZ(diff); p = k + (d+j)//self.nfactors(); q = (d+j)%self.nfactors(); r = p*self.nfactors() + q
        else:
            if(n in ZZ):
                if((not dst in ZZ) or dst < n):
                    raise ValueError("The argument `dst` must be an integer bigger than `n`")
                r = ZZ(dst); d = n - r; p = r//self.nfactors(); q = r%self.nfactors()
            else:
                d = dst-n
                if((not d in ZZ) or d < 0):
                    raise ValueError("The difference between `dst` and `n` must be a positive integer")
                d = ZZ(d); p = k + (d+j)//self.nfactors(); q = (d+j)%self.nfactors(); r = dst

        ## Computing the polynomial
        def decide_index(k,j,i):
            if(i < j):
                return k+1
            return k
        return prod(self.factors()[i].compatible_division(operator, decide_index(k,j,i), A) for i in range(self.nfactors()))

    # FactorialBasis abstract method
    def equiv_DtC(self, bound, shift, *coeffs):
        r'''
            Method to get the equivalence condition for a compatible operator.

            This method *implements* the corresponding abstract method from :func:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.equiv_DtC`.

            For a :class:`ProductBasis`, it is convenient to take the index `n = kF + j` where `F` is
            the number of factors.

            INPUT:

            * ``bound``: value for the lower bound for the compatibility condition (i.e, `A`).
            * ``shift``: value for `j`.
            * ``coeffs``: list of coefficients in representing the coefficients `\alpha_{k,j,i}`.

            TODO: add examples
        '''
        ## Checking the input parameters
        if((not bound in ZZ) or (bound < 0)):
            raise ValueError("The argument `bound` must be a positive integer")
        if(len(coeffs) ==  1 and (type(coeffs) in (tuple, list))):
            coeffs = coeffs[0]
        if((not shift in ZZ) or (shift < 0) or (shift > self.nfactors()-1)):
            raise ValueError("The argument `shift` must be an integer between 0 and %s" %self.nfactors())
        j = ZZ(shift)
        mA = ZZ(bound); B = len(coeffs) - mA - 1; n = self.n(); m = self.nfactors()
        A = (mA - j)//m
        j = (mA - j)%m

        ## At this point we have that `coeffs` is the list of coefficients of
        ## L(P_n)/P_{n-A} in the increasing basis starting with $n-A$.
        ## We only need to change the basis to the Power Basis
        new_alpha = self.matrix_ItP(n-A, j, mA+B+1)*vector(coeffs)

        return [el for el in new_alpha]

    # FactorialBasis abstract method
    def equiv_CtD(self, bound, shift, *coeffs):
        r'''
            Method to get the equivalence condition for a compatible operator.

            This method *implements* the corresponding abstract method from :func:`~psbasis.factorial_basis.FactorialBasis`.
            See method :func:`~psbasis.factorial_basis.FactorialBasis.equiv_CtD`.

            For a ProductBasis, it is convenient to take the index `n = kF + j` where `F` is
            the number of factors.

            INPUT:

            * ``bound``: value for the lower bound for the compatibility condition (i.e, `A`).
            * ``shift``: value for `j`.
            * ``coeffs``: list representing the coefficients of the polynomial `L \cdot P_n(x)/P_{n-A}(x)` in the canonical 
              power basis.

            TODO: add examples
        '''
        ## Checking the input parameters
        if((not bound in ZZ) or (bound < 0)):
            raise ValueError("The argument `bound` must be a positive integer")
        if(len(coeffs) ==  1 and (type(coeffs) in (tuple, list))):
            coeffs = coeffs[0]
        if((not shift in ZZ) or (shift < 0) or (shift > self.nfactors()-1)):
            raise ValueError("The argument `shift` must be an integer between 0 and %s" %self.nfactors())
        j = ZZ(shift)
        mA = ZZ(bound); B = len(coeffs) - mA - 1; n = self.n(); m = self.nfactors()
        A = mA//m
        j = j - mA%m
        if(j < 0):
            j += m
            A += 1

        ## At this point we have that `coeffs` is the list of coefficients of
        ## L(P_n)/P_{n-A} in the power basis. If we change to the increasing
        ## basis starting in $n-A$ then we have the $alpha_{n,i}$.
        new_alpha = self.matrix_PtI(n-A, shift, mA+B+1)*vector(coeffs)

        return [el for el in new_alpha]

    def __repr__(self):
        return "ProductBasis" + "".join(["\n\t- %s" %repr(f) for f in self.factors()])

    def _latex_(self):
        return "".join([f._latex_() for f in self.factors()])
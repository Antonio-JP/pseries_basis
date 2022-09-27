r'''
    Sage package for Product of Factorial Series Basis.
'''
# Sage imports
from sage.all import cached_method, prod, ZZ, vector, ceil

# Local imports
from .factorial_basis import FactorialBasis


class SievedBasis(FactorialBasis):
    r'''
        Class for a Sieved Basis.

        A sieved basis is a factorial basis built from a finite set
        of source basis `B_i = \left(P_n^{(i)}(x)\right)_n` for `i=0,\ldots,F-1`. This is built 
        in `m` sections using a *deciding cycle*:

        .. MATH::

            (\sigma_0,\ldots,\sigma_{m-1})

        where `\sigma_i \in \{0,\ldots,F-1\}`. We can then define the `n`-th element
        of the basis with the following formula:

        .. MATH::

            Q_n(x) = \prod_{i=0}^F P_{e_i(n)}^{(i)}(x)

        where the following formula stands:
        
        * `n = km+r`, 
        * `S_i = \# \{ j \in \{0,\ldots,m-1\}\ :\ \sigma_j = i\}`,
        * `e_i(n) = S_im + \#\{j \in \{0,\ldots,r\}\ :\ \sigma_j = i\}`.

        If we look recursively, we can see that each element is built from the previous
        element by increasing one of the factors one degree in the corresponding basis:

        .. MATH::

            Q_n(x) = Q_{n-1}(x)\frac{P_{e_{\sigma_i}(n)}^{(\sigma_i)}(x)}{P_{e_{\sigma_i}(n)-1}^{(\sigma_i)}(x)}

        INPUT:

        * ``factors``: the basis that build the sieved basis.
        * ``cycle``: a tuple of length `m` indicating which factor use in each step.
        * ``init``: value for the constant element of the basis.
        * ``X``: name of the operator representing the multiplication by `x`.
        * ``ends``: endomorphism which compatibility we will try to extend.
        * ``ders``: derivations which compatibility we wil try to extend.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: B = BinomialBasis(); P = PowerBasis()
            sage: B2 = SievedBasis([B,P], [0,1,1,0])
            sage: B2[:6]
            [1, x, x^2, x^3, 1/2*x^4 - 1/2*x^3, 1/6*x^5 - 1/2*x^4 + 1/3*x^3]

        With this system, we can build the same basis changing the order and the values in the cycle::

            sage: B3 = SievedBasis([P,B], [1,0,0,1])
            sage: all(B3[i] == B2[i] for i in range(30))
            True

        The length of the cycle is the number of associated sections::

            sage: B2.nsections()
            4
            sage: SievedBasis([B,B,P],[0,0,1,2,1,2]).nsections()
            6

        This basis can be use to deduce some nice recurrences for the Apery's `\zeta(2)` sequence::

            sage: b1 = FallingBasis(1,0,1); b2 = FallingBasis(1,1,-1); n = b1.n()
            sage: B = SievedBasis([b1,b2],[0,1]).scalar(1/factorial(n))

        This basis ``B`` contains the elements 

        .. MATH::

            \begin{matrix}
            \binom{x + n}{2n}\ \text{if }n\equiv 0\ (mod\ 2)\\
            \binom{x + n}{2n+1}\ \text{if }n\equiv 1\ (mod\ 2)
            \end{matrix}

        We first extend the compatibility with `E: x\mapsto x+1` by guessing and then we compute the sieved basis
        with the binomial basis with the cycle `(1,0,1)`::

            sage: B.set_compatibility('E', guess_compatibility_E(B, sections=2))
            sage: B2 = SievedBasis([BinomialBasis(), B], [1,0,1], ends=['E'])

        Now the basis ``B2`` is formed in 3 sections by the following elements:

        .. MATH::

            \begin{matrix}
                \binom{x}{n}\binom{x+n}{2n}\ \text{if }n\equiv 0\ (mod\ 3)\\
                \binom{x}{n}\binom{x+n}{2n+1}\ \text{if }n\equiv 1\ (mod\ 3)\\
                \binom{x}{n+1}\binom{x+n}{2n+1}\ \text{if }n\equiv 2\ (mod\ 3)
            \end{matrix}

        We can check that `B2` is compatible with the multiplication by `x` and with 
        the endomorphism `E`::

            sage: a,b,m,alpha = B2.compatibility('x')
            sage: Matrix([[alpha(i,j,B2.n()) for j in range(-a,b+1)] for i in range(m)])
            [      n 2*n + 1]
            [      n   n + 1]
            [ -n - 1 2*n + 2]
            sage: B2.recurrence('x')
            [        n         0 (2*n)*Sni]
            [(2*n + 1)         n         0]
            [        0   (n + 1)  (-n - 1)]
            sage: a,b,m,alpha = B2.compatibility('E')
            sage: Matrix([[alpha(i,j,B2.n()) for j in range(-a,b+1)] for i in range(m)])
            [                      1           (4*n - 3/2)/n                     3/2                       1]
            [    (n - 1/2)/(n + 1/2)         1/2*n/(n + 1/2) (3/2*n + 1/2)/(n + 1/2)                       1]
            [                      0                       1       (3*n + 2)/(n + 1)                       1]
            sage: B2.recurrence('E')                                                                                                                                                         
            [                    Sn + 1        (3*n + 1)/(2*n + 1)                          1]
            [    (8*n + 5)/(2*n + 2)*Sn (2*n + 1)/(2*n + 3)*Sn + 1          (3*n + 2)/(n + 1)]
            [                    3/2*Sn       (n + 1)/(2*n + 3)*Sn                          1]

        Now consider the following difference operator:

        .. MATH::

            L = (x+2)^2 E^2 - (11x^2+33x+25)E - (x+1)^3

        This operator `L` is compàtible with the basis ``B2``. We can get then
        the associated recurrence matrix. Taking the first column and the GCRD
        of its elements, we can see that if a formal power series `y(x) = \sum_n y_n x^n`
        that can be written in the form `y(x) = \sum_{n\geq 0}c_n\binom{x}{n}\binom{x+n}{2n}` satisfies
        that

        .. MATH::

            (n+1)^2c_{n+1} - 2(2n+1)c_n = 0.

        Doing that with the code::

            sage: from ore_algebra import OreAlgebra
            sage: R.<x> = QQ[]; OE.<E> = OreAlgebra(R, ('E', lambda p : p(x=x+1), lambda p : 0))   
            sage: L = (x+2)^2*E^2 - (11*x^2 + 33*x+25)*E - (x+1)^2 
            sage: M = B2.recurrence(L)
            sage: column = [B2.remove_Sni(M.coefficient((j,0))) for j in range(M.nrows())]
            sage: column[0].gcrd(*column[1:])
            (n + 1)*Sn - 4*n - 2
    '''
    def __init__(self, factors, cycle, init=1, X='x', ends=[], ders=[]):
        ## Checking the input
        if(not type(factors) in (list,tuple)):
            raise TypeError("The factors must be either a list or a tuple")
        if(any(not isinstance(el, FactorialBasis) for el in factors)):
            raise TypeError("All the factors has to be factorial basis")

        if(not type(cycle) in (list,tuple)):
            raise TypeError("The deciding cycle must be a list or a tuple")
        cycle = [ZZ(el) for el in cycle]
        if(any(el < 0 or el > len(factors) for el in cycle)):
            raise ValueError("The deciding cycle must be composed of integers indexing the factors basis")

        ## Storing the main elements
        self.__factors = tuple(factors)
        self.__cycle = tuple(cycle)

        ## Calling the previous constructor
        super().__init__(X)

        ## Other cached elements
        self.__init = init
        self.__cached_increasing = {}

        ## Extendeding the required operators
        self.extend_compatibility_X()

        self.__ends = []; self.__ders = []
        for endomorphism in ends:
            try:
                self.extend_compatibility_E(endomorphism)
            except (NotImplementedError):
                pass
        for derivation in ders:
            try:
                self.extend_compatibility_D(derivation)
            except (NotImplementedError):
                pass

    def element(self, n, var_name=None):
        r'''
            Method to return the `n`-th element of the basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.psbasis.PSBasis`.
            See method :func:`~pseries_basis.psbasis.PSBasis.element` for further information.

            For a :class:`SievedBasis` the output will be a polynomial of degree `n`.

            OUTPUT:

            A polynomial with variable name given by ``var_name`` and degree ``n``.

            TODO: add examples
        '''
        indices = [self.index(n,i) for i in range(self.nfactors())]
        return self.__init*prod([self.factors[i].element(indices[i],var_name) for i in range(self.nfactors())])

    @cached_method
    def appear(self, i):
        r'''
            Return the appearances of the basis `i` in the deciding cycle.

            This method computes how many times we increase the `i`-th basis
            in each cycle of the :class:`SievedBasis`. This is equivalent to 
            see how many times the number `i` appears on the deciding cycle (see
            property :func:`cycle`).

            INPUT:

            * ``i``:index we want to check. It must be an element between `0` and
              the number of factors of the :class:`SievedBasis` (see method :func:`nfactors`).

            OUTPUT:

            It returns the number of appearances of `i` in the deciding cycle.

            TODO: add examples.
        '''
        if(not i in ZZ):
            raise TypeError("The index must be an integer")
        i = ZZ(i)
        if((i < 0) or (i > self.nfactors())):
            raise ValueError("The index must be between 0 and %d" %self.nfactors())
        return self.cycle.count(i)

    def index(self, n, i):
        r'''
            Returns the index of the `i`-th basis at the element `n`.

            This method computes the actual index of the `i`-th basis in the
            :class:`SievedBasis` for its `n`-th element. Recall that the
            `n = kF + r` element of a :class:`SievedBasis` can be computed 
            with:

            .. MATH::

                Q_n(x) = \prod_{i=0}^F P_{e_i(n)}^{(i)}(x)

            This method returns the value of `e_i(n)`.

            INPUT:

            * ``n``: element of the basis we are considering. It can be an
              expression involving `n`, but we need that ``n%self.nsections()``
              is an integer. It can also be the tuple `(k,r)` such that `n = kF+r`.
            * ``i``: index we want to check. It must be an element between `0` and
              the number of factors of the :class:`SievedBasis` (see method :func:`nfactors`).

            TODO: add examples
        '''
        ## Checking the input 'i' 
        if(not i in ZZ):
            raise TypeError("The index must be an integer")
        i = ZZ(i)
        if((i < 0) or (i >= self.nfactors())):
            raise ValueError("The index must be between 0 and %d" %self.nfactors())

        ## Checking the input 'n' 
        if(not type(n) in (list, tuple)):
            m,r = self.extended_quo_rem(n, self.nsections())
        else:
            m,r = n
        if((not r in ZZ) or (r < 0) or (r >= self.nsections())):
            raise ValueError("The value for 'n' must be compatible with taking module %d" %self.nsections())

        r = ZZ(r)
        s = self.cycle[:r].count(i)
        return self.appear(i)*m + s

    def __repr__(self):
        return ("Sieved Basis %s of the basis:" %str(self.cycle)) + "".join(["\n\t- %s" %repr(f) for f in self.factors])

    def _latex_(self):
        return (r"\prod_{%s}" %self.cycle)  + "".join([f._latex_() for f in self.factors])

    def root_sequence(self):
        r'''
            Method that returns the root sequence of the polynomial basis.

            This method *overrides* the implementation from class :class:`FactorialBasis`. See :func:`FactorialBasis.root_sequence`
            for a description on the output.

            In a sieved basis, the root sequence is a nice entanglement of the root sequences
            given by the deciding cycle (see method :func:`index`).

            TODO: add examples
        '''
        def _root_sb(n):
            r = n%self.nsections()
            factor = self.cycle[r]
            return self.factors[factor].root_sequence()(n=self.index(n, factor))

        return _root_sb

    def constant_coefficient(self):
        r'''
            Getter for the constant coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.constant_coefficient` for further information 
            in the description or the output.

            TODO: add examples.
        '''
        def _const_sb(n):
            r = n%self.nsections()
            factor = self.cycle[r]
            return self.factors[factor].constant_coefficient()(n=self.index(n, factor))
        return _const_sb
    
    def linear_coefficient(self):
        r'''
            Getter for the linear coefficient of the factorial basis.

            This method *overrides* the corresponding method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.linear_coefficient` for further information 
            in the description or the output.

            TODO: add examples.
        '''
        def _lin_sb(n):
            r = n%self.nsections()
            factor = self.cycle[r]
            return self.factors[factor].linear_coefficient()(n=self.index(n, factor))
        return _lin_sb

    factors = property(lambda self: self.__factors) #: property to get the factors of the :class:`SievedBasis`
    cycle = property(lambda self: self.__cycle) #: property to get the deciding cycle of the :class:`SievedBasis`

    def nfactors(self):
        r'''
            Method to get the number of factors of the sieved basis.

            This method returns the number of factors which compose
            this :class:`SievedBasis`.

            OUTPUT:

            Number of factors of this :class:`SievedBasis`.
            
            TODO: add examples
        '''
        return len(self.factors)

    def nsections(self):
        r'''
            Method to get the number of sections of the sieved basis.

            This method returns the number of elements in the deciding cycle which 
            is the number of sections in which the :class:`SievedBasis` is divided.

            OUTPUT:

            Number of sections of this :class:`SievedBasis`.
            
            TODO: add examples
        '''
        return len(self.cycle)

    def extend_compatibility_X(self):
        r'''
            Method to extend the compatibility of the multiplication by `x`.

            This method computes the compatibility of a he multiplication by `x` over
            the ring `\mathbb{K}[x]`. This operator is always compatible with all 
            :class:`FactorialBasis`.

            If this method was already called (or the compatibility was found in another way)
            this method only returns the compatibility

            OUTPUT:

            The compatibility for the multiplication by `x` computed during this process.

            TODO: add examples
        '''
        if(not self.has_compatibility(self.var_name())):
            self.set_compatibility(self.var_name(), self._extend_compatibility_X())

        return self.compatibility(self.var_name())

    def extend_compatibility_E(self, name):
        r'''
            Method to extend the compatibility of an endomorphism.

            This method computes the compatibility of an endomorphism `L` over
            the ring `\mathbb{K}[x]`. Such derivation must be compatible with all the
            factors on the basis.

            If the operator `L` was already compatible with ``self``, this method does
            nothing.

            INPUT:

            * ``name``: name of the derivation or a generator of a *ore_algebra*
              ring of operators.

            OUTPUT:

            The compatibility for `L` computed during this process.

            WARNING:

            This method do not check whether the operator given is an endomorphism
            or not. That remains as a user responsability.

            TODO: add examples
        '''
        if(not (type(name) is str)):
            name = str(name)

        if(not self.has_compatibility(name)):
            self.set_compatibility(name, self._extend_compatibility_E(name))
            self.__ends += [name]

        return self.compatibility(name)

    def extend_compatibility_D(self, name):
        r'''
            Method to extend the compatibility of a derivation.

            This method computes the compatibility of a derivation `L` over
            the ring `\mathbb{K}[x]`. Such derivation must be compatible with all the
            factors on the basis.

            If the operator `L` was already compatible with ``self``, this method does
            nothing.

            INPUT:

            * ``name``: name of the derivation or a generator of a *ore_algebra*
              ring of operators.

            OUTPUT:

            The compatibility for `L` computed during this process.

            WARNING:

            This method do not check whether the operator given is a derivation
            or not. That remains as a user responsability.

            TODO: add examples
        '''
        if(not (type(name) is str)):
            name = str(name)

        if(not self.has_compatibility(name)):
            self.set_compatibility(name, self._extend_compatibility_D(name))
            self.__ders += [name]

        return self.compatibility(name)

    def _extend_compatibility_X(self):
        r'''
            Method that extend the compatibility of multiplication by `x`.

            This method uses the information in the factor basis to extend 
            the compatibility behavior of the multiplication by `x` to the 
            :class:`SievedBasis`.

            This method can be extended in subclasses for a different behavior.
        '''
        X = self.var_name(); m = self.nsections(); F = self.nfactors()
        comps = [self.factors[i].compatibility(X) for i in range(F)]
        t = [comps[i][2] for i in range(F)]
        S = [self.appear(i) for i in range(F)]
        s = lambda i,r : self.cycle[:r].count(i)        
        alphas = [comps[i][3] for i in range(F)]
        
        ## Computing the optimal value for the sections
        T = 1
        while(any([not T*S[i]%t[i] == 0 for i in range(F)])): T += 1 # this always terminate at most with T = lcm(t_i)
        
        a = [T*S[i]//t[i] for i in range(F)]

        def new_alphas(i,j,k):
            i0, i1 = self.extended_quo_rem(i,m)
            next = self.cycle[i1]
            i2, i3 = self.extended_quo_rem(S[next]*i0 + s(next,i1), t[next])
            return alphas[next](i3,j,a[next]*k + i2)
        
        return (0,1,m*T,new_alphas)

    def _extend_compatibility_E(self, E):
        r'''
            Method that extend the compatibility of an endomorphism `E`.

            This method uses the information in the factor basis to extend 
            the compatibility behavior of an endomorphism `E` to the 
            :class:`SievedBasis`.

            This method can be extended in subclasses for a different behavior.

            INPUT:

            * ``E``: name of the endomorphism to extend.

            OUTPUT:

            A tuple `(A,B,m,\alpha_{i,k,j})` representing the compatibility of ``E``
            with ``self``.
        '''
        A, m, D = self._compatible_division_E(E)
        n = self.n()
        B = D(0,0,n).degree()-A

        alphas = []
        for i in range(m):
            alphas += [self.matrix_PtI(m*n-A+i,A+B+1)*vector([D(i,0,n)[j] for j in range(A+B+1)])]

        return (A, B, m, lambda i,j,k : alphas[i][j+A](n=k))

    def _extend_compatibility_D(self, D):
        r'''
            Method that extend the compatibility of a derivation `D`.

            This method uses the information in the factor basis to extend 
            the compatibility behavior of a derivation `D` to the 
            :class:`SievedBasis`.

            This method can be extended in subclasses for a different behavior.

            INPUT:

            * ``D``: name of the derivation to extend.

            OUTPUT:

            A tuple `(A,B,m,\alpha_{i,k,j})` representing the compatibility of ``D``
            with ``self``.
        '''
        A, m, Q = self._compatible_division_D(D)
        n = self.n()
        B = max(Q(0,0,n).degree()-A,0)

        alphas = []
        for i in range(m):
            alphas += [self.matrix_PtI(m*n-A+i,A+B+1)*vector([Q(i,0,n)[j] for j in range(A+B+1)])]
        
        return (A, B, m, lambda i,j,k : alphas[i][j+A](n=k))

    def increasing_polynomial(self, src, diff=None, dst=None):
        r'''
            Returns the increasing factorial for the factorial basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_polynomial` for further information 
            in the description or the output.

            As a :class:`SievedBasis` is composed with several factors, we compute the difference between each element
            in the factors and compute the corresponding product of the increasing polynomials. 

            In this case, we consider the input given by `n = kF + r` where `F` is the number of sections of the 
            :class:`SievedBasis` (see method :func:`nsections`).

            INPUT:

            * ``src``: either the value of `n` or a tuple with the values `(k,r)`
            * ``diff``: difference between the index `n` and the largest index, `m`. Must be a positive integer.
            * ``dst``: value for `m`. It could be either its value or a tuple `(t,s)` where `m = tF + s`.
        '''
        ## Checking the input "src"
        if(not type(src) in (tuple, list)):
            k,r = self.extended_quo_rem(src, self.nsections())
            if(not r in ZZ):
                raise ValueError("The value for the starting point must be an object where we can deduce the section")
        else:
            k, r = src

        ## If no diff, we use dst instead to build diff
        if(diff == None):
            if(type(dst) in (tuple, list)):
                dst = dst[0]*self.nsections() + dst[1]
            diff = dst - src
        
        ## Now we check the value for 'diff'
        if(not diff in ZZ):
            raise TypeError("The value of 'diff' must be an integer")
        diff = ZZ(diff)
        if(diff < 0):
            raise ValueError("The value for 'diff' must be a non-negative integer")
        if(diff == 0):
            return self.polynomial_ring().one()

        if(not (k,r,diff) in self.__cached_increasing):
            original_index = [self.index((k,r), i) for i in range(self.nfactors())]
            t, s = self.extended_quo_rem(diff+r, self.nsections())
            end_index = [self.index((k+t, s), i) for i in range(self.nfactors())]
            self.__cached_increasing[(k,r,diff)] = prod(
                [self.factors[i].increasing_polynomial(original_index[i],dst=end_index[i]) for i in range(self.nfactors())]
            )
        return self.__cached_increasing[(k,r,diff)]

    @cached_method
    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            This method *implements* the corresponding abstract method from :class:`~pseries_basis.factorial_basis.FactorialBasis`.
            See method :func:`~pseries_basis.factorial_basis.FactorialBasis.increasing_basis` for further information.

            For a :class:`SievedBasis`, the increasing basis is again a :class:`SievedBasis` of the increasing basis
            of its factors. Depending on the actual shift, the increasing basis may differ. Namely, if the shift is 
            `N = kF+j` where `F` is the number of sections of ``self`` and `B_i` are those factors, then the we can express 
            the increasing basis as a :class:`SievedBasis` again.

            INPUT:

            * ``shift``: value for the starting point of the increasing basis. It can be either
              the value for `N` or the tuple `(k,j)`.
              
            OUTPUT:

            A :class:`SievedBasis` representing the increasing basis starting at `N`.

            WARING: currently the compatibilities aer not extended to the increasing basis.

            TODO: add examples
        '''
        ## Checking the input "src"
        if(type(shift) in (tuple, list)):
            N = shift[0]*self.nsections() + shift[1]
        else:
            N = shift
            shift = self.extended_quo_rem(N,self.nsections())
        
        if((shift[1] < 0) or (shift[1] > self.nsections())):
            raise ValueError("The input for the shift is not correct")

        new_cycle = self.cycle[shift[1]:] + self.cycle[:shift[1]]
        indices = [self.index(shift, i) for i in range(self.nfactors())]
        new_basis = [self.factors[i].increasing_basis(indices[i]) for i in range(self.nfactors())]
        return SievedBasis(new_basis, new_cycle, X=self.var_name())
     
    def compatible_division(self, operator):
        r'''
            Method to get the division of a polynomial by other element of the basis after an operator.

            This method *overrides* the implementation from class :class:`FactorialBasis`. See :func:`FactorialBasis.compatible_division`
            for a description on the output.

            For a :class:`SievedBasis`, since its elements are products of elements of other basis, we can compute this 
            division using the information in the factors of ``self``. However, we need to know how this operator
            acts on products distinguishing between three classes:

            * **Multiplication operators**: `L(f(x)) = g(x)f(x)`.
            * **Endomorphisms**: `L(f(x)g(x)) = L(f(x))L(g(x))`.
            * **Derivations**: `L(f(x)g(x)) = L(f(x))g(x) + f(x)L(g(x))`.

            In order to know if an operator is an *endomorphism* or a *derivation*, we check if we have extended already those 
            compatibilities. If we do not found them, we assume they are multiplication operators.

            TODO: add examples
        '''
        if(str(operator) in self.__ders):
            return self._compatible_division_D(operator)
        elif(str(operator) in self.__ends):
            return self._compatible_division_E(operator)
        else:
            return self._compatible_division_X(operator)

    def _compatible_division_X(self, operator):
        r'''
            Method o compute the compatible division for multiplication operators.
        '''
        raise NotImplementedError("_compatible_division_X not implemented for Sieved Basis")

    def _compatible_division_D(self, operator):
        r'''
            Method o compute the compatible division for derivations.
        '''
        F = self.nfactors(); m = self.nsections()
        comp_divisions = [self.factors[i].compatible_division(operator) for i in range(F)] # list with (A_i, t_i, D)
        As = [comp_divisions[i][0] for i in range(F)]; t = [comp_divisions[i][1] for i in range(F)]
        D = [comp_divisions[i][2] for i in range(F)]
        I = [self.factors[i].increasing_polynomial for i in range(F)]
        S = [self.appear(i) for i in range(F)]; s = lambda i,r : self.cycle[:r].count(i)

        ## Computing the optimal value for the sections
        T = 1
        while(any([not T*S[i]%t[i] == 0 for i in range(F)])): T += 1 # this always terminate at most with T = lcm(t_i)
        a = [T*S[i]//t[i] for i in range(F)]

        ## Computing the lower bound for the final compatibility
        S = [self.appear(i) for i in range(F)]; A = max(int(ceil(As[i]/S[i])) for i in range(F))
        b = [A*S[i] - As[i] for i in range(F)]

        def new_D(r,j,n):
            if(j != 0): raise IndexError("Division not computed for more than compatibility")
            r0, r1 = self.extended_quo_rem(r, m)
            r2, r3 = list(zip(*[self.extended_quo_rem(S[i]*r0+s(i,r1), t[i]) for i in range(F)]))

            return sum(
                D[i](r3[i], b[i], a[i]*n+r2[i]) * 
                prod(
                    I[j]((a[j]*n+r2[j])*t[j] + r3[j] - As[j] - b[j], As[j]+b[j])
                    for j in range(F) if i != j
                )
                for i in range(F)
            )
            
        return (m*A, m*T,new_D)

    def _compatible_division_E(self, operator):
        r'''
            Method o compute the compatible division for endomorphisms.
        '''
        F = self.nfactors(); m = self.nsections()
        comp_divisions = [self.factors[i].compatible_division(operator) for i in range(F)] # list with (A_i, t_i, D)
        As = [comp_divisions[i][0] for i in range(F)]; t = [comp_divisions[i][1] for i in range(F)]
        D = [comp_divisions[i][2] for i in range(F)]
        S = [self.appear(i) for i in range(F)]; s = lambda i,r : self.cycle[:r].count(i)      

        ## Computing the optimal value for the sections
        T = 1
        while(any([not T*S[i]%t[i] == 0 for i in range(F)])): T += 1 # this always terminate at most with T = lcm(t_i)
        a = [T*S[i]//t[i] for i in range(F)]

        ## Computing the lower bound for the final compatibility
        S = [self.appear(i) for i in range(F)]; A = max(int(ceil(As[i]/S[i])) for i in range(F))
        b = [A*S[i] - As[i] for i in range(F)]

        def new_D(r,j,n):
            if(j != 0): raise IndexError("Division not computed for more than compatibility")
            r0, r1 = self.extended_quo_rem(r, m)
            r2, r3 = list(zip(*[self.extended_quo_rem(S[i]*r0+s(i,r1), t[i]) for i in range(F)]))

            return prod(D[i](r3[i],b[i],a[i]*n+r2[i]) for i in range(F))
            
        return (m*A, m*T,new_D)

    def is_quasi_func_triangular(self):
        return all(basis.is_quasi_func_triangular() for basis in self.factors)
    def is_quasi_eval_triangular(self):
        return all(basis.is_quasi_eval_triangular() for basis in self.factors)

class ProductBasis(SievedBasis):
    r'''
        Class for Product Basis.

        This class represent basis built using the pieces of other polynomial basis.
        Namely, the `n=km+j` element of the product of `m` basis, is the product of

        .. MATH::

            Q_n(x) = \prod_{i=1}^{j}P_{k+1}^{(j)}(x)\prod_{i=j+1}^{m}P_{k}^{(j)}(x).

        See the paper :arxiv:`2202.05550` for further information.

        INPUT:

        * ``factors``: list of :class:`~pseries_basis.factorial_basis.FactorialBasis` to build the :class:`ProductBasis`.
        * ``init``: value for the constant element of the basis.
        * ``X``: name of the operator representing the multiplication by `x`.
        * ``ends``: endomorphism which compatibility we will try to extend.
        * ``ders``: derivations which compatibility we wil try to extend.

        EXAMPLES::

            sage: from pseries_basis import *
            sage: B1 = BinomialBasis(); B2 = PowerBasis(); B3 = FallingBasis(1,0,1)
            sage: ProductBasis([B1,B2]).factors == (B1, B2)
            True
            sage: ProductBasis([B1,B2]).nfactors()
            2
            sage: ProductBasis([B1,B3,B2]).factors == (B1,B3,B2)
            True
            sage: ProductBasis([B1,B3,B2]).nfactors()
            3

        This class can be use to simplify the notation os :class:`SievedBasis`. The following example
        illustrates how this can be used to understand better the recurrence for the Apery's `\zeta(3)`-recurrence::

            sage: b1 = FallingBasis(1,0,1); b2 = FallingBasis(1,1,-1); n = b1.n()
            sage: B = ProductBasis([b1,b2]).scalar(1/factorial(n))

        This basis ``B`` contains the elements 

        .. MATH::

            \begin{matrix}
            \binom{x + n}{2n}\ \text{if }n\equiv 0\ (mod\ 2)\\
            \binom{x + n}{2n+1}\ \text{if }n\equiv 1\ (mod\ 2)
            \end{matrix}

        We first extend the compatibility with `E: x\mapsto x+1` by guessing and then we compute the product basis
        with itself::

            sage: B.set_compatibility('E', guess_compatibility_E(B, sections=2))
            sage: B2 = ProductBasis([B,B], ends=['E'])

        Now the basis ``B2`` is formed in 4 sections by the following elements:

        .. MATH::

            \begin{matrix}
                \binom{x+n}{2n}^2\ \text{if }n\equiv 0\ (mod\ 4)\\
                \binom{x+n}{2n}\binom{x+n}{2n+1}\ \text{if }n\equiv 1\ (mod\ 4)\\
                \binom{x+n}{2n+1}^2\ \text{if }n\equiv 2\ (mod\ 4)\\
                \binom{x+n+1}{2n+2}\binom{x+n}{2n+1}\ \text{if }n\equiv 3\ (mod\ 4)
            \end{matrix}

        We can check that ``B2`` is compatible with the multiplication by `x` and with 
        the endomorphism `E`::

            sage: a,b,m,alpha = B2.compatibility('x')
            sage: Matrix([[alpha(i,j,B2.n()) for j in range(-a,b+1)] for i in range(m)])
            [      n 2*n + 1]
            [      n 2*n + 1]
            [ -n - 1 2*n + 2]
            [ -n - 1 2*n + 2]
            sage: B2.recurrence('x')
            [        n         0         0 (2*n)*Sni]
            [(2*n + 1)         n         0         0]
            [        0 (2*n + 1)  (-n - 1)         0]
            [        0         0 (2*n + 2)  (-n - 1)]

        Now consider the following difference operator:

        .. MATH::

            L = (x+2)^3 E^2 - (2*x + 3)(17x^2+51x+39)E + (x+1)^3

        This operator `L` is compàtible with the basis ``B2``. We can get then
        the associated recurrence matrix. Taking the first column and the GCRD
        of its elements, we can see that if a formal power series `y(x) = \sum_n y_n x^n`
        that can be written in the form `y(x) = \sum_{n\geq 0}c_n\binom{x+n}{2n}^2` satisfies
        that

        .. MATH::

            (n+1)^2c_{n+1} - 4(2n+1)^2c_n = 0.

        Doing that with the code::

            sage: from ore_algebra import OreAlgebra
            sage: R.<x> = QQ[]; OE.<E> = OreAlgebra(R, ('E', lambda p : p(x=x+1), lambda p : 0))   
            sage: L = (x+2)^3 *E^2 - (2*x+3)*(17*x^2+51*x+39)*E+(x+1)^3
            sage: M = B2.recurrence(L)
            sage: column = [B2.remove_Sni(M.coefficient((j,0))) for j in range(4)]
            sage: column[0].gcrd(*column[1:])
            (n^2 + 2*n + 1)*Sn - 16*n^2 - 16*n - 4
    '''
    def __init__(self, factors, init=1, X='x', ends=[], ders=[]):
        super(ProductBasis, self).__init__(factors, list(range(len(factors))),init,X,ends,ders)

    @cached_method
    def increasing_basis(self, shift):
        r'''
            Method to get the structure for the `n`-th increasing basis.

            This method *overrides* the corresponding method from :class:`SievedBasis`.
            See method :func:`~SievedBasis.increasing_basis` for further information.

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
        F = self.nfactors(); factors = self.factors

        k, j = self.extended_quo_rem(shift, F)
        return ProductBasis(
            [factors[i].increasing_basis(k) for i in range(j, F)]+[factors[i].increasing_basis(k+1) for i in range(j)],
            X=self.var_name())

    def __repr__(self):
        return "ProductBasis" + "".join(["\n\t- %s" %repr(f) for f in self.factors])

    def _latex_(self):
        return "".join([f._latex_() for f in self.factors])
    
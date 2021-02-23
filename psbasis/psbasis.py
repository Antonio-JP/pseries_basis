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
from sage.all import FractionField, PolynomialRing, ZZ, QQ, Matrix, cached_method
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
        * :func:`~PSBasis._scalar_basis`.
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
            
            This method sets the operator given by ``name`` the translation rule
            given by ``trans``. The latter argument must be a compatible operator
            in the ring :func:`~PSBasis.OS`.
            
            See :arxiv:`1804.02964v1` for further information about the
            definition of a compatible operator.
            
            INPUT:
            
            * ``name``: the operator we want to set the compatibility. It can be the
                name for any generator in the *ore_algebra* package or the generator
                itself.
            * ``trans``: an operator that can be casted into the ring :func:`~PSBasis.OS`. This can
                be also be a matrix of operators, associating then the compatibility
                with its sections. This input can be a triplet ``(A, B, coeffs)``
                where the values of the lower bound, the upper bound and the coefficients
                are given. This method builds then the corresponding difference operator.
            * ``sub`` (optional): if set to True, the compatibility rule for ``name``
                will be updated even if the operator was already compatible.
        '''
        name = str(name)
        
        if(name in self.__compatibility and (not sub)):
            print("The operator %s is already compatible with this basis -- no changes are done" %name)
            return
        
        if(isinstance(trans, tuple)):
            if(len(trans) != 3):
                raise ValueError("The operator given has not the appropriate format: not a triplet")
            A, B, coeffs = trans
            if(len(coeffs) != A+B+1):
                raise ValueError("The operator given has not the appropriate format: list of coefficients of wrong size")
            Sn = self.Sn(); Sni = self.Sni(); n = self.n()
            coeffs = [self.OB()(el) for el in coeffs]
            trans = coeffs[A] + sum(coeffs[A-i](n=n+i)*Sn**i for i in range(1, A+1)) + sum(coeffs[A+i](n=n-i)*Sni**i for i in range(1, B+1))
             
        if(is_Matrix(trans)):
            trans = Matrix([
                [
                    self.reduce_SnSni(self.OS()(trans.coefficient((i,j))))
                for j in range(trans.ncols())]
            for i in range(trans.nrows())])
        else:
            trans = self.reduce_SnSni(self.OS()(trans))
            
        self.__compatibility[name] = trans
        
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
        compatibility = self.get_compatibility(operator)
        
        if(is_Matrix(compatibility)):
            raise NotImplementedError("The lower bound for matrix compatibilities is not implemented")
            
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
        compatibility = self.get_compatibility(operator)
        
        if(is_Matrix(compatibility)):
            raise NotImplementedError("The lower bound for matrix compatibilities is not implemented")
            
        return compatibility.degree(self.Sni())
        
    def compatible_operators(self):
        r'''
            Method that returns a list with the compatible operators stored in the dictionary.

            This method allows the user to know the names of the basic compatible operators with this 
            basis. Any polynomial built on these operators will be valid for the method :func:`get_compatibility`.

            OUTPUT:

            Return the key set of the dictionary of compatibilities. This set will be composed of the names of 
            the compatible operators with ``self``.

            TODO: add examples
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

            TODO: add examples
        '''
        
        return isinstance(operator, str) and operator in self.__compatibility

    def get_compatibility(self, operator):
        r'''
            Method to get the compatibility for an operator.
            
            This method returns the compatibility of an operator showing the associated
            sequence operator. In :arxiv:`1804.02964v1` this compatibility
            is shown to be an algebra isomorphism, so we can compute the compatibility
            final sequence operator using the ``ore_algebra`` package and a plain 
            substitution.
            
            INPUT:

            * ``operator``: the operator we want to get the compatibility. It has to be the
              name for any generator in an ``ore_algebra`` package or the generator
              itself.

            OUTPUT:

            An operator in the algebra returned by :func:`OS` that represents the compatibility
            condition of ``operator`` with the basis ``self``.

            TODO: add examples
        '''
        if(isinstance(operator,str)):
            if(not operator in self.__compatibility):
                raise ValueError("The operator %s is not compatible with %s" %(operator,self))
            else:
                return self.__compatibility[operator]
        else:
            try:
                poly = operator.polynomial()
            except TypeError:
                poly = operator

            return self.reduce_SnSni(poly(**self.__compatibility))

    @cached_method
    def get_compatibility_sections(self, size, operator):
        r'''
            Compute the matrix operator for the sections of a compatibility operator.
            
            This method computes a recurrence operator matrix for a basis
            associated with its compatibility with an operator. Such compatibility
            must be given as a method taking 3 parameters `k`,`j`,`i` such that:
            
            .. MATH::

                L\cdot b_{km+j} = \sum_{i=-A}^B \alpha_{k,j,i}b_{km+j+i}

            INPUT:

            * ``size``: the value of the section size (i.e., `m`).
            * ``operator``: this must be either a triplet with the values `A`, `B`
              and a function with three arguments (i,j,k) that returns the compatibility 
              coefficients in the appropriate order or a compatible operator with this basis.

            OUTPUT:

            A matrix representing the compatibility condition by sections. See :arxiv:`1804.02964v1`
            for further information. 

            TODO: add examples
        '''
        ## Considering the case of an operator
        if(not (isinstance(operator, tuple) or isinstance(operator, list))):
            aux = operator
            operator = (self.A(operator),self.B(operator),lambda k,j,i : self.compatibility_coefficient(aux, k*size+j, i))

        ## Checking the input
        if(len(operator) != 3):
            raise TypeError("The input must be a valid operator or a tuple with 3 elements")
        A,B,alpha = operator

        if(not size in ZZ or size <= 0):
            raise ValueError("The section size must be a positive integer (got %s)" %size)
        elif(not A in ZZ or A < 0):
            raise ValueError("The upper bound condition is not valid")
        elif(not B in ZZ or B < 0):
            raise ValueError("The lower bound condition is not valid")

        ## Creating an auxiliary function
        def SN(index):
            index = ZZ(index)
            if(index < 0):
                return self.Sni()**(-index)
            else:
                return self.Sn()**index

        ## Computing the operator matrix
        M = Matrix(
            [
                [self.reduce_SnSni(sum(
                    alpha(self.n()+(r-i-j)//size,j,i)*SN((r-i-j)//size) # we use exact division to avoid casting issues
                    for i in range(-A,B+1) if ((r-i-j)%size == 0)
                )) for j in range(size)
                ] for r in range(size)
            ])

        ## Returning the solution
        if(size == 1): # Case of section of size 1: we do not return a matrix
            return M.coefficient((0,0))
        return M
    
    @cached_method
    def compatibility_coefficient(self, operator, pos, ind):
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
        compatibility = self.get_compatibility(operator)
        A = self.A(operator); B = self.B(operator)
            
        if(is_Matrix(compatibility)):
            raise NotImplementedError("The coefficient for matrix compatibilities is not implemented")
        
        if(ind < 0 and -ind <= A):
            Sn = self.Sn()
            coeff = self.OB()(compatibility.polynomial().coefficient({Sn : -ind}))
            return coeff(n = pos + ind)
        elif(ind > 0 and ind <= B):
            Sni = self.Sni()
            coeff = self.OB()(compatibility.polynomial().coefficient({Sni : ind}))
            return coeff(n = pos + ind)
        elif(ind == 0):
            return self.OB()(compatibility.polynomial().constant_coefficient())
        else:
            return self.OB().zero()

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
        factor = self.OB()(factor) # we cast the factor into a rational function

        ## We check the denominator never vanishes on positive integers
        if(any((m > 0 and m in ZZ) for m in [root[0][0] for root in factor.denominator().roots()])):
            raise ValueError("The scalar factor is not valid: it takes value infinity at some 'n'")

        ## We check the numerator never vanishes on the positive integers
        if(any((m > 0 and m in ZZ) for m in [root[0][0] for root in factor.numerator().roots()])):
            raise ValueError("The scalar factor is not valid: it takes value 0 at some 'n'")

        new_basis = self._scalar_basis(factor) # we create the structure for the new basis

        compatibilities = [key for key in self.compatible_operators() if (not key in new_basis.compatible_operators())]
        for key in compatibilities:
            A = self.A(key); B = self.B(key); n = self.n()
            coeffs = [self.alpha(key, n, i)*(factor/factor(n=n+i)) for i in range(-A, B+1)]

            new_basis.set_compatibility(key, (A, B, coeffs))
        
        return new_basis

    def _scalar_basis(self, factor):
        r'''
            Method that actually builds the structure for the new basis.

            This method (that is abstract) build the actual structure for the new basis. This may have
            some intrinsic compatibilities that will be extended with the compatibilities that 
            are in ``self`` according with the factor.
        '''
        raise NotImplementedError("Method '_scalar_basis' must be implemented in each subclass of PSBasis")

    ### MAGIC METHODS
    def __getitem__(self, n):
        r'''
            See method :func:`element`
        '''
        return self.element(n)

    def __mul__(self,other):
        r'''
            See method :func:`scalar`.
        '''
        try:
            other = self.OB()(other)
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
    
class PolyBasis(PSBasis):
    r'''
        Abstract class for a polynomial power series basis. 
        
        Their elements must be indexed by natural numbers such that the n-th
        element of the basis has degree exactly `n`.
        
        This class **must never** be instantiated.

        List of abstract methods:

        * :func:`PSBasis.element`.
        * :func:`PSBasis._scalar_basis`.
    '''
    def __init__(self):
        super(PolyBasis,self).__init__(True)

    def basis_matrix(self, nrows, ncols):
        r'''
            Method that returns the matrix that make the change from the canonical basis
            `\{1, x, x^2, \ldots\}` to the current polynomial basis.
            
            All polynomials basis are based on elements `P_n(x)` of degree `n` for each
            value of `n \in \mathbb{N}`. This means that we can write:

            .. MATH::

                P_n(x) = \sum_{k=0}^n a_{n,k} x^k

            And by taking `a_{n,k} = 0` for all `k > 0`, we can build a matrix 
            `M = \left(a_{n,k}\right)_{n,k \geq 0}`, such that:

            .. MATH::

                \begin{pmatrix}P_0(x)\\P_1(x)\\P_2(x)\\P_3(x)\\\vdots\end{pmatrix} = 
                \begin{pmatrix}
                    a_{0,0} & 0 & 0 & 0 & \ldots\\
                    a_{1,0} & a_{1,1} & 0 & 0 & \ldots\\
                    a_{2,0} & a_{2,1} & a_{2,2} & 0 & \ldots\\
                    a_{3,0} & a_{3,1} & a_{3,2} & a_{3,3} & \ldots\\
                    \vdots&\vdots&\vdots&\vdots&\ddots
                \end{pmatrix} \begin{pmatrix}1\\x\\x^2\\x^3\\\vdots\end{pmatrix}

            This matrix is infinite, so this method only returns a bounded region of the matrix.

            INPUT:

            * ``nrows``: number of rows for the matrix
            * ``ncols``: number of columns for the matrix

            OUTPUT:
            
            The matrix `M = \left(a_{n,k}\right)_{0\leq n \leq nrows}^{0 \leq k \leq ncols}`.

            EXAMPLES::

                sage: from psbasis import *
                sage: B = BinomialBasis()
                sage: B.basis_matrix(5,5)
                [    1     0     0     0     0]
                [    0     1     0     0     0]
                [    0  -1/2   1/2     0     0]
                [    0   1/3  -1/2   1/6     0]
                [    0  -1/4 11/24  -1/4  1/24]
                sage: B.basis_matrix(3,7)
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
        * :func:`PSBasis._scalar_basis`.
    '''
    def __init__(self):
        super(OrderBasis,self).__init__(False)
    
    def __repr__(self):
        return "PolyBasis -- WARNING: this is an abstract class"

__all__ = ["PSBasis", "PolyBasis", "OrderBasis"]
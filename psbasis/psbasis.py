r'''
    Sage package for Power Series Basis.
'''

from sage.all_cmdline import *   # import sage library
from sage.structure.element import is_Matrix;

from ore_algebra import *;

class PSBasis(object):
    r'''
        Generic (abstract) class for a power series basis. 
        
        Their elements must be indexed by natural numbers and ordered by 
        `degree` or `order`.
        
        This class must never be instantiated.
    '''
    ### STATIC  IMMUTABLE ELEMENTS
    _OB = FractionField(PolynomialRing(QQ, ['n']));
    _n = _OB.gens()[0];
    _OS = None;
    _OSS = None;
    _Sn = None;
    _Sni = None;

    @property
    def OB(self):
        return PSBasis._OB;

    @property
    def n(self):
        return PSBasis._n;

    @property
    def OS(self):
        if(PSBasis._OS is None):
            PSBasis._OS = OreAlgebra(self.OB, ('Sn', lambda p: p(n=self.n+1), lambda p : 0), ('Sni', lambda p : p(n=self.n-1), lambda p : 0));
        return PSBasis._OS;

    @property
    def OSS(self):
        if(PSBasis._OSS is None):
            PSBasis._OSS = OreAlgebra(self.OB, ('Sn', lambda p: p(n=self.n+1), lambda p : 0));
        return PSBasis._OSS;

    @property
    def Sn(self):
        if(PSBasis._Sn is None):
            PSBasis._Sn = self.OS.gens()[0];
        return PSBasis._Sn;

    @property
    def Sni(self):
        if(PSBasis._Sni is None):
            PSBasis._Sni = self.OS.gens()[1];
        return PSBasis._Sni;
    
    ### CONSTRUCTOR
    def __init__(self, degree=True):
        self.__degree = degree;
        self.__compatibility = {};
    
    ### BASIC METHODS
    def get_element(self, n, var_name=None):
        r'''
            Method to return the $n$th element of the basis.
            
            This method must return the $n$th element for this series. This means
            a power series with degree $n$ if ``self.by_degree()`` or of order
            $n$ if ``self.by_order()``.
        '''
        raise NotImplementedError("Method get_element must be implemented in each subclass of polynomial_basis");
        
    def by_degree(self):
        r'''
            Getter for the type of ordering of the basis.
            
            Return ``True`` if the $n$th element of the basis is a polynomial
            of degree $n$.
        '''
        return self.__degree;
    
    def by_order(self):
        
        '''
            Getter for the type of ordering of the basis.
            
            Return ``True`` if the $n$th element of the basis is a power series
            of order $n$.
        '''
        return (not self.__degree);

    ### AUXILIARY METHODS
    def reduce_SnSni(self,operator):
        r'''
            Method to reduce operators with Sn and Sni. 

            Operators in Sn and Sni can be usually simplified until we obtain a Laurent polynomial where Sni = Sn^{-1}.
            Since they are inverses of each other, Sn*Sni = Sni*Sn = 1.
        '''
        if(is_Matrix(operator)):
            base = operator.parent().base(); n = operator.nrows();
            return Matrix(base, [[self.reduce_SnSni(operator.coefficient((i,j))) for j in range(n)] for i in range(n)]);
        operator = self.OS(str(operator));
        Sn = self.Sn; Sni = self.Sni;

        poly = operator.polynomial();
        monomials = poly.monomials();
        coefficients = poly.coefficients();
        result = operator.parent().zero(); 

        for i in range(len(monomials)):
            d1,d2 = monomials[i].degrees();
            if(d1 > d2):
                result += coefficients[i]*Sn**(d1-d2);
            elif(d2 > d1):
                result += coefficients[i]*Sni**(d2-d1);
            else:
                result += coefficients[i];
        return result;
    
    def polynomial_ring(self,var_name='x'):
        return PolynomialRing(self.OB.base(), [var_name]);
    
    ### COMPATIBILITY RELATED METHODS
    def set_compatibility(self, name, trans, sub=False):
        r'''
            Method to set a new compatibility operator.
            
            This method sets the operator given by ``name`` the translation rule
            given by ``trans``. The latter argument must be a compatible operator
            of self.OS.
            
            See https://arxiv.org/abs/1804.02964v1 for further information about the
            definicion of a compatible operator.
            
            INPUT::
                - ``name``: the operator we want to set the compatibility. It can be the
                  name for any generator in the *ore_algebra* package or the generator
                  itself.
                - ``trans``: an operator that can be casted into self.OS. This can
                  be also be a matrix of operators, associating then the compatibility
                  with its sections.
                - ``sub`` (optional): if set to True, the compatibility rule for ``name``
                  will be updated even if the operator was already compatible.
        '''
        name = str(name);
        
        if(name in self.__compatibility and (not sub)):
            raise ValueError("The operator is already compatible with this basis");
        
        if(is_Matrix(trans)):
            trans = Matrix([
                [
                    self.reduce_SnSni(self.OS(trans.coefficient((i,j))))
                for j in range(trans.ncols())]
            for i in range(trans.nrows())]);
        else:
            trans = self.reduce_SnSni(self.OS(trans));
            
        self.__compatibility[name] = trans;
        
    @cached_method
    def get_lower_bound(self, operator):
        r'''
            Method to get the lower bound compatibility for an operator.
            
            This method returns the minimal index for the compatibility property
            for a particular operator. In the notation of the paper
            https://arxiv.org/abs/1804.02964v1, for a $(A,B)$-compatible operator,
            this lower bound corresponds to the value of $A$.
            
            INPUT::
                - ``operator``: the operator we want to check. It can be the
                  name for any generator in the *ore_algebra* package or the generator
                  itself.
                
            WARNING::
                - The case when the compatibility rule is a matrix is not implemented.
        '''
        ## Case of the name
        compatibility = self.get_compatibility(operator);
        
        if(is_Matrix(compatibility)):
            raise NotImplementedError("The lower bound for matrix compatibilities is not implemented");
            
        return compatibility.degree(self.Sn);
    
    @cached_method
    def get_upper_bound(self, operator):
        r'''
            Method to get the upper bound compatibility for an operator.
            
            This method returns the maximal index for the compatibility property
            for a particular operator. In the notation of the paper
            https://arxiv.org/abs/1804.02964v1, for a $(A,B)$-compatible operator,
            this lower bound corresponds to the value of $B$.
            
            INPUT::
                - ``operator``: the operator we want to check. It can be the
                  name for any generator in the *ore_algebra* package or the generator
                  itself.
                
            WARNING::
                - The case when the compatibility rule is a matrix is not implemented.
        '''
        compatibility = self.get_compatibility(operator);
        
        if(is_Matrix(compatibility)):
            raise NotImplementedError("The lower bound for matrix compatibilities is not implemented");
            
        return compatibility.degree(self.Sni);
        
    @cached_method
    def get_compatibility(self, operator):
        r'''
            Method to get the compatibility for an operator.
            
            This method returns the compatibility of an operator showing the associated
            sequence operator. In https://arxiv.org/abs/1804.02964v1 this compatibility
            is shown to be an algebra isomorphism, so we can compute the compatibility
            final sequence operator using the ore_algebra package and a plain 
            substitution.
            
            INPUT::
                - ``operator``: the operator we want to get the compatibility. It can be the
                  name for any generator in the *ore_algebra* package or the generator
                  itself.
        '''
        if(isinstance(operator,str)):
            if(not operator in self.__compatibility):
                raise ValueError("The operator %s is not compatible with %s" %(operator,self));
            else:
                return self.__compatibility[operator];
        else:
            return self.reduce_SnSni(operator.polynomial()(**self.__compatibility));

    @cached_method
    def get_compatibility_sections(self, size, operator):
        '''
            Compute the matrix operator for the sections of a compatibility operator.
            
            This method computes a recurrence operator matrix for a basis
            associated with its compatibility with an operator. Such compatibility
            must be given as a method taking 3 parameters k,j,i and then:
            
            $$L(b_{km+j}) = \sum_{i=-A}^B compatibility(k,j,i)b_{km+j+i}$$

            INPUT::
                * ``size``: the value of the section size (i.e., $m$).
                * ``operator``: this must be either a tuple with the values $A$, $B$
                  and a function with three arguments (i,j,k) that returns the compatibility 
                  coefficients in the appropiate order or a compatible operator with this basis.
        '''
        ## Considering the case of an operator
        if(not (isinstance(operator, tuple) or isinstance(operator, list))):
            aux = operator;
            operator = (self.A(operator),self.B(operator),lambda k,j,i : self.compatibility_coefficient(aux, k*size+j, i));

        ## Checking the input
        if(len(operator) != 3):
            raise TypeError("The input must be a valid operator or a tuple with 3 elements");
        A,B,alpha = operator;

        if(not size in ZZ or size <= 0):
            raise ValueError("The section size must be a positive integer (got %s)" %m);
        elif(not A in ZZ or A < 0):
            raise ValueError("The upper bound condition is not valid");
        elif(not B in ZZ or B < 0):
            raise ValueError("The lower bound condition is not valid");

        ## Creating an auxiliary function
        def SN(index):
            index = ZZ(index);
            if(index < 0):
                return self.Sni**(-index);
            else:
                return self.Sn**index;

        ## Computing the operator matrix
        M = Matrix(
            [
                [self.reduce_SnSni(sum(
                    alpha(self.n+(r-i-j)/size,j,i)*SN((r-i-j)/size) 
                    for i in range(-A,B+1) if ((r-i-j)%size == 0)
                )) for j in range(size)
                ] for r in range(size)
            ]);

        ## Returning the solution
        if(size == 1): # Case of section of size 1: we do not return a matrix
            return M.coefficient((0,0));
        return M;
    
    @cached_method
    def compatibility_coefficient(self, operator, pos, ind):
        r'''
            Method to get the compatibility coefficient.
            
            Following https://arxiv.org/abs/1804.02964v1, an operator $L$ is
            $(A,B)$-compatible if there are some $\alpha_{n,i}$ such that for all $n$
            $$L(P_n) = \sum_{i=-A}^B \alpha_{n,i}P_{n+i}.$$
            
            This method returns, for the operator given by ``name``, the value 
            $\alpha_{n,i}$ where $n$ is ``pos`` and $i$ is ``ind``.
            
            INPUT::
                - ``operator``: the operator we want to set the compatibility. It can be the
                  name for any generator in the *ore_algebra* package or the generator
                  itself.
                - ``pos``: position of the compatibility. This is the $n$ in the
                  definicion of compatibility. 
                - ``ind``: index of the compatibility coefficient. If it is bigger than
                  ``self.get_upper_bound(operator)`` or smaller than ``self.get_lower_bound(operator)``
                  then 0 is returned.
                
            WARNING::
                - The case when the compatibility rule is a matrix is not implemented.
        '''
        compatibility = self.get_compatibility(operator);
        A = self.A(operator); B = self.B(operator);
            
        if(is_Matrix(compatibility)):
            raise NotImplementedError("The coefficient for matrix compatibilities is not implemented");
        
        if(ind < 0 and -ind <= A):
            Sn = self.Sn;
            coeff = self.OB(compatibility.polynomial().coefficient({Sn : -ind}));
            return coeff(n = pos + ind);
        elif(ind > 0 and ind <= B):
            Sni = self.Sni;
            coeff = self.OB(compatibility.polynomial().coefficient({Sni : ind}));
            return coeff(n = pos + ind);
        elif(ind == 0):
            return self.OB(compatibility.polynomial().constant_coefficient());
        else:
            return self.OB.zero();

    ### MADIC METHODS
    def __getitem__(self, n):
        return self.get_element(n);
    
    ### MAGIC REPRESENTATION METHODS
    def __repr__(self):
        return "PSBasis -- WARNING: this is an abstract class";
    
    ### OTHER ALIASES FOR METHODS
    A = get_lower_bound;
    B = get_upper_bound;
    alpha = compatibility_coefficient;
    
class PolyBasis(PSBasis):
    r'''
        Abstract class for a polynomial power series basis. 
        
        Their elements must be indexed by natural numbers such that the n-th
        element of the basis has degree exactly `n`.
        
        This class must never be instantiated.
    '''
    def __init__(self):
        super(PolyBasis,self).__init__(True);
    
    def __repr__(self):
        return "PolyBasis -- WARNING: this is an abstract class";

class OrderBasis(PSBasis):
    r'''
        Abstract class for a order power series basis. 
        
        Their elements must be indexed by natural numbers such that the n-th
        element of the basis has order exactly `n`.
        
        This class must never be instantiated.
    '''
    def __init__(self):
        super(OrderBasis,self).__init__(False);
    
    def __repr__(self):
        return "PolyBasis -- WARNING: this is an abstract class";
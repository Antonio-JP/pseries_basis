r'''
    Sage package for Factorial Series Basis.
'''

from sage.all_cmdline import *   # import sage library

from .psbasis import *

class FactorialBasis(PolyBasis):
    r'''
        Abstract class for a factorial basis.

        A factorial basis is a type of polynomial basis for power series where
        the $(n+1)$th element is build from the $n$th element. This can be seeing
        as a two-term recurrence basis.

        It provides several functionalities and methods that all Factorial Basis
        must provide, but may differ in the implementation.

        INPUT::
            - ``X``: the name for the operator representing the multiplication by $x$.
    '''
    def __init__(self, X='x'):
        super(FactorialBasis,self).__init__();

        self.__var_name = X;

    ## Getters and setters
    def var_name(self):
        return self.__var_name;

    def root_sequence(self):
        r'''
            Method that returns the root sequence of the polynomial basis.

            Since a factorial basis satisties that `P_n(x)` divides `P_{n+1}(x)` for all
            `n`, we have that the basis forms a sequence of polynomials with a persistent
            set of roots.

            We can then define the root sequence with `\rho_n` the new root in the polynomial
            `P_{n+1}(x)`.
        '''
        return lambda n : (self.get_element(n+1)/self.get_element(n)).numerator().roots()[0][0];

    ## Method related with equivalence of Proposition 1
    def increasing_polynomial(self, *args, **kwds):
        raise NotImplementedError("Method from FactorialBasis not implemented (Abstract method)");

    def matrix_ItP(self, *args, **kwds):
        raise NotImplementedError("Method from FactorialBasis not implemented (Abstract method)");

    @cached_method
    def matrix_PtI(self, *args, **kwds):
        r'''
            Method for getting the matrix from the power basis to the increasing basis.

            In a Factorial Basis, the $n$th element of the basis divides all the following.
            This means for any pair of indices $m > n$, there is a particular polynomial
            $Q_{n,m} = P_m/P_n$.

            In particular, for a fixed $n$ and $i \in \mathfrak{Z}$, the polynomials $Q_{n,n+i}$
            form another Factorial Basis. This method computes a matrix that represents the 
            identity between polynomials of degree smaller or equal to ``size`` from the
            power basis to the basis $Q_{n,n+i}$.

            For further information about the input, see the documentation of ``self.matrix_ItP``.
        '''
        return self.matrix_ItP(*args, **kwds).inverse();

    def equiv_DtC(self, *args, **kwds):
        raise NotImplementedError("Method from FactorialBasis not implemented (Abstract method)");

    def equiv_CtD(self, *args, **kwds):
        raise NotImplementedError("Method from FactorialBasis not implemented (Abstract method)");
    
    def __repr__(self):
        return "FactorialBasis -- WARNING: this is an abstract class";
    
class SFactorialBasis(FactorialBasis):
    r'''
        Class for representing a simple factorial basis.
        
        A factorial basis is a type of polynomial basis for power series where
        the $(n+1)$th element is build from the $n$th element. This can be seeing
        as a two-term recurrence basis.
        
        The first element in the sequence will always be the constant polynomial 1.
        
        This factorial nature is representing using two coefficients $a_n$ and $b_n$
        such that for all $n$:
        $$P_n = (a_nx + b_n)P_{n-1}$$
        
        INPUT::
            - ``an``: the sequence of leading coefficients to build the factorial basis.
            - ``bn``: the sequence of constant coefficients to build the factorial basis.
            - ``X``: the name for the operator representing the multiplication by $x$.
    '''
    def __init__(self, an, bn, X='x'):
        ## Initializing the PolyBasis strcuture
        super(SFactorialBasis,self).__init__(X);
        
        ## Adding the extra information
        an = self.OB(an); self.__an = an; 
        bn = self.OB(bn); self.__bn = bn;
        self.__var_name = X;
        
        ## The multiplication by X compatibility is given
        Sni = self.Sni; n = self.n;
        self.set_compatibility(X, -bn(n=n+1)/an(n=n+1) + (1/an)*Sni);

        ## Extra cached variables
        self.__cached_increasing = {};
        
    # PSBasis abstract method
    @cached_method
    def get_element(self, n, var_name=None):
        if(var_name is None):
            name = self.__var_name;
        else:
            name = var_name;
        R = self.polynomial_ring(name);
        x = R.gens()[0];
        
        if(n > 0):
            an = self.__an; bn = self.__bn;
            return self.get_element(n-1) * (an(n=n)*x + bn(n=n));
        elif(n == 0):
            return R.one();

    # Override from FactorialBasis
    def root_sequence(self):
        return lambda n : -self.__bn(n=n+1)/self.__an(n=n+1);

    def constant_coefficient(self):
        r'''
            Getter for the constant coefficient of the factorial basis.

            This method return a sequence (in n) for the constant coefficient of the
            increasing polynomial for the Factorial Basis. Recall that for any Factorial
            Basis, the $n$th element divide the following, so we have:
                $$P_n = (a_nx + b_n)P_{n-1}$$

            This method returns the value of $b_n$.
        '''
        return self.__bn;

    def linear_coefficient(self):
        r'''
            Getter for the constant coefficient of the factorial basis.

            This method return a sequence (in n) for the constant coefficient of the
            increasing polynomial for the Factorial Basis. Recall that for any Factorial
            Basis, the $n$th element divides the following, so we have:
                $$P_n = (a_nx + b_n)P_{n-1}$$

            This method returns the value of $a_n$.
        '''
        return self.__an;

    # FactorialBasis abstract method
    def increasing_polynomial(self, src, diff=None, dst=None):
        r'''
            Method to get the increasing polynomial given the appropriate indices.

            In a Factorial Basis, the $n$th element of the basis divides all the following.
            This means for any pair of indices $m > n$, there is a particular polynomial
            $Q_{n,m} = P_m/P_n$.

            This method computes such polynomial where $n = src$ and $m = src+diff$ or 
            $m = dst$. Depending which one is given.

            INPUT::
                - ``src``: value for $n$.
                - ``diff``: difference between $n$ and $m$. Must be a positive integer.
                - ``dst``: value for $m$. Only used (and required) if ``diff`` is None. Must 
                  be bigger than $n$.
        '''
        ## Checking the arguments
        if(((src in ZZ) and src < 0) or (not src in self.OB)):
            raise ValueError("The argument `src` must be a expression involving `self.n` or a positive integer");
        n = src;

        if(not diff is None):
            if((not diff in ZZ) or diff < 0):
                raise ValueError("The argument `diff` must be None or a positive integer");
            else:

                d = ZZ(diff); m = n + d;
        else:
            if(n in ZZ):
                if((not dst in ZZ) or dst < n):
                    raise ValueError("The argument `dst` must be an integer bigger than `src`");
                m = ZZ(dst); d = m - n;
            else:
                d = dst-n;
                if((not d in ZZ) or d < 0):
                    raise ValueError("The difference between `dst` and `src` must be a positive integer");
                d = ZZ(d); m = dst;

        ## Building the polynomial
        PR = self.polynomial_ring(self.__var_name); x = PR.gens()[0];
        if(d == 0):
            return PR.one();

        if(not (n,d) in self.__cached_increasing):
            n_name = str(self.n);

            self.__cached_increasing[(n,d)] = prod(self.constant_coefficient()(**{n_name : n+i}) + x*self.linear_coefficient()(**{n_name : n+i}) for i in range(1,d+1));

        return self.__cached_increasing[(n,d)];

    def applied_division_polynomial(self, operator, src, diff=None, dst=None):
        r'''
            Method to get the division of a polynomial by other element of the basis after an operator.

            In a Factorial Basis, the $n$th element of the basis divides all the following.
            This means for any pair of indices $m > n$, there is a particular polynomial
            $Q_{n,m} = P_m/P_n$.

            Moreover, by Proposition 1 of https://arxiv.org/abs/1804.02964v1, for a fixed operator $L$
            that is $(A,B)$-compatible, we know that $P_{n-A}$ divides $L(P_n)$.

            This method computes that division where $n = src$ and $m = src+diff$ or 
            $m = dst$ with $m < n=-A$. Depending which one is given.

            INPUT::
                - ``src``: value for $n$.
                - ``operator``: the operator we want to check. It can be the
                  name for any generator in the *ore_algebra* package or the generator
                  itself.
                - ``diff``: difference between $n$ and $m$. Must be a positive integer.
                - ``dst``: value for $m$. Only used (and required) if ``diff`` is None. Must 
                  be bigger than $n$.
        '''
        ## Checking the arguments
        ## Reading ``src``
        if(((src in ZZ) and src < 0) or (not src in self.OB)):
            raise ValueError("The argument `src` must be a expression involving `self.n` or a positive integer");
        n = src;

        ## Compatibility of ``operator``
        A = self.A(operator); B = self.B(operator);

        ## Reading ``diff`` or ``dst``
        if(not diff is None):
            if((not diff in ZZ) or diff < A):
                raise ValueError("The argument `diff` must be None or a positive integer bigger than %s" %A);
            else:
                d = ZZ(diff); m = n - d;
        else:
            if(n in ZZ):
                if((not dst in ZZ) or dst < n-A):
                    raise ValueError("The argument `dst` must be an integer bigger than `src`");
                m = ZZ(dst); d = n - m;
            else:
                d = n-dst;
                if((not d in ZZ) or d < A):
                    raise ValueError("The difference between `dst` and `src` must be a positive integer bigger than %s" %A);
                d = ZZ(d); m = dst;

        ## Computing the polynomial
        return sum(self.compatibility_coefficient(operator, n, i-A)*self.increasing_polynomial(m, diff=i) for i in range(A+B+1));

    # FactorialBasis abstract method
    @cached_method
    def matrix_ItP(self, src, size):
        r'''
            Method to get the matrix for converting from the increasing basis to the power basis.

            In a Factorial Basis, the $n$th element of the basis divides all the following.
            This means for any pair of indices $m > n$, there is a particular polynomial
            $Q_{n,m} = P_m/P_n$.

            In particular, for a fixed $n$ and $i \in \mathfrak{N}$, the polynomials $Q_{n,n+i}$
            form another Factorial Basis. This method computes a matrix that represents the 
            identity between polynomials of degree smaller or equal to ``size`` from the
            basis $Q_{n,n+i}$ and the power basis.

            INPUT::
                - ``src``: value for $n$.
                - ``size``: bound on the degree for computing the matrix.
        '''
        if(((src in ZZ) and src < 0) or (not src in self.OB)):
            raise ValueError("The argument `src` must be a expression involving `self.n` or a positive integer");
        n = src;
        if(n in ZZ):
            n =  ZZ(n); dest = QQ;
        else:
            dest = self.OB;

        if((not size in ZZ) or size <= 0):
            raise ValueError("The argument `size` must be a positive integer");

        ## Computing the matrix
        polys = [self.increasing_polynomial(n,diff=i) for i in range(size)];
        return Matrix(dest, [[polys[j][i] for j in range(size)] for i in range(size)]);

    # FactorialBasis abstract method
    def equiv_DtC(self, bound, *coeffs):
        r'''
            Method to get the equivalence condition for a compatible operator.

            Folloring the notation and ideas of https://arxiv.org/abs/1804.02964v1, there is an
            equivalent condition to be a compatible operator. Namely, and operator is compatible
            by definition if it expands:
                $$L(P_n) = \sum_{i=-A}^B \alpha_{n,i}P_{n+i},$$
            and that is equivalent to the following two conditions:
                - $\deg(L(P_n)) \leq n + B$,
                - $P_{n-A}$ divides $L(P_n)$.

            This method takes the list of $\alpha_{n,i}$ and computes
            the division $L(P_n)/P_{n-A}$ for $n \geq A$ as a polynomial of degree $A+B$.

            INPUT::
                - ``bound``: value for $A$.
                - ``coeffs``: list of coefficients in ``self.OB`` representing the coefficients
                  $\alpha_{n,i}$, i.e., $coeffs[j] = \alpha_{n,j-A}$.

            OUPUT::
                List of coefficients of $L(P_n)/P_{n-A}$.
        '''
        ## Checking the input parameters 
        if((not bound in ZZ) or (bound < 0)):
            raise ValueError("The argument `bound` must be a positive integer");
        if(len(coeffs) ==  1 and (type(coeffs) in (tuple, list))):
            coeffs = coeffs[0];
        A = ZZ(bound); B = len(coeffs) - A - 1; n = self.n;

        ## At this point we have that `coeffs` is the list of coefficients of 
        ## L(P_n)/P_{n-A} in the increasing basis starting with $n-A$.
        ## We only need to change the basis to the Power Basis
        new_alpha = self.matrix_ItP(n-A, A+B+1)*vector(coeffs);

        return [el for el in new_alpha];

    # FactorialBasis abstract method
    def equiv_CtD(self, bound, *coeffs):
        r'''
            Method to get the equivalence condition for a compatible operator.

            Folloring the notation and ideas of https://arxiv.org/abs/1804.02964v1, there is an
            equivalent condition to be a compatible operator. Namely, and operator is compatible
            by definition if it expands:
                $$L(P_n) = \sum_{i=-A}^B \alpha_{n,i}P_{n+i},$$
            and that is equivalent to the following two conditions:
                - $\deg(L(P_n)) \leq n + B$,
                - $P_{n-A}$ divides $L(P_n)$.

            This method takes the division $L(P_n)/P_{n-A}$ as a list of its $A+B+1$ coefficients and, 
            computes the compatibility coefficients for the operator $L$.

            INPUT::
                - ``bound``: value for $A$.
                - ``coeffs``: list of coefficients in ``self.OB`` representing the coefficients
                  of the polynomial $L(P_n)/P_{n-A}$ in the usual Power Basis.

            OUPUT::
                List of coefficients of $\alpha_{n,i}$.
        '''
        ## Checking the input parameters 
        if((not bound in ZZ) or (bound < 0)):
            raise ValueError("The argument `bound` must be a positive integer");
        if(len(coeffs) ==  1 and (type(coeffs) in (tuple, list))):
            coeffs = coeffs[0];
        A = ZZ(bound); B = len(coeffs) - A - 1; n = self.n;

        ## At this point we have that `coeffs` is the list of coefficients of 
        ## L(P_n)/P_{n-A} in the power basis. If we change to the increasing
        ## basis starting in $n-A$ then we have the $alpha_{n,i}$.
        new_alpha = self.matrix_PtI(n-A, A+B+1)*vector(coeffs);

        return [el for el in new_alpha];

###############################################################
### EXAMPLES OF PARTICULAR FACTORIAL BASIS
###############################################################
class FallingBasis(SFactorialBasis):
    r'''
        Class for the Falling factorial Basis $(1, (ax+b), (ax+b)(ax+b-c), (ax+b)(ax+b-c)(ax+b-2c), \dots))$.

        This class represent the FactorialBasis formed by the falling factorial basis
        for the power series ring with two extra paramenters $a$ and $b$:
            $$1,\quad (ax+b),\quad (ax+b)(ax+b-c),\quad (ax+b)(ax+b-c)(ax+b-2c),\dots$$

        In the case of $a = 1$, $b = 0$ and $c = 0$, we have the usual power basis (see class 
        PowerBasis) and in the case of $a=1$ $b = 0$ and $c = \pm 1$ we have the falling (or
        raising) factorial basis.

        Following the notation in https://arxiv.org/abs/1804.02964v1, this basis
        has compatibilities with the multiplication by 'x' and with the isomorphism
        $E_c: x \rightarrow x+c$. 

        INPUT::
            - ``dilation``: the natural number corresponfing to the value $a$.
            - ``shift``: the shift corresponding to the value $b$.
            - ``decay``: the value for $c$
            - ``X``: the name for the operator representing the multiplication by $x$.
            - ``E``: the name for the operator representing the derivation by $x$.
    '''
    def __init__(self, dilation, shift, decay, X='x', E=None):
        if(not dilation in ZZ or dilation <= 0):
            raise ValueError("The dilation of the basis ust be a natural number");
        dilation = ZZ(dilation); shift = self.OB.base()(shift); decay = self.OB.base()(decay);
        self.__a = dilation; a = self.__a;
        self.__b = shift; b = self.__b;
        self.__c = decay; c = self.__c;

        if(E is None):
            if(c == 1):
                E_name = "E";
            else:
                E_name = "E_%s" %b;
        else:
            E_name = E;
        
        n = self.n;
        super(FallingBasis, self).__init__(a, b-c*(n-1), X);
        
        Sn = self.Sn;
        aux_PR = self.polynomial_ring(X); x = aux_PR.gens()[0];
        aux_OE = OreAlgebra(aux_PR, (E_name, lambda p : p(x=x+b), lambda p: 0));
        P = aux_OE(prod(a*x+b-c*i for i in range(-a,0)));
        self.set_compatibility(E_name, 0);
        self.set_compatibility(E_name, self.get_compatibility(P)*(Sn**a), True);
        
    def __repr__(self):
        a = self.__a; b = self.__b; c = self.__c;
        x = self.polynomial_ring(self.var_name()).gens()[0];
        if(c == -1):
            return "Raising Factorial Basis (1, %s, %s(%s+1),...)" %(self.get_element(1), self.get_element(1), self.get_element(1));
        elif(c == 1):
            return "Falling Factorial Basis (1, %s, %s(%s-1),...)" %(self.get_element(1), self.get_element(1), self.get_element(1));
        return "General (%s,%s,%s)-Falling Factorial Basis (%s, %s,%s,...)" %(a,b,c,self.get_element(0), self.get_element(1), self.get_element(2));
    
    def _latex_(self):
        a = self.__a; b = self.__b; c = self.__c;
        x = self.polynomial_ring(self.var_name()).gens()[0];
        if(c == -1):
            return r"\left\{(%s)^{\upperline{n}}\right\}_{n \geq 0}" %self.get_element(1);
        elif(c == 1):
            return r"\left\{(%s)^{\underline{n}}\right\}_{n \geq 0}" %self.get_element(1);
        return r"\left\{(%s)^{\underline{n}_%s}\right\}_{n \geq 0}" %(a*x, b);
    
class PowerBasis(FallingBasis):
    r'''
        Class for the Power Basis $(1,x,x^2,\dots)$.
        
        This class represents the FactorialBasis formed by the simplest basis
        for the power series: $1$, $(ax+b)$, $(ax+b)^2$, etc.
        
        Following the notation in https://arxiv.org/abs/1804.02964v1, this basis
        corresponds with $\mathfrak{P}_{a,b}$. In that paper we can find that this basis
        has compatibilities with the multiplication by 'x' and with the derivation
        by 'x'.
        
        INPUT::
            - ``dilation``: the natural number corresponfing to the value $a$.
            - ``shift``: the shift corresponding to the value $b$.
            - ``X``: the name for the operator representing the multiplication by $x$.
            - ``Dx``: the name for the operator representing the derivation by $x$.
    '''
    def __init__(self, dilation=1, shift=0, X='x', Dx='Dx'):
        super(PowerBasis, self).__init__(dilation,shift,0,X,'Id');
        
        n = self.n; Sn = self.Sn; a = self.linear_coefficient();
        self.set_compatibility(Dx, a*(n+1)*Sn);
        
    def __repr__(self):
        a = self.linear_coefficient(); b = self.constant_coefficient();
        if(a == 1 and b == 0):
            return "Power Basis %s^n" %self.get_element(1);
        else:
            return "(%s,%s)-Power Basis (%s)^n" %(a,b,self.get_element(1));
    
    def _latex_(self):
        a = self.linear_coefficient(); b = self.constant_coefficient();
        if(a == 1 and b == 0):
            return r"\left\{%s^n\right\}_{n \geq 0}" %self.get_element(1);
        else:
            return r"\left\{(%s)^n\right\}_{n \geq 0}" %self.get_element(1);
    
class BinomialBasis(SFactorialBasis):
    r'''
        Class for the generic binomial basis.
        
        This class represents a binomial basis with a shitf and dilation effect on the
        top variable. Namely, a basis of the form
        $$\binom{ax+b}{n},$$
        where $a$ is a natural number and $b$ is a rational number.
        
        In https://arxiv.org/abs/1804.02964v1 this corresponds to $\mathfrak{C}_{a,b}$
        and it is compatible with the multiplication by $x$ and by the shift operator
        $x \rightarrow x+1$.
        
        INPUT::
            - ``dilation``: the natural number corresponfing to the value $a$.
            - ``shift``: the shift corresponding to the value $b$.
            - ``X``: the name for the operator representing the multiplication by $x$.
            - ``E``: the name for the operator representing the derivation by $x$.
    '''
    def __init__(self, dilation=1, shift=0, X='x', E='E'):
        if(not dilation in ZZ or dilation <= 0):
            raise ValueError("The dilation of the basis ust be a natural number");
        dilation = ZZ(dilation); shift = self.OB.base()(shift);
        self.__a = dilation; a = self.__a;
        self.__b = shift; b = self.__b;
        
        n = self.n;
        super(BinomialBasis, self).__init__(a/n, (b-n + 1)/n, X);
        
        Sn = self.Sn;
        self.set_compatibility(E, sum(binomial(a, i)*Sn**i for i in range(a+1)));
    
        
    def __repr__(self):
        x = self.polynomial_ring(self.var_name()).gens()[0];
        return "Binomial basis (%s) choose n" %(self.__a*x + self.__b);
    
    def _latex_(self):
        x = self.polynomial_ring(self.var_name()).gens()[0];
        return r"\left\{\binom{%s}{n}\right\}_{n\geq 0}" %(self.__a*x+self.__b);

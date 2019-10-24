r'''
    Sage package for Orthogonal Series Basis.
'''

from sage.all_cmdline import *   # import sage library

from .psbasis import *;

class OrthogonalBasis(PolyBasis):
    r'''
        Class for representing a basis of orthogonal polynomials.
        
        A basis of orthogonal polynomials is a type of polynomial basis for power series 
        where the $(n+2)$th element is build from the $n$th and $n+1$ elements. This can be 
        seeing as a three-term recurrence basis. See https://dlmf.nist.gov/18 for further
        information and formulas for orthogonal polynomials.
        
        The first element in the sequence will always be the constant polynomial 1.

        The second element in the sequence is such that the recurrence still holds when taking
        $P_{-1} = 0$.

        We represent a orthogonal polynomial basis with the three coefficients required for
        the three term recurrence:
            $$P_{n+1} = (a_n x + b_n)P_n - c_nP_{n-1}.$$
        
        INPUT::
            - ``an``: the first coefficient of the three term recurrence.
            - ``bn``: the second coefficient of the three term recurrence.
            - ``cn``: the third coefficient of the three term recurrence.
            - ``X``: the name for the operator representing the multiplication by $x$.
    '''
    def __init__(self, an, bn, cn, X='x'):
        ## Initializing the PolyBasis strcuture
        super(OrthogonalBasis,self).__init__();
        
        ## Adding the extra information
        an = self.OB(an); self.__an = an; 
        bn = self.OB(bn); self.__bn = bn;
        cn = self.OB(cn); self.__cn = cn;
        self.__var_name = X;
        
        ## The multiplication by X compatibility is given
        Sni = self.Sni; n = self.n; Sn = self.Sn;
        self.set_compatibility(X, (cn(n=n+1)/an(n=n+1))*Sn - bn/an + (1/an(n=n-1))*Sni);
        
    @cached_method
    def get_element(self, n, var_name=None):
        if(var_name is None):
            name = self.__var_name;
        else:
            name = var_name;
        R = self.polynomial_ring(name);
        x = R.gens()[0];
        an = self.__an; bn = self.__bn; cn = self.__cn;
        
        ## Basic cases
        if(n == 0):
            return R.one();
        elif(n == 1):
            return an(n=0)*x + bn(n=0);
        else: # General (recursive) case
            return (an(n=n-1)*x + bn(n=n-1))*self.get_element(n-1, name) - cn(n=n-1)*self.get_element(n-2, name);

    def var_name(self):
        return self.__var_name;

###############################################################
### EXAMPLES OF PARTICULAR ORTHOGONAL BASIS
###############################################################
class JacobiBasis(OrthogonalBasis):
    r'''
        Class for the Jacobi Basis with parameters $\alpa$ and $beta$.
        
        This class represents the OrthgonalBasis formed by the set of Jacobi polynomials
        with some fixed parameters $\alpha, \beta$, which are a class of orthogonal polynomials 
        with weight function $(1-x)^\alpha (1+x)^\beta$.
        
        Following the notation in https://arxiv.org/abs/1804.02964v1, we can find that 
        this basis has compatibilities with the multiplication by 'x'.
        
        INPUT::
            - ``alpha``: a rational number greater than -1
            - ``beta``: a rational number greater than -1
            - ``X``: the name for the operator representing the multiplication by $x$.
    '''
    def __init__(self, alpha, beta, X='x'):
        if(not alpha in QQ or alpha <= -1):
            raise TypeError("The argument `alpha` must be a rational number greater than -1");
        self.__alpha = QQ(alpha); alpha = self.__alpha;
        if(not beta in QQ or beta <= -1):
            raise TypeError("The argument `beta` must be a rational number greater than -1");
        self.__beta = QQ(beta); beta = self.__beta;
        n = self.n;

        self.__special_zero = (alpha + beta == 0) or (alpha + beta == -1);

        an = (2*n + alpha + beta + 1)*(2*n + alpha + beta + 2)/2/(n + 1)/(n + alpha + beta + 1);
        bn = (alpha**2 - beta**2)*(2*n + alpha + beta + 1)/2/(n + 1)/(n + alpha + beta + 1)/(2*n + alpha + beta);
        cn = (n + alpha)*(n + beta)*(2*n + alpha + beta + 2)/(n + 1)/(n + alpha + beta + 1)/(2*n + alpha + beta);

        super(JacobiBasis, self).__init__(an,bn,cn,X);

    @cached_method
    def get_element(self, n, var_name=None):
        if(self.__special_zero and n == 1):
            if(var_name is None):
                name = self.__var_name;
            else:
                name = var_name;
            R = self.polynomial_ring(name);    x = R.gens()[0];
            a0 = (self.__alpha + self.__beta)/2 + 1; b0 = (self.__alpha - self.__beta)/2;
            return a0*x + b0;

        return super(JacobiBasis, self).get_element(n,var_name);

    def __repr__(self):
        return "Jacobi (%s,%s)-Basis (%s, %s, %s,...)" %(self.__alpha, self.__beta,self.get_element(0), self.get_element(1), self.get_element(2));
    
    def _latex_(self):
        return r"\left\{P_n^{(%s,%s)}(%s)\right\}_{n \geq 0}" %(self.__alpha, self.__beta,self.var_name());

class GegenbauerBasis(OrthogonalBasis):
    r'''
        Class for the Gegenbauer Basis with parameter $\lambda$.
        
        This class represents the OrthgonalBasis formed by the set of Gegenbauer polynomials
        with some fixed parameter $\lambda$, which are a class of orthogonal polynomials 
        with weight function $(1-x^2)^{\lambda - 1/2}$.

        Gegenbauer polynomials are (up to scale) a special case of Jacobi polynomials
        with parameters $\alpha = \beta = \lambda - 1/2$.
        
        Following the notation in https://arxiv.org/abs/1804.02964v1, we can find that 
        this basis has compatibilities with the multiplication by 'x'.
        
        INPUT::
            - ``lambd``: a rational number greater than -1/2 different from 0
            - ``X``: the name for the operator representing the multiplication by $x$.
    '''
    def __init__(self, lambd, X='x'):
        if(not lambd in QQ or lambd <= -1/2 or lambd == 0):
            raise TypeError("The argument `alpha` must be a rational number greater than -1/2 different from 0");
        self.__lambda = QQ(lambd); lambd = self.__lambda;
        n = self.n;

        an = 2*(n+lambd)/(n+1);
        cn = (n+2*lambd-1)/(n+1);

        super(GegenbauerBasis, self).__init__(an,0,cn,X);

    def __repr__(self):
        return "Gegenbauer (%s)-Basis (%s, %s, %s,...)" %(self.__lambda, self.get_element(0), self.get_element(1), self.get_element(2));
    
    def _latex_(self):
        return r"\left\{C_n^{(%s)}(%s)\right\}_{n \geq 0}" %(self.__lambda, self.var_name());

class LegendreBasis(JacobiBasis):
    r'''
        Class for the Legendre Basis.
        
        This class represents the OrthgonalBasis formed by the set of Legendre polynomials
        which are a class of orthogonal polynomials with weight function $1$.

        Legendre polynomials are a special case of Jacobi polynomials
        with parameters $\alpha = \beta = 0$.
        
        Following the notation in https://arxiv.org/abs/1804.02964v1, we can find that 
        this basis has compatibilities with the multiplication by 'x'.
        
        INPUT::
            - ``X``: the name for the operator representing the multiplication by $x$.
    '''
    def __init__(self, X='x'):
        super(LegendreBasis, self).__init__(0,0,X);

    def __repr__(self):
        return "Legendre Basis (%s, %s, %s,...)" %(self.get_element(0), self.get_element(1), self.get_element(2));
    
    def _latex_(self):
        return r"\left\{P_n(%s)\right\}_{n \geq 0}" %(self.var_name());

class HermiteBasis(OrthogonalBasis):
    r'''
        Class for the Hermite Basis $(1,2x,4x^2-2,\dots)$.
        
        This class represents the OrthgonalBasis formed by the set of Hermite polynomials
        which are a class of orthogonal polynomials with weight function $e^{-x^2}$.
        
        Following the notation in https://arxiv.org/abs/1804.02964v1, we can find that 
        this basis has compatibilities with the multiplication by 'x' and with the derivation
        by 'x'.
        
        INPUT::
            - ``X``: the name for the operator representing the multiplication by $x$.
            - ``Dx``: the name for the operator representing the derivation by $x$.
    '''
    def __init__(self, X='x', Dx='Dx'):
        super(HermiteBasis, self).__init__(2,0,2*self.n,X);
        
        n = self.n; Sn = self.Sn;
        self.set_compatibility(Dx, 2*(n+1)*Sn);
        
    def __repr__(self):
        return "Hermite Basis (%s, %s, %s,...)" %(self.get_element(0), self.get_element(1), self.get_element(2));
    
    def _latex_(self):
        return r"\left\{H_n(%s)\right\}_{n \geq 0}" %self.var_name();

class HermitePBasis(OrthogonalBasis):
    r'''
        Class for the Probabilistic Hermite Basis $(1,x,x^2-1,\dots)$.
        
        This class represents the OrthgonalBasis formed by the set of probabilsitic Hermite polynomials
        which are a class of orthogonal polynomials with weight function $e^{-x^2/2}$.
        
        Following the notation in https://arxiv.org/abs/1804.02964v1, we can find that 
        this basis has compatibilities with the multiplication by 'x' and with the derivation
        by 'x'.
        
        INPUT::
            - ``X``: the name for the operator representing the multiplication by $x$.
            - ``Dx``: the name for the operator representing the derivation by $x$.
    '''
    def __init__(self, X='x', Dx='Dx'):
        super(HermitePBasis, self).__init__(1,0,self.n,X);
        
        n = self.n; Sn = self.Sn;
        self.set_compatibility(Dx, (n+1)*Sn);
        
    def __repr__(self):
        return "Probabilistic Hermite Basis (%s, %s, %s,...)" %(self.get_element(0), self.get_element(1), self.get_element(2));
    
    def _latex_(self):
        return r"\left\{He_n(%s)\right\}_{n \geq 0}" %self.var_name();
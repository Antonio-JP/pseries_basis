r'''
    Auxiliary file for extra utility methods
'''

from sage.all import ZZ, Matrix, vector

from psbasis.factorial_basis import BinomialBasis
from psbasis.product_basis import ProductBasis

def DefiniteSumSolutions(operator, *input):
    r'''
        Petkovsek's algorithm for transforming operators into recurrence equations.
        
        This method is the complete execution for the algorithm **DefiniteSumSolutions** described in
        :arxiv:`1804.02964v1`. This methods takes an operator `L` and convert the problem
        of being solution `L \cdot y(x) = 0` to a recurrence equation assuming some hypergeometric
        terms in the expansion.
        
        The operator must be a difference operator of `\mathbb{Q}[x]<E>` where `E: x \mapsto x+1`. The operator may 
        belong to a different ring from the package *ore_algebra*, but the generator must have the 
        behavior previously described.
        
        This function does not check the nature of the generator, so using this algorithm with different 
        types of operators may lead to some inconsistent results.
        
        INPUT:

        * ``operator``: difference operator to be transformed.
        * ``input``: the coefficients of the binomial coefficients we assume appear in the expansion
          of the solutions. This input can be given with the following formats:
            - ``a_1,a_2,...,a_m,b_1,b_2,...,b_m``: an unrolled list of `2m` elements.
            - ``[a_1,a_2,...,a_m,b_1,b_2,...,b_m]``: a compress list of `2m` elements.
            - ``[a_1,...,a_m],[b_1,...,b_m]``: two lists of `m` elements.

        OUTPUT:

        An operator `\tilde{L}` such that if a sequence `(c_n)_n` satisfies `L \cdot (c_n)_n = 0` then 
        the power series

        .. MATH::

            y(x) = \sum_{n \geq 0}\prod{i=1}^m c_n\binom{a_i*x+b_i}{n}

        satisfies `L \cdot y(x) = 0`.

        EXAMPLES::

            sage: from psbasis import *
            sage: R.<x> = QQ[]; OE.<E> = OreAlgebra(R, ('E', lambda p : p(x=x+1), lambda p : 0))
            sage: DefiniteSumSolutions((x+1)*E - 2*(2*x+1), 1,1,0,0)
            Sn - 1
            sage: example_2 = 4*(2*x+3)^2*(4*x+3)*E^2 - 2*(4*x+5)*(20*x^2+50*x+27)*E + 9*(4*x+7)*(x+1)^2
            sage: DefiniteSumSolutions(example_2, 1,1,0,0)
            (n + 1/2)*Sn - 1/4*n - 1/4
    '''
    ## Checking the input
    if(len(input) == 1 and type(input) in (tuple, list)):
        input = input[0]

    if(len(input)%2 != 0):
        raise TypeError("The input must be a even number of elements")
    elif(len(input) !=  2 or any(type(el) not in (list,tuple) for el in input)):
        m = len(input)//2
        a = input[:m]; b = input[m:]
    else:
        a,b = input; m = len(a)
    
    if(len(a) != len(b)):
        raise TypeError("The length of the two arguments must be exactly the same")
        
    if(any(el not in ZZ or el <= 0 for el in a)):
        raise ValueError("The values for `a` must be all positive integers")
    if(any(el not in ZZ for el in b)):
        raise ValueError("The values for `a` must be all integers")
        
    ## Getting the name of the difference operator
    E = str(operator.parent().gens()[0])
    
    if(m == 1): # Non-product case
        return BinomialBasis(a[0],b[0],E=E).recurrence(operator)
    
    ## Building the appropriate ProductBasis
    B = ProductBasis([BinomialBasis(a[i],b[i],E=E) for i in range(m)], E=E)
    
    ## Getting the compatibility matrix R(operator)
    compatibility = B.recurrence(operator)
    
    ## Cleaning the appearance of Sni
    column = [compatibility.coefficient((j,0)) for j in range(m)]
    deg = max(el.degree(B.Sni()) for el in column)
    column = [B.OSS()(B.reduce_SnSni(B.Sn()**deg * el)) for el in column]
    
    ## Extracting the gcrd for the first column
    result = column[0].gcrd(*column[1:])
    
    return result

def unroll_sequence(operator, init, bound=10):
    r'''
        This method takes an operator of the shape

        .. MATH::

        p_d(n)S_n^d + \ldots p_1(n)S_n + p_0(n)

        and some initial values given by ``init`` and unrolls
        the sequence defined by that operator equal to zero
        and the initial values up to some bound.
    '''
    d = operator.order()
    if(d == 0):
        if(all(el == 0 in init for el in init)):
            return bound*[0]
        raise ValueError("The initial values are not valid for the solution")

    p = operator.coefficients()
    solution = init[:d]
    ## Cheking the initial values
    if(len(solution) < d):
        raise ValueError("Not enough initial data was given")

    for k in range(d, bound):
        if(p[-1](n=k) == 0): # can not compute the value (we need it with ``init``)
            if(k < len(init)):
                solution += [init[k]]
            raise ValueError("The initial data did not provided enough information (zero in leading coefficient at n=%d)" %k)
        solution += [-sum(p[i](n=k)*solution[-d+i] for i in range(d))/p[-1](n=k)]
        if(k < len(init)): # we check the value is correct
            if(solution[-1] != init[k]):
                raise ValueError("The initial values are not valid for the solution (too many given and do not fit at index %d)" %k)
    
    return solution

def compute_in_basis(basis, sequence, section=1, shift=0, bound=10):
    r'''
        This method takes a :class:`PSBasis` and a sequence and computes
        the expansion of that sequence in the usual power basis up to 
        a bound.

        If the sequence is given as a list, the bound may not be reached if not enough
        data is provide.

        We take only the elements in the basis that are in the `j`-th `F`-section
        of the basis, i.e., elements of the shape `b_{kF+j}`. By default, the parameters
        ``section`` and ``shift`` are prepared to consider the whole basis.
    '''
    ## Cheking the section/shift arguments
    if(not (section in ZZ and section > 0)):
        raise ValueError("The section numbers must be a positive integer")
    if(not (shift in ZZ and shift >= 0)):
        raise ValueError("The shift must be a non-negative integer")
    if(shift >= section):
        raise ValueError("The shift must be smaller than the number of sections")

    ## Checking the type of sequence
    if(not (type(sequence) in (list, tuple))):
        sequence = [sequence(i) for i in range(bound)]

    ## Computing the iterative expansions
    b = min(len(sequence), bound)
    solution = [sequence[0]*basis[shift]]
    for i in range(1,b):
        solution += [solution[-1]+sequence[i]*basis[i*section+shift]]

    return solution

def unroll_in_basis(basis, operator, init, section=1, shift=0, bound=10):
    r'''
        This method takes a :class:`PSBasis` and a sequence in form of P-finite
        recurrence (with initial data) and compute the expansion in the usual power 
        basis up to a bound.

        This method is equivalent to :func:`compute_in_basis`, but the sequence is 
        not in list format or in a function but in a P-finite representation, that
        can be unrolled using the method :func:`unroll_sequence`.

        We take only the elements in the basis that are in the `j`-th `F`-section
        of the basis, i.e., elements of the shape `b_{kF+j}`. By default, the parameters
        ``section`` and ``shift`` are prepared to consider the whole basis.
    '''
    return compute_in_basis(basis, unroll_sequence(operator, init, bound), section, shift, bound)

def multiset_inclusion(l1, l2):
    r'''
        Method to check that the list ``l1`` is completely included in ``l2``.

        This method checks the multiset inclusion between two lists. This means that 
        all the elements appear in the second list at least the same amount of times as 
        in the first list.

        INPUT:

        * ``l1``: the multiset to be checked
        * ``l2``: the multiset where we will check

        OUTPUT:

        ``True`` if all elements in ``l1`` can be found in ``l2``, ``False`` otherwise.
    '''
    return all(l1.count(el) <= l2.count(el) for el in set(l1))

def guess_compatibility_E(basis, sections = None, bound_roots = 50, bound_data=50):
    ## Checking the condition on the roots
    rho = basis.root_sequence()
    r0 = rho(0); A = 0

    while(r0+1 != rho(A)): A += 1

    if(not all(multiset_inclusion([rho(i)+1 for i in range(k)], [rho(i) for i in range(k+A)]) for k in range(bound_roots))):
        raise TypeError("The roots condition does not hold up to %d" %bound_roots)

    x = basis[1].parent().gens()[0]
    actual_data_bound = bound_data+bound_roots
    M = Matrix([list(basis[i](x=x+1))+(actual_data_bound-i-1)*[0] for i in range(actual_data_bound)])*basis.basis_matrix(actual_data_bound).inverse()
    # the rows of M have all the coordinates of basis[i](x+1) in term of basis itself
    if(any(any(el != 0 for el in M[i][:i-A]) for i in range(A,M.nrows()))):
        raise ValueError("The guessed bound is incorrect: a non-zero coefficient found")

    F = None
    if(isinstance(basis, ProductBasis)):
        if(sections != None and sections%basis.nfactors() != 0):
            raise ValueError("The argument sections must be a multiple of the number of factors")
        elif(sections != None):
            F = sections
        else:
            F = basis.nfactors()
    elif(sections != None):
        F = sections

    if(F != None): # we work by sections
        data = [[[M[i+r][i+r-j] for i in range(j,M.nrows()-r,F)] for j in range(A+1)] for r in range(F)]
        functions = [[guess_rational_function(data[i][j], basis.OSS())(n=basis.n()/F) for j in range(len(data[i]))] for i in range(len(data))]

        return (A, 0, lambda i,j,k : functions[i][-j](n=k))
    else:
        data = [[M[i][i-j] for i in range(j,M.nrows())] for j in range(A+1)]
        # we guess a recurrence from the data
        functions = [guess_rational_function(data[i], basis.OSS()) for i in range(len(data))]

        return (A, 0, functions[::-1])
        
def guess_rational_function(data, algebra):
    # special case all zero
    if(all(el == 0 for el in data)):
        return algebra.base().zero()
    from ore_algebra import guess
    ## Getting the recurrence (error if not found)
    ann = guess(data, algebra)
    ## Computing rational solutions
    solutions = ann.rational_solutions(); nsols = len(solutions)

    ## Adjusting the solutions
    if(nsols == 0):
        raise ValueError("No rational solution found: no solution for a recurrence")

    values = Matrix([[solutions[i][0](n=j) for i in range(nsols)] for j in range(len(data))])
    try:
        sol = values.solve_right(vector(data))
    except ValueError:
        raise ValueError("No rational solution found: the data does not match")

    return sum(sol[i]*solutions[i][0] for i in range(nsols))

def check_compatibility(basis, operator, action, bound=100):
    a,b,m,alpha = basis.compatibility(operator)
    return all(
        all(
            sum(basis[k*m+r+i]*alpha(r,i,k) for i in range(-a,b+1)) == action(basis[k*m+r]) 
            for r in range(m)) 
        for k in range(m//a, bound))

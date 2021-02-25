r'''
    Auxiliary file for extra utility methods
'''

from sage.all import ZZ

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
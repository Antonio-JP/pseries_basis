from sage.all import PolynomialRing, FractionField, QQ

## Basic variables for replicating the reductions
R = PolynomialRing(QQ, ['a','b','n'])
F = FractionField(R)
RR = PolynomialRing(F, ['x'])

a,b,n = R.gens(); x = RR.gens()[0]

## Auxiliary methods to manage the dictionaries to handle the reductions
def apply_op_dict(op, default=0, *dicts):
    if(len(dicts) < 1 or any((not isinstance(el, dict)) for el in dicts)):
        raise TypeError("Invalid arguments")
    key_set = set(dicts[0].keys())
    for i in range(1,len(dicts)):
        key_set = key_set.union(dicts[i].keys())
    return {key:op(*[el.get(key,default) for el in dicts]) for key in key_set}

def ssum(*summand):
    return sum(summand)

def add_dicts(*dicts):
    return apply_op_dict(ssum, 0, *dicts)

def sub_dicts(d1, d2):
    return add_dicts(d1, apply_op_dict(lambda p: -p, 0, d2))

def scalar_dict(d1, factor):
    return apply_op_dict(lambda p : p*factor, 0, d1)

def get_shift(dict):
    return {k : v(n=2*n-k).factor() for (k,v) in dict.items()}

## Methods to reduce Jacobi polynomials
def jacobi_reduce_beta(n):
    return {n : 2*(n+b+1)/(2*n+a+b+2), n+1 : 2*(n+1)/(2*n+a+b+2)}, (1+x)

def jacobi_reduce_alpha(n):
    db,_ = jacobi_reduce_beta(n)
    da = {n : 2}
    return sub_dicts(da, db), (1-x)

def jacobi_reduce_derivation(n):
    da, _ = jacobi_reduce_alpha(n-1)
    d_ab = add_dicts(*[scalar_dict(jacobi_reduce_beta(k)[0], v) for (k,v) in da.items()])

    dd = scalar_dict(d_ab, ((n+a+b+1)/2))
    return dd, (1-x**2)

## Methods to reduce Gegenbauer polynomials
def gegenbauer_reduce_lambda(n):
    return {n: (n+2*a)*(n+2*a+1)/(4*a*(n+a+1)), n+2: -(n+1)*(n+2)/(4*a*(n+a+1))}, (1-x**2)

def gegenbauer_reduce_derivation(n):
    da, fa = gegenbauer_reduce_lambda(n-1)

    dd = scalar_dict(da, 2*a)
    return dd, fa
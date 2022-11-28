r'''
    Module containing methods, structures and functions related with `q`-Sequences.

    A `q`-sequence is a function `f: \mathbb{N} \rightarrow \mathbb{K}(q)` where the space `\mathbb{K}(q)` is 
    called the *universe* of the sequence. The difference between a sequence and a `q`-sequence is that the 
    appearance of the argument `n` in a `q`-sequence is not arbitrary, but always appearing in the form `q^n`.
'''
from .ore import has_variable
from .sequences import Sequence, LambdaSequence, ExpressionSequence, RationalSequence

class QSequence(Sequence):
    r'''
        Class to represent a `q`-sequence. 

        This class transforms the universe of the sequence to incorporate a `q` variable in it and 
        will be interpreted always as sequences in `q^n`.
    '''
    def __init__(self, universe=None, q_name="q", dim = 1, allow_sym=True, **kwds):
        if universe is None:
            raise TypeError("The universe of a sequence must never be None")
        has_q, q = has_variable(universe, q_name)
        if not has_q:
            raise TypeError(f"The universe of a `{q_name}`-sequence must have a variable called {q_name}")

        self.__q = q
        super().__init__(
            universe=universe, dim=dim, allow_sym=allow_sym, # arguments for Sequence
            **kwds # other arguments (allowing multi-inheritance)
        )

    @property
    def q(self):
        return self.__q

    def element(self, *indices: int):
        return super().element(*[self.q**i for i in indices])

class QLambdaSequence(QSequence, LambdaSequence):
    r'''
        Adaptation of :class:`.sequences.LambdaSequence` to the `q`-sequences setting
    '''
    def __init__(self, func=None, universe=None, q_name="q", dim = 1, allow_sym = False, **kwds):
        if func is None:
            raise TypeError("The universe of a sequence must never be None")
        super().__init__(
            universe=universe, q_name=q_name, dim=dim, allow_sym=allow_sym, # arguments for QSequence
            func=func, **kwds # arguments for other builders (like LambdaSequence) allowing multi-inheritance
        )

class QExpressionSequence(QSequence, ExpressionSequence):
    r'''
        Adaptation of :class:`.sequences.ExpressionSequence` to the `q`-sequences setting
    '''
    def __init__(self, expr=None, universe=None, variables=None, q_name="q", **kwds):
        super().__init__(
            universe=universe, q_name=q_name, dim=None,allow_sym=True, # arguments for QSequence
            expr=expr, variables=variables, **kwds # arguments for other builders (like ExpressionSequence) allowing multi-inheritance    
        )

class QRationalSequence(QSequence, RationalSequence):
    r'''
        Adaptation of :class:`.sequences.ExpressionSequence` to the `q`-sequences setting
    '''
    def __init__(self, func=None, ring=None, variables=None, q_name="q", **kwds):
        func, ring, variables, universe = RationalSequence._init_arguments(func, ring, variables)
        if q_name in [str(v) for v in variables]:
            raise ValueError(f"The q-variable ({q_name}) can not be part of the variables for a Rational sequence")

        # removing possible repeated arguments
        kwds.pop("universe",None); kwds.pop("dim", None); kwds.pop("allow_sym", None)
        super().__init__(
            universe=universe, q_name=q_name, dim=len(variables), allow_sym=True, # arguments for QSequence builder
            func=func, ring=ring, variables=variables, **kwds # arguments for other builders (like RationalSequence) to allow multi-inheritance
        ) 

__all__ = ["QSequence", "QLambdaSequence", "QExpressionSequence", "QRationalSequence"]
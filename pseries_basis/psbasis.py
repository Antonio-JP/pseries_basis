r'''
    Sage package for Power Series Basis.

    This module introduces the basic structures in Sage for computing with *Power
    Series Basis*. We based this work in the paper :doi:`10.1016/j.jsc.2022.11.002`
    by A. Jiménez-Pastor and M. Petkovšek, where all definitions and proofs for the algorithms 
    can be found.

    In particular, this module allows to consider basis of formal power series subrings, i.e., 
    sets of formal power series `f_n(x) \in \mathbb{K}[[x]]` that are `\mathbb{K}`-linearly 
    independent.

    There are two easy types of basis:

    * Polynomial basis: here `f_n \in \mathbb{K}[x]` with degree equal to `n`.
    * Order basis: here `ord(f_n) = n`, meaning that `f_n = x^n g_n(x)` such that `g(0) \neq 0`.

    But this module will allow any linearly independent set of functions. Then, we will assume
    that a formal power series `g(x)` is expressible in terms of the basis `f_n`:

    .. MATH::

        g(x) = \sum_{n \in \mathbb{N}} \alpha_n f_n.

    The main aim of this work is to understand which `\mathbb{K}`-linear operators over the
    ring of formal power series are *compatible* with a power series basis, meaning that, 
    `L\cdot g(x) = 0` if and only if the sequence `\alpha_n` is P-recursive.

    It is important to remark that this can be analogously develop for basis of sequences
    by using the canonical bijection between formal power series and sequences:

    .. MATH::

        \sum_{n} \alpha_n x^n \leftrightarrow (\alpha_n)_n

    For any given basis, we say that a `\mathbb{K}`-linear operator `L` is `(A,B)`-*compatible*
    in `t` sections if, for all `n \in \mathbb{N}` and `m \in \{0,\ldots,t-1\}` we can write:

    .. MATH

    L f_{nt+m} = \sum_{i=-A}^B c_{m,i,n} f_{nt+m+i},

    where `(c_{m,i,n})_n` are **valid** sequences, where **valid** means they are hypergeometric 
    in some sense (either `q`-hypergeometric or normal hypergeometric). Then, if we have
    `g(x)` an element in the spanned space by our basis with 

    .. MATH 

    g = \sum_k \alpha_k f_k, \qquad L \cdot g = 0,

    then the sequence `(\alpha_k)_k` satisfy a set of recurrences inherited from the compatibility
    equation of `L` for the basis.

    EXAMPLES::

        sage: from pseries_basis import *
        sage: from pseries_basis.sequences import *
        sage: B = PSBasis(lambda k : RationalSequence(QQ[x](x**k)), QQ) # B_k(x) = x^k
        sage: B[0]
        Sequence over [Rational Field]: (1, 1, 1,...)
        sage: B[1]
        Sequence over [Rational Field]: (0, 1, 2,...)
        sage: B[2]
        Sequence over [Rational Field]: (0, 1, 4,...)
        sage: B[10].generic()
        x^10

    We can from this point set up the compatibilities for this basis::

        sage: B.set_compatibility(
        ....:     'x', 
        ....:     Compatibility([[ConstantSequence(0,QQ,1), ConstantSequence(1,QQ,1)]], 0,1,1), 
        ....:     type="any"
        ....: ) # compatibility with x
        Compatibility condition (0, 1, 1) with following coefficient matrix:
        [0 1]
        sage: B.set_compatibility(
        ....:     'Dx', 
        ....:     Compatibility([[RationalSequence(QQ['n']('n')), ConstantSequence(0,QQ,1)]], 1,0,1), 
        ....:     type="derivation"
        ....: ) # compatibility with Dx
        Compatibility condition (1, 0, 1) with following coefficient matrix:
        [n 0]

    And then the compatibility conditions can be obtained for any expression involving objects called `x`
    and `Dx`::

        sage: B.is_compatible("x**2 * Dx**2 - Dx + x^3")
        True
        sage: B.compatibility_type("x**2 * Dx**2 - Dx + x^3")
        'derivation'
'''
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

from collections.abc import Callable
from functools import reduce
from ore_algebra.ore_algebra import OreAlgebra_generic
from ore_algebra.ore_operator import OreOperator

from sage.algebras.free_algebra import FreeAlgebra, is_FreeAlgebra
from sage.arith.functions import lcm
from sage.categories.pushout import pushout
from sage.matrix.constructor import Matrix
from sage.misc.cachefunc import cached_method
from sage.misc.latex import latex
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.symbolic.ring import SR

from .sequences.base import ConstantSequence, Sequence, SequenceSet
from .sequences.element import ExpressionSequence, RationalSequence
from .sequences.qsequences import is_QSequence, QExpressionSequence, QRationalSequence

class PSBasis(Sequence):
    r'''
        Base class for a basis of formal power series/sequences.

        This class allows to represent any basis of formal power series of sequences over a 
        SageMath field `\mathbb{K}`. In general, a basis is a sequence `f_k` indexed by
        an natural number `k` of formal power series or sequences. Hence, this class
        inherits from the generic sequence class in :class:`~.sequences.base.Sequence`,
        and always provides a representation as a sequence of sequences.

        .. MATH::

            `\mathcal{B}: k \mapsto (\alpha_{k,n})_n`

        where `f_k(x) = \sum_n \alpha_{k,n}x^n`.

        This class will implement any other method for basis of formal power series related with
        the compatibility of a sequence.
    '''
    def __init__(self, sequence: Callable[[int], Sequence], universe=None, *, _extend_by_zero=False, **kwds):
        # We check the case we provide a 2-dimensional sequence
        if isinstance(sequence, Sequence) and sequence.dim == 2:
            universe = sequence.universe if universe is None else universe
            or_sequence = sequence

            sequence = lambda k : or_sequence.slicing((0,k))
            self.__original_sequence = or_sequence
        else:
            if universe is None:
                raise ValueError(f"When argument is callable we require a universe argument for the basis.")
            self.__original_sequence = None
        self.__inner_universe = universe

        ## Attributes for storing compatibilities
        self.__basic_compatibilities : dict[str, Compatibility] = dict()
        self.__homomorphisms : list[str] = list()
        self.__derivations : list[str] = list()
        self.__any : list[str] = list()

        ## Other cached variables
        self.__ore_algebra : OreAlgebra_generic = None
        self.__double_algebra : OreAlgebra_generic = None
        self.__quasi_triangular : Sequence = None

        ## Calling the super method
        super().__init__(sequence, SequenceSet(1, universe), 1, _extend_by_zero=_extend_by_zero, **kwds)

    ##########################################################################################################
    ###
    ### PROPERTY METHODS
    ### 
    ##########################################################################################################
    @property
    def base(self):
        return self.__inner_universe
    
    def change_base(self, base) -> PSBasis:
        r'''
            Method to compute the same basis of sequences as ``self`` with a different base ring.

            This method creates a new :class:`PSBasis` that changes the ring over which the elements
            are defined. This method does not guarantee the new basis makes real sense and it
            may raise some errors when creating the elements of the basis.

            **Overriding recommended**: this method acts as a default way of changing the base
            ring of a :class:`PSBasis` but it may lose some information in the process. In order
            to preserve the structure of the :class:`PSBasis`, this method should be overridden.

            INPUT:

            * ``base``: a new parent structure for the universe of the elements of ``self``

            OUTPUT:

            A equivalent basis where the elements have as common universe ``base``
        '''
        args, kwds = self.args_to_self()
        kwds["universe"] = base
        output = self.__class__(*args, **kwds)

        ## Recreating the "original_sequence" part
        if self.__original_sequence is not None:
            output._PSBasis__original_sequence = self.__original_sequence.change_universe(base)
        
        ## We guarantee the quasi-triangular property still remains
        output.is_quasi_triangular = lambda : self.is_quasi_triangular()

        ## Recreating the compatibilities
        for operator in self.basic_compatibilities():
            if operator not in output.basic_compatibilities():
                output.set_compatibility(operator, self.compatibility(operator).change_base(base), True, self.compatibility_type(operator))
        return output
    
    def _element(self, *indices: int):
        output = super()._element(*indices)
        return output.change_universe(self.base)
    
    @cached_method
    def as_2dim(self) -> Sequence:
        return (self.__original_sequence if self.__original_sequence is not None else 
                Sequence(lambda n,k : self(n)(k), self.base, 2, _extend_by_zero = self._Sequence__extend_by_zero, **self.extra_info()["extra_args"]))
    
    def generic(self, *names : str):
        if self.__original_sequence:
            try:
                return self.__original_sequence.generic(*names)
            except ValueError: 
                pass
        return super().generic(*names)
    
    ##########################################################################################################
    ###
    ### SEQUENCE METHODS
    ### 
    ##########################################################################################################
    ### Casting methods
    def args_to_self(self):
        return [self.as_2dim()], {"universe":self.base, "_extend_by_zero": self._Sequence__extend_by_zero, **self.extra_info()["extra_args"]}
        
    def _change_class(self, cls, **extra_info): # pylint: disable=unused-argument
        if cls != Sequence:
            raise NotImplementedError(f"Class {cls} not recognized as {self.__class__}")
        return Sequence(self._element, self.universe, 1, _extend_by_zero=self._Sequence__extend_by_zero, **self.extra_info()["extra_args"])
    
    @classmethod
    def _change_from_class(cls, sequence: Sequence, **extra_info): # pylint: disable=unused-argument
        if not isinstance(ConstantSequence):
            raise NotImplementedError(f"Class {sequence.__class__} not recognized from {cls}")
        return PSBasis(lambda _ : sequence, sequence.universe, **extra_info.get("extra_args"))
    
    ### Arithmetic methods
    def _neg_(self) -> PSBasis:
        r'''
            Addition inverse of a :class:`PSBasis`.

            The compatibilities are simple inverted as well, since:

            .. MATH::

                L (-P_k(n)) = - L (P_k(n))
        '''
        output = PSBasis(lambda *n : (-1)*self._element(*n), self.universe) 
        ## We extend compatibilities
        for operator in self.basic_compatibilities():
            compatibility = self.compatibility(operator)
            A,B,t = compatibility.data()
            ctype = self.compatibility_type(operator)
            output.set_compatibility(operator, 
                Compatibility([[-compatibility[b,i] for i in range(-A,B+1)] for b in range(t)], A, B, t),
                sub=True,
                type=ctype
            )
        return output
    
    def _final_add(self, other:PSBasis) -> PSBasis:
        r'''
            Addition of two :class:`PSBasis`
            
            **WARNING**: No compatibility is extended!
        '''
        return PSBasis(lambda n: self._element(n) + other._element(n), self.universe, **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]})
    def _final_sub(self, other:PSBasis) -> PSBasis:
        r'''
            Difference of two :class:`PSBasis`
            
            **WARNING**: No compatibility is extended!
        '''
        return PSBasis(lambda n: self._element(n) - other._element(n), self.universe, **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]})
    def _final_mul(self, other:PSBasis) -> PSBasis:
        r'''
            Hadamard product of two :class:`PSBasis`
            
            **WARNING**: No compatibility is extended!
        '''
        return PSBasis(lambda n: self._element(n)*other._element(n), self.universe, **{**self.extra_info()["extra_args"], **other.extra_info()["extra_args"]})
    def _final_div(self, _:PSBasis) -> PSBasis:
        r'''
            Quotient of two :class:`PSBasis`
            
            **WARNING**: This method is not implemented!
        '''
        return NotImplemented
    def _final_mod(self, _:Sequence) -> PSBasis:
        r'''
            Modulus between two :class:`PSBasis`
            
            **WARNING**: This method is not implemented!
        '''
        return NotImplemented
    def _final_floordiv(self, _:Sequence) -> PSBasis:
        r'''
            Exact division of two :class:`PSBasis`
            
            **WARNING**: This method is not implemented!
        '''
        return NotImplemented
    
    ### Other sequences operations
    def _shift(self, *shifts):
        shift = shifts[0] # we know the dimension is 1
        output = PSBasis(lambda n : self._element(n + shift), self.universe, **self.extra_info()["extra_args"])
        for operator in self.basic_compatibilities():
            compatibility = self.compatibility(operator)
            ctype = self.compatibility_type(operator)
            A,B,t = compatibility.data()
            output.set_compatibility(operator,
                Compatibility([[compatibility[b,i].shift(shift) for i in range(-A,B+1)] for b in range(t)], A,B,t),
                sub=True,
                type=ctype
            )
        return output
    ## Slicing not implemented because PSBasis have dimension 1.
    def _subsequence(self, final_input: dict[int, Sequence]):
        return PSBasis(lambda n : self._element(final_input[0]._element(n) if 0 in final_input else n), self.universe, **self.extra_info()["extra_args"])

    ### Creating new PSBasis by scaling its elements
    def scalar(self, factor: Sequence) -> PSBasis:
        r'''
            Method to scale a :class:`PSBasis` preserving compatibilities.

            This method computes a new :class:`PSBasis` structure and extends
            when possible the compatibility conditions over ``self``. The elements
            of the new sequence is the scaling by the sequence given in ``factor``.

            This method works on two steps:

            * First, we create a new :class:`PSBasis` with the corresponding elements.
              This may differ from different classes and can be extended in the method
              :func:`_scalar_basis`.
            * Second, we extend the compatibilities. Since some of the compatibilities
              can be automatically computed when creating the basis, we only extends those
              compatibilities that are not already created.

            This method exploits when possible the fact that the given factor is hypergeometric.
            This is based in the method :func:`.sequences.base.Sequence.is_hypergeometric`. If this 
            method succeeds, we use the rational function obtained to extend compatibilities.

            INPUT:

            * ``factor``: a :class:`.sequences.base.Sequence` with the scaling factor. 

            OUTPUT:

            A new basis with the extended compatibilities.

            **WARNING**: 

            This method assumes that the sequence given by ``factor`` never vanishes.

            TODO: add examples.
        '''
        if not isinstance(factor, Sequence):
            if factor in self.base:
                factor = ConstantSequence(factor, self.base, 1)
            elif factor in SR:
                factor = SR(factor)
                factor = ExpressionSequence(factor, factor.variables(), self.base)
            else:
                raise TypeError(f"The given factor must be like a sequence.")
            
        ## Creating the new basis object
        new_basis = self._scalar_basis(factor)

        ## Extending compatibilities
        for operator in self.basic_compatibilities():
            if operator not in new_basis.basic_compatibilities():
                new_basis.set_compatibility(
                    operator, 
                    self.compatibility(operator).scale_basis(factor), 
                    True, 
                    self.compatibility_type(operator)
                )

        return new_basis

    def _scalar_basis(self, factor: Sequence) -> PSBasis:
        r'''
            Method that creates a new basis scaled by a given factor.

            This method assume the factor sequences is of correct format. This method can be
            overridden for creating more specific types of basis.
        '''
        return PSBasis(lambda n : self._element(n)*factor(n), self.base, **self.extra_info()["extra_args"])
    
    ##########################################################################################################
    ###
    ### INFORMATION OF THE BASIS
    ### 
    ##########################################################################################################
    @cached_method
    def is_quasi_triangular(self) -> Sequence:
        r'''
            Method to check whether a basis is quasi-triangular or not.

            We say that `\{B_k(n)\}_k` is quasi-triangular if there is a sequence `(i_n)_n` such that:

            * `B_{i_n}(n) \neq 0`.
            * `B_k(n) = 0` for all `k > i_n`.
            * `i_{n+1} > i_n`.

            If the sequences `B_k(n)` define formal power series `f_k(x) = \sum_{n>0} B_k(n)x^n`, then
            this is equivalent to `x^{i_n} | f_k(n)` if `k > i_n` and `x^{i_n} \nmid f_{i_n}(x)`.

            This method returns ``None`` if we do not know if the basis is quasi-triangular and the sequence
            `i_n`.
        '''
        return self.__quasi_triangular
    
    def inner_init_values(self, sum_sequence: Sequence, desired_values: int, inner_recurrence: OreOperator = None, full: bool = False, section: int = 1, shift: int = 0):
        r'''
            Method to obtain the initial values of an inner sequence using this basis.

            Let `(a_n)_n` be a sequence, and assume that we know that

            .. MATH::

                a_n = \sum_{k\geq 0} c_k B_k(n).

            We say that `(c_k)_k` is the *inner sequence of `(a_n)_n` wrt `B_k(n)`*. This method takes the sequence
            `(a_n)_n`, the number of elements of `c_k` we want to compute and (optionally) a recurrence for the 
            sequence `c_k`. 

            If this recurrence for `(c_k)_k` is not provided, we can only compute this for triangular sequences (i.e., 
            the quasi-triangular sequence given by :func:`is_quasi_triangular` is the identity sequence).

            Otherwise, we can obtain an unrolling matrix `\Lambda_T = (\alpha_{i,k}) \in \mathbb{K}^{r\times T}` such that
            `(c_0,\ldots, c_{T-1}) = (c_0,\ldots, c_{r-1}) \Lambda_T` where `r` is the order of the ``inner_recurrence``. Then,
            we can compute for any `T` the product of matrices `\Lambda_T B_{T,N}` where the matrix `B_{T,N}` is the output
            of :func:`basis_matrix` over ``self`` for `T` rows and `N` columns.

            Moreover, if `T \geq i_N`, then

            .. MATH::

            (a_0,\ldots, a_{N-1}) = (c_0,\ldots, c_{r-1}) \left(\Lambda_T B_{T,N}\right).

            If the matrix `\left(\Lambda_T B_{T,N}\right)` has rank `r`, then there is a pseudo-inverse matrix `A^+` such that

            .. MATH::

            (c_0, \ldots, c_{r-1}) = (a_0,\ldots, a_{N-1}) A^+.

            Then, from this position, we can compute the required terms of `(c_k)_k`. Otherwise, for each `N`, we can compute the 
            `i_N` of the quasi-triangular sequence and obtain a space of possible solutions for `(c_0,\ldots,c_{r-1})`, denote it by `S_N`.
            If there is a solution, then `\bigcap_N S_N` has to stabilize. This intersection is still an affine space and any element
            in it will be a valid solution to this function.

            If this intersection does not stabilize, then we conclude there is no solution.

            INPUT:

            * ``sum_sequence``: the :class:`Sequence` for the `(a_n)_n`.
            * ``desired_values``: number of elements of the inner sequence we are aiming to compute.
            * ``inner_recurrence`` (optional): recurrence for the inner sequence.
        '''
        from sage.geometry.hyperplane_arrangement.affine_subspace import AffineSubspace
        from sage.modules.free_module import VectorSpace
        from sage.modules.free_module_element import vector
        from .misc.ore import unroll_matrix, is_qshift_algebra, is_recurrence_algebra

        I = self.is_quasi_triangular()
        if I is None:
            raise TypeError(f"The basis is not quasi-triangular: impossible to compute inner sequences.")
        elif inner_recurrence is None and I == Sequence(lambda n : n, ZZ) and section == 1:
            return basis_matrix(self, desired_values).solve_left(vector(sum_sequence[:desired_values]))
        elif inner_recurrence is None:
            raise TypeError(f"The basis is not triangular and no recurrence is given: impossible to compute inner sequences.")
        
        I = (I//section + 1) if section != 1 else I
        
        ## General case
        gen_seq = (lambda p: QRationalSequence(p, universe=self.base, q=self.q)) if is_QSequence(self) else (lambda p: RationalSequence(p, universe=self.base))
        if ((is_QSequence(self) and is_qshift_algebra(inner_recurrence.parent(), str(self.q))) or 
            is_recurrence_algebra(inner_recurrence.parent())
        ):
            r = inner_recurrence.order()
        F = inner_recurrence.parent().base().base_ring().fraction_field() # field to look the affine spaces
        S = AffineSubspace(r*[F.zero()], VectorSpace(F, r))

        cdimension = S.dimension()
        repeated = 0
        N = 1
        while (repeated < 2*r) and (not S is None):
            ## We take the value from I. We make it 1 if it takes value 0.
            II = I[N]
            II = 1 if II == 0 else II

            A = (unroll_matrix(inner_recurrence, gen_seq, II)*basis_matrix(self, II, N, section=section, shift=shift)).change_ring(F)
            nS = AffineSubspace(A.solve_left(vector(sum_sequence[:N])), A.left_kernel())
            S = S.intersection(nS)
            logger.debug(f"[inner_init_values] Computing the space for {N = } -> I(N) = {II}. Solution: {nS.dimension()}. Dimension: {cdimension} -> {S.dimension()}")

            if not S is None:
                if cdimension != S.dimension():
                    cdimension = S.dimension()
                    repeated = 0
                else:
                    repeated += 1
            N += 1
        
        if not S is None:
            if full:
                return S
            else:
                return S.point()*unroll_matrix(inner_recurrence, gen_seq, desired_values)
        else:
            raise ValueError("There is no solution to the given problem")

    # @property
    # TODO def functional_seq(self) -> Sequence
    # TODO def functional_matrix(self, nrows: int, ncols: int = None) -> matrix_class:
    # TODO def is_quasi_func_triangular(self) -> bool
    # TODO def functional_to_self(self, sequence: Sequence | Collection, size: int) -> matrix_class
    # @property
    # TODO def evaluation_seq(self) -> Sequence
    # TODO def evaluation_matrix(self, nrows: int, ncols: int = None) -> matrix_class:
    # TODO def is_quasi_eval_triangular(self) -> bool
    # TODO def evaluation_to_self(self, sequence: Sequence | Collection, size: int) -> matrix_class

    ##########################################################################################################
    ###
    ### COMPATIBILITY METHODS
    ### 
    ##########################################################################################################
    def _compatibility_from_recurrence(self, recurrence) -> Compatibility:
        # TODO def _compatibility_from_recurrence(self, recurrence: OreOperator) -> TypeCompatibility
        raise NotImplementedError("Method _compatibility_from_recurrence not implemented")
    
    def set_compatibility(self, name: str, compatibility: Compatibility, sub: bool = False, type: str = None) -> Compatibility:
        r'''
            Method to set a new compatibility with this :class:`PSBasis`.

            This method receives a name for the operator whose compatibility will be set and a 
            compatibility condition for this operator. This method does not check whether 
            this compatibility is real or not and this job is left to the user. 

            In general, this method is not *recommended* for users.

            The compatibility condition can be given in 3 different ways:

            * A :class:`Compatibility` object: it is directly stored
            * A tuple `(A,B,\alpha)` or `(A,B,m,\alpha)`: we create a :class:`Compatibility`
              using directly this data.
            * An ore operator in a ring with one shift where we can read the data
              to create a :class:`Compatibility` right from it.

            If the argument ``sub`` is ``True``, we substitute the compatibility, if not, 
            we raise a :class:`ValueError`. 

            INPUT:

            * ``name``: the name of the operator that we are defining as compatible.
            * ``compatibility``: the compatibility condition to be set.
            * ``sub``: if set to ``True``, we substitute the old compatibility with the new.
            * ``type``: a string (or None) describing the type of the operator. It can be ``"homomorphism"``
              or ``"derivation"``.

            OUTPUT:

            It will return the created compatibility condition in case of success or an error
            if something goes wrong.

            TODO: add examples
        '''
        if not sub and name in self.__basic_compatibilities:
            raise ValueError(f"The operator {name} was already compatible with this basis.")

        if name in self.__basic_compatibilities:
            del self.__basic_compatibilities[name]
            if name in self.__homomorphisms: 
                self.__homomorphisms.remove(name)
            elif name in self.__derivations: 
                self.__derivations.remove(name)
            elif name in self.__any: 
                self.__any.remove(name)

        if isinstance(compatibility, tuple):
            compatibility = Compatibility(
                compatibility[-1],                                 # \alpha
                compatibility[0],                                  # A
                compatibility[1],                                  # B
                1 if len(compatibility) == 3 else compatibility[2] # t
            )
        elif isinstance(compatibility, OreOperator):
            compatibility = self._compatibility_from_operator(compatibility)
        elif not isinstance(compatibility, Compatibility):
            raise TypeError(f"The given compatibility is not of valid type.")
        
        self.__basic_compatibilities[name] = compatibility
        if type == "homomorphism": 
            self.__homomorphisms.append(name)
        elif type == "derivation": 
            self.__derivations.append(name)
        elif type == "any": 
            self.__any.append(name)

        return compatibility
    
    def set_homomorphism(self, name: str, compatibility: Compatibility, sub: bool = False) -> Compatibility:
        r'''
            See :func:`set_compatibility`. This method adds a compatibility for an homomorphism
        '''
        return self.set_compatibility(name, compatibility, sub, "homomorphism")
    def set_derivation(self, name: str, compatibility: Compatibility, sub: bool = False) -> Compatibility:
        r'''
            See :func:`set_compatibility`. This method adds a compatibility for a derivation
        '''
        return self.set_compatibility(name, compatibility, sub, "derivation")
    
    def basic_compatibilities(self) -> list[str]:
        r'''
            Method that return a copy of the current basic compatibilities that exist for ``self``.

            See method :func:`compatibility` for further information.
        '''
        return list(self.__basic_compatibilities.keys())
    def compatible_endomorphisms(self) -> list[str]:
        r'''
            Method that return a copy of the current compatible homomorphisms that exist for ``self``.

            See method :func:`compatibility` for further information.
        '''
        return self.__homomorphisms.copy()
    def compatible_derivations(self) -> list[str]:
        r'''
            Method that return a copy of the current compatible derivations that exist for ``self``.

            See method :func:`compatibility` for further information.
        '''
        return self.__derivations.copy()
    
    def __get_algebra(self):
        if len(self.basic_compatibilities()) > 1:
            return FreeAlgebra(self.base, self.basic_compatibilities())
        else:
            return PolynomialRing(self.base, self.basic_compatibilities())
        
    def __cast_to_algebra(self, operator: str):
        from sage.misc.sage_eval import sage_eval
        A = self.__get_algebra()

        ## We now look for the "constants" by taking parents until the 1 is the generator of `C`
        C = A
        locals = dict()
        while 1 not in C.gens():
            locals.update(C.gens_dict())
            C = C.base()

        ## We evaluate now the operator
        value = sage_eval(operator, locals=locals)

        ## We cast the output
        return A(value)

    @cached_method
    def compatibility(self, operator) -> Compatibility:
        r'''
            Method that returns the compatibility of a given operator with ``self``.

            Check documentation of :class:`Compatibility` for further information on
            what a compatible operator is.

            INPUT:

            * ``operator``: the operator that we want to compute the compatibility. It
              can be any object that can be casted into a free algebra over the names
              that are currently compatible with ``self``.

            OUTPUT:

            A :class:`Compatibility` condition for the operator. It raises en error when 
            it is not possible to cast the operator to a compatible operator.
            
            INFORMATION: 
            
            This method is cached.

            TODO: add examples
        '''
        # Base case when the input is the name of an operator
        if isinstance(operator, str) and operator in self.__basic_compatibilities:
            return self.__basic_compatibilities[operator]
        
        FA = self.__get_algebra()
        operator = self.__cast_to_algebra(str(operator))
        ## We split evaluation in cases to avoid problems with FreeAlgebra implementations
        if is_FreeAlgebra(FA):
            ## Special evaluation since the method in FreeAlgebra does not work
            to_eval = [self.__basic_compatibilities[v] for v in FA.variable_names()]
            output = sum(
                (c*m(*to_eval) if m != 1 else Compatibility([[ConstantSequence(c, self.base, 1)]], 0,0,1) 
                for (m,c) in [(el.monomials()[0], el.coefficients()[0]) for el in operator.terms()]),
                Compatibility([[ConstantSequence(0, self.base, 1)]], 0,0,1)
            )
        else:
            comp = next(iter(self.__basic_compatibilities.values()))
            output = sum(c*comp**i for i,c in enumerate(operator.coefficients(False)))
        if output in self.base: # the operator is a constant
            return Compatibility(
                [[ConstantSequence(self.base(operator), self.base, 1, _extend_by_zero=False)]], # coefficient #pylint: disable=not-callable
                0,# A
                0,# B
                1 # t
            )
        return output # this is the compatibility  
    @cached_method
    def is_compatible(self, operator) -> bool:
        r'''
            Method that checks whether an object is compatible with a :class:`PSBasis`.

            This method tries to create the free algebra of compatible operators and cast
            the object ``operator`` to this ring. This is also done in the method 
            :func:`compatibility`. The main difference is that this method do **not**
            compute the actual compatibility. 

            INPUT:

            * ``operator``: an object to be checked for compatibility.

            INFORMATION:

            This method is cached.

            OUTPUT:

            A boolean value indicating if the method :func:`compatibility` will
            return something or an error.

            TODO: add examples
        '''
        if isinstance(operator, str) and operator in self.__basic_compatibilities:
            return True
                    
        try:
            self.__cast_to_algebra(str(operator))
            return True
        except Exception:
            return False
        
    @cached_method
    def compatibility_type(self, operator) -> None | str:
        r'''
            Method to determine if an operator belong to a specific type.

            Operators may have three different types:

            * "homomorphism": it behaves nicely with the product.
            * "derivation": it satisfies the Leibniz rule for the product.
            * "any": an operator that can be combined for any of the two previous.

            It may also have type "None", meaning we could not deduce any specific behavior.

            This methods behaves similar to the methods :func:`compatibility` and 
            :func:`is_compatible` but, contrary to method :func:`compatibility`,
            this method do **not** compute the actual compatibility.

            INPUT:

            * ``operator``: the object that we want to check.

            INFORMATION:

            This method is cached.

            OUTPUT:

            Either a string in ("homomorphism", "derivation", "any") or None.

            TODO: add examples
        '''
        if isinstance(operator, str) and operator in self.__basic_compatibilities:
            if operator in self.__homomorphisms: 
                return "homomorphism"
            elif operator in self.__derivations: 
                return "derivation"
            elif operator in self.__any: 
                return "any"
            else:
                return None
        
        operator = self.__cast_to_algebra(str(operator))
        basic_operations = [self.compatibility_type(str(v)) for v in operator.variables()]
        output = basic_operations[0]
        for basic in basic_operations[1:]:
            if output is None: 
                break
            elif basic == "homomorphism" and output in ("any", "homomorphism"): 
                output = "homomorphism"
            elif basic == "homomorphism": 
                output = None
            elif basic == "derivation" and output in ("any", "derivation"): 
                output = "derivation"
            elif basic == "derivation": 
                output = None
            elif basic is None: 
                output = None
            # The else case means that basic == "any", so output does not change
        return output

    ##########################################################################################################
    ###
    ### RECURRENCES METHODS
    ### 
    ##########################################################################################################
    def _create_algebra(self, double: bool):
        r'''
            Method to create uniformly in a class a (double) shift ore algebra when needed.
        '''
        if is_QSequence(self):
            from .misc.ore import get_qshift_algebra, get_double_qshift_algebra
            if double:
                return get_double_qshift_algebra("q_k", str(self.q), "Sk", base=self.base)
            else:
                return get_qshift_algebra("q_k", str(self.q), "Sk", base=self.base)
        else:
            from .misc.ore import get_recurrence_algebra, get_double_recurrence_algebra
            if double:
                return get_double_recurrence_algebra("k", "Sk", base=self.base)
            else:
                return get_recurrence_algebra("k", "Sk", base=self.base)

    def ore_algebra(self):
        r'''
            Method to get the ore algebra for recurrences

            This method builds the corresponding Ore Algebra that can will appear in the 
            recurrences that are derived from compatibility conditions.
        '''
        if self.__ore_algebra is None:
            self.__ore_algebra = self._create_algebra(False)
        return self.__ore_algebra[0]
    
    def ore_var(self):
        r'''Method that returns the variable affected by the shift in the shift algebra'''
        self.ore_algebra()
        return self.__ore_algebra[1][0]
    
    def ore_gen(self):
        r'''Method that returns the shift of the shift algebra'''
        self.ore_algebra()
        return self.__ore_algebra[1][1]
    
    def double_algebra(self):
        r'''
            Method to get the ore algebra for recurrences with inverses

            This method builds the corresponding Ore Algebra that can will appear in the 
            recurrences that are derived from compatibility conditions.
        '''
        if self.__double_algebra is None:
            # We create the corresponding ore algebra
            self.__double_algebra = self._create_algebra(True)
            
        return self.__double_algebra[0]
    
    def double_var(self):
        r'''Method that returns the variable affected by the shift in the double-shift algebra'''
        self.double_algebra()
        return self.__double_algebra[1][0]

    def double_gens(self):
        r'''Method that returns the shifts of the double-shift algebra'''
        self.double_algebra()
        return self.__double_algebra[1][1:]
    
    def _basic_recurrence(self, operator, sections : int = None):
        ## Get the compatibility condition from the operator
        if not isinstance(operator, Compatibility):
            operator = self.compatibility(operator)
        
        ## Putting appropriate number of sections
        if sections is not None and sections > 0 and sections % operator.t == 0:
            operator = operator.in_sections(sections)
        
        ## Computing the recurrence
        if operator.t == 1: # only one sequence
            recurrence = {-i: operator[0,i].shift(-i) for i in range(-operator.A,operator.B+1)} #pylint: disable=invalid-unary-operand-type
        else: # the output is a matrix of recurrences
            recurrence = []
            for r in range(operator.t):
                row = []
                for j in range(operator.t):
                    element = dict()
                    for i in range(-operator.A, operator.B+1): #pylint: disable=invalid-unary-operand-type
                        if (r-i-j)%operator.t == 0:
                            exp = (r-i-j)//operator.t
                            element[exp] = element.get(exp, 0) + operator[j,i].shift(exp)
                    row.append(element)
                recurrence.append(row)
        return recurrence

    def _process_recurrence(self, recurrence, output: str = None):
        if isinstance(recurrence, list): # matrix output
            logger.debug(f"[recurrence] Processing a matrix of recurrences")
            if output in ("ore", "ore_double"):
                logger.debug(f"[recurrence] Output is ore-related ({output=})")
                recurrence = [[self._process_recurrence(el, "ore_double") for el in row] for row in recurrence]
                if output == "ore": # we need to remove the inverse shift in an equal way through the matrix
                    S, Si = self.double_gens()
                    E = self.ore_gen()
                    D = max(max(el.polynomial().degree(Si.polynomial()) for el in row) for row in recurrence)
                    for row in recurrence:
                        for i in range(len(row)):
                            row[i] = sum(
                                (((S**D * c).polynomial().coefficients()[0]) * 
                                (E**(D + (m.degree(S.polynomial()) - m.degree(Si.polynomial())))))
                                for c,m in zip(row[i].polynomial().coefficients(), row[i].polynomial().monomials())    
                            )
                recurrence = Matrix(recurrence)
            else: # no need to build a matrix -> we process each input
                recurrence = [[self._process_recurrence(el, output) for el in row] for row in recurrence]
        else:
            logger.debug(f"[recurrence] Processing a single recurrence")
            if output in ("rational", "expression"):
                logger.debug(f"[recurrence] {output=}")
                for k,v in recurrence.items():
                    recurrence[k] = self._process_recurrence_sequence(v, output)
            elif output in ("ore_double", "ore"):
                logger.debug(f"[recurrence] Output is ore-related ({output=})")
                recurrence = self._process_ore_algebra(recurrence, output == "ore_double")
            elif output is not None:
                raise ValueError(f"Output type ({output}) not recognized")
        return recurrence
            
    def _process_recurrence_sequence(self, sequence, output):
        logger.log(logging.DEBUG-1, f"[recurrence @ process] Processing a sequence of type {sequence.__class__} into {output}")
        if isinstance(sequence, ExpressionSequence): # this includes both expression and rational
            logger.log(logging.DEBUG-1, f"[recurrence @ process] Variables: {sequence.variables()}; Expr: {sequence.generic()}")
        if is_QSequence(self):
            RatSeqCreation = lambda rational : QRationalSequence(rational, variables=[str(self.ore_var())], universe=sequence.universe, q=self.q)
            ExprSeqCreation = lambda expr : QExpressionSequence(expr, variables=[str(self.ore_var())], universe=sequence.universe, q=self.q)
        else:
            RatSeqCreation = lambda rational : RationalSequence(rational, variables=[str(self.ore_var())], universe=sequence.universe)
            ExprSeqCreation = lambda expr : ExpressionSequence(expr, variables=[str(self.ore_var())], universe=sequence.universe)

        ## Getting the generic with the "self.ore_var()" as value
        if isinstance(sequence, RationalSequence): # we evaluate the generic formula substituting the variable by the ore_var
            var = sequence.variables()[0]
            expr = sequence._eval_generic(**{str(var): self.ore_var()})
        else:
            expr = sequence.generic(str(self.ore_var())) # we try blindly

        return ExprSeqCreation(expr) if output == "expression" else RatSeqCreation(expr)

    def _process_ore_algebra(self, recurrence, double: bool = False):
        recurrence = self._process_recurrence(recurrence, "rational")
        if len(recurrence) == 0:
            return (self.double_algebra() if double else self.ore_algebra()).zero()
        
        if double:
            logger.debug(f"[recurrence] Processing a double-recurrence operator")
            OA = self.double_algebra()
            E, Ei = self.double_gens()
            return sum((OA(v.generic())*(E**i if i >= 0 else Ei**(-i)) for (i,v) in recurrence.items()), OA.zero())
        else:
            logger.debug(f"[recurrence] Processing a recurrence operator")
            OA = self.ore_algebra()
            E = self.ore_gen()
            neg_power = -min(0, min(recurrence.keys()))
            return sum((OA.base()(v.shift(neg_power).generic())*E**(i+neg_power) for (i,v) in recurrence.items()), OA.zero())

    def recurrence(self, operator, sections : int = None, output : str = "ore_double"):
        r'''
            Method to obtain a recurrence for a compatible operator.

            Following the theory in :doi:`10.1016/j.jsc.2022.11.002`, when we have an 
            operator `L` that is `(A,B,t)`-compatible with a basis os sequences such that

            .. MATH::

                L P_{kt+b} = \sum_{i=-A}^B c_{b,i}(k) P_{kt+b+i},

            then the solutions `L\cdot (\sum_k a_kP_k) = 0` can be obtained from solutions to
            `\tilde{L} \cdot a_k` where `\tilde{L}` can be automatically computed from 
            the compatibility and the coefficients `c_{b,i}(k)`.

            This method creates such recurrence (or system of recurrences) given the compatible 
            operator.

            INPUT:

            * ``operator``: an object that will be fed to method :func:`compatibility`.
            * ``sections``: indicate the number of sections that will be considered for the 
              compatibility condition. If has to be a multiple of the default number of 
              sections for the ``operator``.
            * ``output``: by default the output of this method is a Laurent Polynomial in 
              a shift operator with sequences as coefficients. This argument indicates 
              where we should transform this output to be used later. It allow two options:

              * "rational": force the sequence in the output to be rational sequences.
              * "expression": force the sequence in the output to be an expression sequence.
              * "ore_double": a Ore Algebra with two operators will be used as output.
                This requires the sequences are rational functions in the shift variable 
                and the two operators on the algebra will act as the forward and backward 
                shift.
              * "ore": a Ore Algebra with just a shift will be used as output. Similar to the
                double case, but we remove completely the inverse shift.
            
            OUTPUT:

            A recurrence or a matrix of recurrences as described in :doi:`10.1016/j.jsc.2022.11.002`.

            TODO: add examples.
        '''
        logger.debug(f"[recurrence] --- STARTING COMPUTATION OF A RECURRENCE")
        recurrence = self._basic_recurrence(operator, sections)
        logger.debug(f"[recurrence] --- FINISHED COMPUTATION OF A DICTIONARY REPRESENTATION")
        recurrence = self._process_recurrence(recurrence, output)
        logger.debug(f"[recurrence] --- FINISHED COMPUTATION OF THE FINAL RECURRENCE\n{recurrence=}")

        return recurrence

    ##########################################################################################################
    ###
    ### MAGIC METHODS
    ### 
    ##########################################################################################################
    def __repr__(self):
        output = f"Basis of Sequences over {self.base}"
        try:
            generic = self.as_2dim().generic("k","n")
            output += f": ({generic})"
        except ValueError:
            try:
                first_elements = [self(i).generic("n") for i in range(5)]
                output += ": (" + ", ".join(str(el) for el in first_elements) + ",...)"
            except ValueError:
                pass
        return output
    
    def _latex_(self):
        try:
            output = r"\left\{"
            generic = self.as_2dim().generic("k","n")
            output += latex(generic)
            output += r"\right\}_{k \in \mathbb{N}}"
        except ValueError:
            output = r"\left\{"
            try:
                first_elements = [self(i).generic("n") for i in range(5)]
                output += ", ".join(latex(el) for el in first_elements)
                output += ",..."
            except ValueError:
                output += r"B_k(n)"
            output += r"\right\}"

        return output
    
class Compatibility:
    r'''
        Class representing a compatibility condition in sections.

        A compatibility condition is associated to a basis of power series and with a linear operator
        that behaves *nicely* with the basis. More precisely, if we consider the basis `(f_k)_k` and 
        the linear operator `L`, we say that `L` is `(A,B)`-*compatible* in `t` sections with the basis if, for all 
        `a \in \mathbb{N}` and `b \in \{0,\ldots,t-1}`:

        .. MATH::

            L f_{at+b} = \sum_{i=-A}^B c_{b,i,a} f_{at+b+i},

        where, for fixed indices `b,i`, the element `(c_{b,i,a})_a` are *nice sequences*. In general, the niceness of the 
        sequences `(c_{b,i,a})` required depends on the operations we pretend to apply over the compatibility.

        INPUT:

        * ``c``: coefficients of the compatibility. It must be an object that should be accessed with syntax `c[b][i]`. 
          These objects must be sequences.
        * ``A``: lower bound of the compatibility.
        * ``B``: upper bound of the compatibility. 
        * ``t``: number of sections for the compatibility.

        The input can be given partially or totally. The coefficients `c` are mandatory and `A` or `B` is also required.
        The other bound and the value for `t` can be obtained from the structure of `c`, since:

        * ``len(c) == t``
        * ``len(c[*]) == A + B + 1``.

        If some of the optional arguments is given, it is used as sanity checks on the input of ``c``. If the user
        pretend to obtain the compatibility in more sections than given with the coefficients, consider the method
        :func:`in_sections` after the creation of the basic compatibility.

        TODO: add examples from the use of :class:`PSBasis`.
    '''
    def __init__(self, c, A: int = None, B: int = None, t: int = None):
        ## Processing number of sections
        if t is None:
            t = len(c)
        elif t != len(c):
            raise TypeError(f"[compatibility] Requested compatibility in {t} sections but only provided information for {len(c)} sections")
        if t == 0:
            raise ValueError(f"[compatibility] Compatibility in 0 sections is not properly defined.")

        ## Processing the number of elements
        if any(len(c[i]) != len(c[0]) for i in range(t)):
            raise TypeError(f"[compatibility] Incoherent information for compatibility: different size in each section.")
        if A is None and B is None:
            raise TypeError(f"[compatibility] At least `A` or `B` must be provided to a compatibility")
        elif (not A is None) and (not B is None):
            if (len(c[0]) != A+B+1):
                raise ValueError(f"[compatibility] Incoherent information for compatibility: given {len(c[0])} coefficients per section, but requested a ({A},{B}) compatibility")
        elif (not A is None):
            B = len(c[0]) - A - 1
        else:
            A = len(c[0]) - B - 1

        if A < 0 or B < 0:
            raise ValueError(f"[compatibility] Incorrect value for compatibility: given `A` or `B` too big.")
        
        if any(any(not isinstance(coeff, Sequence) or coeff.dim != 1 for coeff in section) for section in c):
            raise TypeError(f"[compatibility] coefficients mus be one-dimensional sequences")
        
        ## Computing a common universe for all sequences
        self.__base = reduce(lambda p,q : pushout(p,q), [coeff.universe for section in c for coeff in section], ZZ)

        ## Storing data for compatibility
        self.__lower = A
        self.__upper = B
        self.__nsections = t
        self.__data = [[coeff.change_universe(self.__base) for coeff in section] for section in c]
        self.__cache_pow = dict()

    @property
    def upper(self) -> int: return self.__upper #: Property of the upper bound for the compatibility
    @property
    def lower(self) -> int: return self.__lower #: Property of the lower bound for the compatibility
    @property
    def nsections(self) -> int: return self.__nsections #: Property of the number of sections for the compatibility
    A = lower #: alias for the upper bound
    B = upper #: alias for the lower bound
    t = nsections # alias for the number of sections

    def data(self) -> tuple[int,int,int]:
        r'''Return the compatibility data (`A`, `B`, `t`)'''
        return self.A, self.B, self.t
    
    def base(self):
        r'''Return the common universe of the sequences in the compatibility coefficients'''
        return self.__base
    def change_base(self, new_base) -> Compatibility:
        r'''Returns a new compatibility condition changing the universe of the sequences to a new ring'''
        return Compatibility([[coeff.change_universe(new_base) for coeff in section] for section in self.__data], self.A, self.B, self.t)
    
    def __getitem__(self, item):
        r'''Given `(b,i)` returns the compatibility on the `b`-th section on the `i`-th coefficients'''
        if item in ZZ: # only one element given
            item = (0 if self.t == 1 else item, item if self.t == 1 else None)
            if self.t == 1:
                item = (0, item)
            else:
                item = (item, None)
        
        if len(item) != 2:
            raise KeyError("Only tuples allowed for getting compatibility coefficients")
        elif item[0] < 0 or item[0] >= self.t:
            raise KeyError(f"Compatibility with {self.t} sections. Can not access section {item[0]}")
        elif item[0] is None or item[0] == slice(None,None,None): # requesting the full 
            return self.__data[item[0]]
        elif item[1] >= -self.A and item[1] <= self.B: #pylint: disable=invalid-unary-operand-type
            return self.__data[item[0]][item[1]+self.A]
        else:
            return ConstantSequence(0, self.base(), 1)
        
    def in_sections(self, new_sections: int) -> Compatibility:
        r'''
            Method to compute the compatibility condition with respect to more sections.
        '''
        if new_sections%self.t != 0:
            raise ValueError(f"[compatibility] Compatibilities can only be extended when the new sections ({new_sections}) are multiple of previous sections ({self.t}).")
        
        A, B = self.A, self.B
        a = new_sections // self.t # proportion to new sections

        coeffs = []
        for s in range(new_sections):
            new_section = []
            for i in range(-A, B+1): #pylint: disable=invalid-unary-operand-type
                s0,s1 = ZZ(s).quo_rem(self.t)
                new_section.append(self[s1,i].subsequence((0, (a, s0))))
            coeffs.append(new_section)
        
        return Compatibility(coeffs, A, B, new_sections)

    def add(self, other: Compatibility) -> Compatibility:
        r'''
            Method to compute the compatibility of the sum of the two compatibilities.

            One of the main results in :doi:`10.1016/j.jsc.2022.11.002` is that whenever
            a basis is compatible with two linear operators, then it is compatible with the sum of 
            the operators. 

            When we take into account the bounds of the compatibilities and the sections, the 
            final statement is the following: let `L_1` be a `(A_1,B_1)`-compatible operator
            in `t_1` sections and `L_2` be a `(A_2, B_2)`-compatible operator in `t_2` sections.
            Then `L_1 + L_2` is a `(A,B)`-compatible operator in `t` sections where

            * `A = \max(A_1,A_2)`,
            * `B = \min(B_1,B_2)`,
            * `t = \text{lcm}(t_1,t_2)`.

            This method return the corresponding compatibility for the addition of two operators
            from the compatibilities of these two operators.

            INPUT:

            * ``other``: a new compatibility condition that will *added* to ``self``.

            OUTPUT:

            A new :class:`Compatibility` for the operator obtained as the addition of the operators
            that defined ``self`` and ``other``.

            TODO: add examples
        '''
        if not isinstance(other, Compatibility):
            raise TypeError("[comp-add] Require a compatibility to compute the addition")
        elif self.base() != other.base():
            R = pushout(self.base(), other.base())
            return self.change_base(R).add(other.change_base(R))
        elif other.t != self.t:
            t = lcm(self.t, other.t)
            return self.in_sections(t).add(other.in_sections(t))

        ## Here we can assume that the base ring coincides and the number of sections is the same        
        ## Creating the elements for the compatibility
        A, B = max(self.A, other.A), max(self.B, other.B)
        coeffs = []
        for b in range(self.t):
            section = []
            for i in range(-A, B+1):
                section.append(self[b,i] + other[b,i])
            coeffs.append(section)
        
        return Compatibility(coeffs, A, B, self.t)
    
    def mul(self, other: Compatibility) -> Compatibility:
        r'''
            Method to compute the compatibility of the product/composition of the two compatibilities.

            One of the main results in :doi:`10.1016/j.jsc.2022.11.002` is that whenever
            a basis is compatible with two linear operators, then it is compatible with the product
            (i.e., the composition) of the operators. 

            When we take into account the bounds of the compatibilities and the sections, the 
            final statement is the following: let `L_1` be a `(A_1,B_1)`-compatible operator
            in `t_1` sections and `L_2` be a `(A_2, B_2)`-compatible operator in `t_2` sections.
            Then `L_1 + L_2` is a `(A,B)`-compatible operator in `t` sections where

            * `A = A_1+A_2`,
            * `B = B_1+B_2`,
            * `t = \text{lcm}(t_1,t_2)`.

            This method return the corresponding compatibility for the product of two operators
            from the compatibilities of these two operators.

            INPUT:

            * ``other``: a new compatibility condition that will *multiplied* to ``self``.

            OUTPUT:

            A new :class:`Compatibility` for the operator obtained as the product of the operators
            that defined ``self`` and ``other``.

            TODO: add examples
        '''
        if not isinstance(other, Compatibility):
            raise TypeError("[comp-mul] Require a compatibility to compute the product")
        elif self.base() != other.base():
            R = pushout(self.base(), other.base())
            return self.change_base(R).mul(other.change_base(R))
        elif other.t != self.t:
            t = lcm(self.t, other.t)
            return self.in_sections(t).mul(other.in_sections(t))

        ## Here we can assume that the base ring coincides and the number of sections is the same        
        ## Creating the elements for the compatibility
        A, B = (self.A + other.A), (self.B + other.B)
        coeffs = []
        for b in range(self.t):
            section = []
            for l in range(-A, B+1):
                coeff = ConstantSequence(0, self.base(), 1)
                for i in range(-other.A, other.B+1):
                    s0,s1 = ZZ(b+i).quo_rem(self.t)
                    coeff += other[b,i]*self[s1,l-i].shift(s0)
                section.append(coeff)
            coeffs.append(section)
        
        return Compatibility(coeffs, A, B, self.t)

    def scale(self, factor: Sequence) -> Compatibility:
        if not isinstance(factor, Sequence):
            try:
                factor = ConstantSequence(factor, factor.parent(), 1)
            except Exception:
                raise TypeError(f"[comp-scale] Scaling element must be a Sequence or something with a parent.")
            
        if not factor.dim == 1:
            raise TypeError(f"[comp-scale] THe scaling sequence must have dimension 1")
        elif factor.universe != self.base():
            R = pushout(self.base(), factor.universe)
            return self.change_base(R).scale(factor.change_universe(R))
        
        ## Here we can assume that the base ring coincides 
        return self.mul(Compatibility([[factor]], 0, 0, 1))

    def scale_basis(self, factor: Sequence) -> Compatibility:
        r'''
            Method that computes the equivalent compatibility for the scaled basis.

            Let us assume that a :class:`Compatibility` is associated with a basis `P_k(n)`. Now,
            assume we consider the scaled basis `Q_k(n) = c(k)P_k(n)`. Then the operator
            of ``self`` is also compatible with the new basis and such compatibility is closely related
            with ``self``. Namely, for `k = mt + r`

            .. MATH::

                L Q_k(n) = c(k) L P_k(n) = c(k) \sum_{i=-A}^B \alpha_{r,i}(m) P_{k+i}(n) = \sum_{i=-A}^B \alpha_{r,i}(m) \frac{c(k)}{c(k+i)}Q_{k+i}(n)

            Then the compatibility of `L` w.r.t. `Q_k(n)` is again `(A,B)`-compatible in `t` sections and the new compatibility coefficients are::

            .. MATH::

                \tilde{\alpha}_{r,i}(m) = \alpha_{r,i}(m) \frac{c(mt+r)}{c(mt+r+i)}

            This computation has several layers of simplification:

            * If the sequence `c(k)` is a :class:`RationalSequence`, then all the quotients `\frac{c(mt+r)}{c(mt+r+i)}` are again rational sequences, 
              hence we can use these computations right away.
            * If the sequence `c(k)` is **not** rational but it is hypergeometric. Then `c(k+1)/c(k)` is a rational function and all the quotients necessary 
              can be computed as rational sequences.
            * Otherwise, we simply compute the values as they appear in the formula and hope they are useful later on.
        '''
        is_hyper, quot = factor.is_hypergeometric()
        if isinstance(factor, RationalSequence) or (not is_hyper): # easiest case
            new_coeffs = [[
                self[r,i]*factor.linear_subsequence(0, self.t, r)/factor.linear_subsequence(0, self.t, (r+i)%self.t).shift((r+i)//self.t) 
                for i in range(-self.A, self.B+1)]
            for r in range(self.t)]
        else:
            quots = {0: ConstantSequence(1, self.base())}
            for i in range(1,max(self.A, self.B)+1):
                if i <= self.A:
                    quots[-i] = quots[-i+1]*quot.shift(-i)
                if i <= self.B:
                    quots[i] = quots[i-1]/quot.shift(i-1)
            new_coeffs = [[self[r,i]*quots[i].linear_subsequence(0, self.t, r) for i in range(-self.A, self.B+1)] for r in range(self.t)]

        return Compatibility(new_coeffs, self.A, self.B, self.t)
                
    def __coerce_into_compatibility__(self, other) -> Compatibility:
        if isinstance(other, Compatibility):
            return other
        else:
            return Compatibility([[ConstantSequence(other,self.base(),1)]], 0,0,1)

    def __add__(self, other) -> Compatibility:
        return self.add(self.__coerce_into_compatibility__(other))
    def __radd__(self, other) -> Compatibility:
        return self.__coerce_into_compatibility__(other).add(self)
    def __mul__(self, other) -> Compatibility:
        return self.mul(self.__coerce_into_compatibility__(other))
    def __rmul__(self, other) -> Compatibility:
        return self.__coerce_into_compatibility__(other).mul(self)
    def __pow__(self, other) -> Compatibility:
        if other not in ZZ or other < 0:
            raise TypeError(f"[compatibility] Power only valid for natural numbers")
        if other not in self.__cache_pow:
            if other == 0:
                self.__cache_pow[other] = Compatibility([[ConstantSequence(1, self.base())]], 0,0,1)
            elif other == 1:
                self.__cache_pow[other] = self
            else:
                p1 = other//2
                p2 = p1 + other%2
                comp1 = self**p1
                comp2 = self**p2
                self.__cache_pow[other] = comp1 * comp2
        return self.__cache_pow[other]

    def __repr__(self) -> str:
        start = f"Compatibility condition {self.data()}"
        try:
            M = Matrix([[self[t,i].generic() for i in range(-self.A, self.B+1)] for t in range(self.t)]) #pylint: disable=invalid-unary-operand-type
            start += f" with following coefficient matrix:\n{M}"
        except Exception:
            pass
        return start
    
    def _latex_(self) -> str:
        int_with_sign = lambda n : f"- {-n}" if n < 0 else "" if n == 0 else f"+ {n}"
        code = r"\text{Compatibility condition with shape " + f"(A={self.A}, B={self.B}, t={self.t})" + r":}\\"
        if self.t > 1:
            code += r"\left\{\begin{array}{rl}"
        
        for b in range(self.t):
            if self.t > 1:
                code += r"L \cdot P_{" + latex(self.t) + r"k + " + latex(b) + r"} & = "
            else:
                code += r"L \cdot P_{k} = "

            monomials = []
            for i in range(-self.A, self.B+1): #pylint: disable=invalid-unary-operand-type
                ## Creating the coefficient
                try:
                    c = self[b,i].generic()
                    if c == 0: 
                        continue
                    new_mon = r"\left(" + latex(c) + r"\right)"
                except Exception:
                    if self.t > 1:
                        new_mon = r"c_{" + latex(b) + r"," + latex(i) + r"}(k)"
                    else:
                        new_mon = r"c_{" + latex(i) + r"}(k)"
                ## Creating the P_{k+i}
                if self.t > 1:
                    new_mon += r"P_{" + latex(self.t) + r"k" + int_with_sign(b+i) + r"}"
                else:
                    new_mon += r"P_{k" + int_with_sign(i) + r"}"
                monomials.append(new_mon)
            code += " + ".join(monomials)
            code += r"\\" if self.t > 1 else ""
        if self.t > 1:
            code += r"\end{array}\right."
        return code 

    def equiv(self, other, bound=None) -> bool:
        r'''
            Check equivalence between compatibility conditions

            This method defines the equivalence between two compatibility conditions.
            Let `C_1` and `C_2` be two compatibility conditions with general data 
            `(A_1,B_1,t_1)` and `(A_2,B_2,t_2)` and coefficients `\alpha_{b,i}(n)` and 
            `\beta_{b,i}(n)` respectively.

            * If `t_1 \neq t_2`, we say `C_1 \equiv C_2` if and only if `C_1(t) \equiv C_2(t)`,
              where `t = \lcm(t_1,t_2)` and `C_*(t)` is the compatibility `C_*` in `t` sections 
              (see method :func:`in_sections`)
            * If `t_1 = t_2`, then we check the equality of 

              .. MATH::
                
                \alpha_{b,i}(n) = \beta_{b,i}(n) \ \text{for } b=0,\ldots,t_1-1;\ i=-\max\{A_1,A_2\},\ldots,\max{B_1,B-2\};\ n=0,\ldots,\text{bound}.
            
              where the ``bound`` is given with the optional argument of this method. If not given, we will
              use the default bound for almost equality of the module :mod:`~pseries_basis.sequences`.

            INPUT:

            * ``other``: a :class:`Compatibility` to be checked.
            * ``bound`` (optional): bound for equality of sequences to be used.
        '''
        other = self.__coerce_into_compatibility__(other)

        if self.t != other.t:
            t = lcm(self.t, other.t)
            logger.debug(f"[equiv] We need to extend to more sections ({self.t}, {other.t}) --> {t}")
            return self.in_sections(t).equiv(other.in_sections(t))
        ## Now we assume the sections are the same in both
        A, B = max(self.A, other.A), max(self.B, other.B)
        ## This loop could be more in Python style, we keep it unrolled to keep debugging notes
        for b in range(self.t):
            for i in range(-A, B+1):
                if not self[b,i].almost_equals(other[b,i], bound if bound else 10):
                    logger.debug(f"[equiv] Found different in section {b}, coefficient {i}")
                    return False
        return True

def basis_matrix(basis, nrows=5, ncols=None, section: int = 1, shift: int = 0):
    r'''
        Method to build a matrix with the first part of a basis or a 2-size sequence.

        This method allow the optional arguments ``section`` and shift`` in order to take a subsequence of the basis.
    '''
    if isinstance(basis, PSBasis): # case of a PSBasis
        basis = basis.as_2dim() 
    if not isinstance(basis, Sequence): # case of just a callable
        basis = Sequence(basis, universe=basis(0,0).parent(), dim=2) 

    if ncols is None: 
        ncols = nrows

    if nrows not in ZZ or nrows <= 0: 
        raise TypeError(f"Number of rows must be a positive number (got {nrows=})")
    elif ncols not in ZZ or ncols <= 0: 
        raise TypeError(f"Number of columns must be a positive number (got {ncols=})")
    
    nrows, ncols = ZZ(nrows), ZZ(ncols)

    return Matrix([[basis((section*i+shift,j)) for j in range(ncols)] for i in range(nrows)])

def check_compatibility(basis: PSBasis, compatibility : Compatibility, action: Callable, bound: int = 100, *, _full=False):
    r'''
        Method that checks whether a basis has a particular compatibility for a given action.

        This method takes a :class:`PSBasis` (i.e., a sequence of sequences), a given
        operator compatible with it (or simply the :class:`Compatibility` object representing
        such compatibility) and check whether the action that is defined for the operator/compatibility
        (which is provided by the argument ``action``) has precisely this compatibility.

        More precisely, if an operator `L` the operator is `(A,B)`-compatible with the basis `P=(P_n)_n` 
        with the formula:
        
        .. MATH::

            L P_n = \sum_{i=-A}^B \alpha_{n,i}P_{n+i},

        then this method checks this identity for the ``action`` defining `L`, and the compatibility
        condition `(A,B,m,\alpha)` defined in ``compatibility``.

        This checking is perform until a given `n` bounded by the input ``bound``.

        INPUT:

        * ``basis``: a :class:`PSBasis` that defines the basis `P=(P_n)_n`.
        * ``compatibility``: a compatibility condition. If an operator is given, then compatibility condition
          for ``basis`` is computed (check method :func:`PSBasis.compatibility`)
        * ``action``: a callable that actually computes the element `L P_n` so it can be compared.
        * ``bound``: a bound for the limit this equality will be checked. Since `L P_n` is a sequence
          this bound is used both for checking equality at each level `n` and until which level the 
          identity is checked.

        OUTPUT:

        ``True`` if all the checking provide equality, and ``False`` otherwise. Be cautious when reading
        this output: ``False`` guarantees that the compatibility is **not** for the action, however, ``True``
        provides a nice hint the result should be True, but it is not a complete proof.

        TODO: add examples 
    '''
    if not isinstance(compatibility, Compatibility):
        compatibility = basis.compatibility(compatibility)

    A,B,t = compatibility.data()

    for n in range(A, bound):
        lhs:Sequence = action(basis[n]) # sequence obtained from L P_n
        k,s = ZZ(n).quo_rem(ZZ(t))
        rhs = sum(compatibility[s,i](k)*basis[n+i] for i in range(-A, B+1)) # sequence for the rhs

        if not lhs.almost_equals(rhs, bound):
            return ((n,lhs, rhs), False if _full else False)
    return True

__all__ = ["PSBasis", "Compatibility", "basis_matrix", "check_compatibility"]
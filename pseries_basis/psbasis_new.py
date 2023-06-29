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

    TODO: add examples of usage of this package
'''
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

from collections.abc import Callable
from functools import reduce
from sage.all import cached_method, lcm, Matrix, ZZ
from sage.categories.pushout import pushout
from .sequences.base import ConstantSequence, Sequence, SequenceSet

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
    def __init__(self, sequence: Callable[[int], Sequence], universe=None, *, _extend_by_zero=False):
        # We check the case we provide a 2-dimensional sequence
        if isinstance(sequence, Sequence) and sequence.dim == 2:
            universe = sequence.universe if universe is None else universe; or_sequence = sequence
            sequence = lambda n : or_sequence.slicing((0,n))
        else:
            if universe is None:
                raise ValueError(f"When argument is callable we require a universe argument for the basis.")
        self.__inner_universe = universe
        super().__init__(sequence, SequenceSet(1, universe), 1, _extend_by_zero=_extend_by_zero)

    ##########################################################################################################
    ###
    ### PROPERTY METHODS
    ### 
    ##########################################################################################################
    @property
    def base(self):
        return self.__inner_universe
    
    # TODO def change_base(self, base : Parent) -> PSBasis
    
    def _element(self, *indices: int):
        return super()._element(*indices).change_universe(self.base)
    
    @cached_method
    def as_2dim(self) -> Sequence:
        return Sequence(lambda n,k : self(n)(k), self.base, 2, _extend_by_zero = self._Sequence__extend_by_zero)
    
    ##########################################################################################################
    ###
    ### SEQUENCE METHODS
    ### 
    ##########################################################################################################
    # TODO * :func:`_change_class`: receives a class (given when registering the sequence class) and cast the current sequence to the new class.
    #       - :func:`_change_from_class`: class method that receives a sequence in a different class and transform into the current class. 
    #       - :func:`_neg_`: implement the negation of a sequence for a given class.
    #       - :func:`_final_add`: implement the addition for two sequences of the same parent and class.
    #       - :func:`_final_sub`: implement the difference for two sequences of the same parent and class.
    #       - :func:`_final_mul`: implement the hadamard product for two sequences of the same parent and class.
    #       - :func:`_final_div`: implement the hadamard division for two sequences of the same parent and class.
    #       - :func:`_final_mod`: implement the hadamard module for two sequences of the same parent and class.
    #       - :func:`_final_floordiv`: implement the hadamard floor division for two sequences of the same parent and class.
    #     * Consider updating the following methods:
    #       - :func:`_shift` to adjust the output of the shifted sequence.
    #       - :func:`_subsequence` to adjust the output of the subsequence.
    #       - :func:`_slicing` to adjust the output of the sliced sequence.

    # TODO def scalar(self, factor: element.Element) -> PSBasis
    # TODO def _scalar_basis(self, factor: element.Element) -> PSBasis
    # TODO def _scalar_hypergeometric(self, factor: element.Element, quotient: element.Element) -> PSBasis
    # TODO def __scalar_extend_compatibilities(self, new_basis: PSBasis, factor: element.Element)
    # TODO def __scalar_hyper_extend_compatibilities(self, new_basis: PSBasis, factor: element.Element, quotient: element.Element)

    ##########################################################################################################
    ###
    ### INFORMATION OF THE BASIS
    ### 
    ##########################################################################################################
    @property
    def functional_seq(self) -> Sequence:
        r'''
            Method to get the functional sequence of the basis.

            A :class:`PSBasis` can be seen as a sequence of functions or polynomials. However, these
            functions and polynomials are sequences by themselves. In fact, a :class:`PSBasis` is 
            a basis of the ring of formal power series which, at the same time, is a basis of the 
            ring of sequences. 

            However, the relation between the sequences and the formal power series is not unique:

            * We can consider the formal power series `f(x) = \sum_n a_n x^n`, then we have the 
              natural (also called functional) sequence `(a_n)_n`.
            * We can consider formal power series as functions `f: \mathbb{K} \rightarrow \mathbb{K},
              and (if convergent) we can define the (evaluation) sequence `(f(n))_n`.

            This method returns a bi-indexed sequence that allows to obtain the functional sequences
            of this basis. This is equivalent to the bi-dimensional sequence that defines the basis
            (see method :func:`as_2dim`)
        '''
        return self.as_2dim()
    
    def functional_matrix(self, nrows: int, ncols: int = None):
        r'''
            Method to get a matrix representation of the basis.

            This method returns a matrix `\tilde{M} = (m_{i,j})` with ``nrows`` rows and
            ``ncols`` columns such that `m_{i,j} = [x^j]f_i(x)`, where `f_i(x)` is 
            the `i`-th element of ``self``.

            This is the upper-left part of the matrix `M` that represent, by rows,
            the elements of this basis in terms of the canonical basis of the formal
            power series ring (`\{1,x,x^2,\ldots\}`). Hence, if we have an element 
            `y(x) \in \mathbb{K}[[x]]` with:

            .. MATH::

                y(x) = \sum_{n\geq 0} y_n x^n = \sum_{n\geq 0} c_n f_n(x),

            then the infinite vectors `\mathbf{y}` and `\mathbf{c}` satisfies:

            .. MATH::

                \mathbf{y} = \mathbf{c} M

            INPUT:

            * ``nrows``: number of rows of the final matrix
            * ``ncols``: number of columns of the final matrix. If ``None`` is given, we
              will automatically return the square matrix with size given by ``nrows``.
        '''
        ## Checking the arguments
        if(not ((nrows in ZZ) and nrows > 0)):
            raise ValueError("The number of rows must be a positive integer")
        if(ncols is None):
            ncols = nrows
        elif(not ((ncols in ZZ) and ncols > 0)):
                raise ValueError("The number of columns must be a positive integer")

        return Matrix([[self.functional_seq((i,j)) for j in range(ncols)] for i in range(nrows)])
    
    def is_quasi_func_triangular(self, bound:int=20) -> bool:
        r'''
            Method to check whether a basis is quasi-triangular or not as a functional basis up to a bound

            A basis `\mathcal{B} = \{f_n(x)\}_n` is a functional *quasi-triangular* if its 
            functional matrix representation `M = \left(m_{n,k}\right)_{n,k \geq 0}` (see 
            :func:`functional_matrix`) is *quasi-upper triangular*, i.e.,
            there is a strictly monotonic function `I: \mathbb{N} \rightarrow \mathbb{N}` such that

            * For all `k \in \mathbb{N}`, and `m > I(k)`, `m_{n,k} =0`, i.e., `x^k` divides `f_n(x)`.
            * For all `k \in \mathbb{N}`, `m_{I(k),k} \neq 0`, i.e., `x^k` does not divide `f_{I(k)}(x)`.

            This property will allow to transform the initial conditions from the canonical basis
            of formal power series (`\{1,x,x^2,\ldots\}`) to the initial conditions of the expansion
            over ``self``.
        '''
        M = self.functional_matrix(bound)
        I = M.ncols()*[None]
        for i in range(M.ncols()):
            for j in range(M.nrows()-1, -1, -1):
                if M[i,j] != 0:
                    I[i] = j+1; break
            else:
                I[i] = 0

        for i in range(M.ncols()-1):
            if (I[i+1] not in (0, bound)) and I[i+1] <= I[i]:
                return False
        return True
    
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
    # TODO def _compatibility_from_recurrence(self, recurrence: OreOperator) -> TypeCompatibility
    # TODO def set_compatibility(self, name: str, trans: OreOperator | TypeCompatibility, sub: bool = False, type: None | str = None)
    # TODO def set_endomorphism(self, name: str, trans: OreOperator | TypeCompatibility, sub: bool = False)
    # TODO def set_derivation(self, name: str, trans: OreOperator | TypeCompatibility, sub: bool = False)
    # @cached_method
    # TODO def get_lower_bound(self, operator: str | OreOperator) -> int
    # @cached_method
    # TODO def get_upper_bound(self, operator: str | OreOperator) -> int
    # TODO def compatible_operators(self) -> Collection[str]
    # TODO def compatible_endomorphisms(self) -> Collection[str]
    # TODO def compatible_derivations(self) -> Collection[str]
    # TODO def has_compatibility(self, operator: str | OreOperator) -> bool
    # TODO def compatibility_type(self, operator: str | OreOperator) -> None | str
    # TODO def compatibility(self, operator: str | OreOperator) -> TypeCompatibility
    # TODO def compatibility_matrix(self, operator: str | OreOperator, sections: int = None) -> matrix_class
    # @cached_method
    # TODO def compatibility_sections(self, compatibility: str | OreOperator | TypeCompatibility, sections : int) -> TypeCompatibility
    # @cached_method
    # TODO def compatibility_coefficient(self, operator: str | OreOperator) -> Callable
    
    ##########################################################################################################
    ###
    ### RECURRENCES METHODS
    ### 
    ##########################################################################################################
    # TODO def _recurrence_from_compatibility(self, compatibility: TypeCompatibility) -> OreOperator
    # TODO def recurrence(self, operator: str | OreOperator | TypeCompatibility, sections: int = None, cleaned: bool = False) -> OreOperator | matrix_class
    # TODO def recurrence_orig(self, operator: str | OreOperator | TypeCompatibility) -> OreOperator
    # TODO def simplify_operator(self,operator: OreOperator) -> OreOperator
    # TODO def _simplify_operator(self, operator: OreOperator) -> OreOperator
    # TODO def remove_Sni(self, operator: OreOperator) -> OreOperator

    ##########################################################################################################
    ###
    ### MAGIC METHODS
    ### 
    ##########################################################################################################
    # TODO def __mul__
    # TODO def __rmul__
    # TODO def __truediv__
    # TODO def __repr__(self)

    ##########################################################################################################
    ###
    ### OTHER METHODS
    ### 
    ##########################################################################################################
    # TODO def system(self, operator: str | OreOperator | TypeCompatibility, sections: int = None)
    
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
        elif (not A is None) and (not B is None) and (len(c[0]) != A+B+1):
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

    @property
    def upper(self): return self.__upper #: Property of the upper bound for the compatibility
    @property
    def lower(self): return self.__lower #: Property of the lower bound for the compatibility
    @property
    def nsections(self): return self.__nsections #: Property of the number of sections for the compatibility
    A: int = upper #: alias for the upper bound
    B: int = lower #: alias for the lower bound
    t: int = nsections # alias for the number of sections

    def data(self):
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
        elif item[1] < -self.A or item[1] > self.B:
            return self.__data[item[0], item[1]-self.A]
        else:
            return ConstantSequence(0, self.base(), 1)
        
    def in_sections(self, new_sections: int) -> Compatibility:
        r'''
            Method to compute the compatibility condition with respect to more sections.
        '''
        if new_sections%self.t != 0:
            raise ValueError(f"[compatibility] Compatibilities can only be extended when the new sections ({new_sections}) are multiple of previous sections ({self.t}).")
        
        A = self.A; B = self.B
        a = new_sections // self.t # proportion to new sections

        coeffs = []
        for s in range(new_sections):
            new_section = []
            for i in range(-A, B+1):
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
        A = max(self.A, other.A); B = max(self.B, other.B)
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
        A = self.A + other.A; B = self.B + other.B
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
            except:
                raise TypeError(f"[comp-scale] Scaling element must be a Sequence or something with a parent.")
            
        if not factor.dim == 1:
            raise TypeError(f"[comp-scale] THe scaling sequence must have dimension 1")
        elif factor.universe != self.base():
            R = pushout(self.base(), factor.universe)
            return self.change_base(R).scale(factor.change_universe(R))
        
        ## Here we can assume that the base ring coincides 
        return self.mul(Compatibility([[factor]], 0, 0, 1))

# TODO def check_compatibility(basis: PSBasis, operator: OreOperator | TypeCompatibility, action: Callable, bound: int = 100)

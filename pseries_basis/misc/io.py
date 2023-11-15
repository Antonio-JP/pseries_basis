
r'''
    File with some connections of the :mod:`pseries_basis` with other packages.

    The methods, classes and functionalities in this module are focused on the 
    interaction of sequences, basis, and solutions with things outside SageMath.
'''

import logging
import re

from sage.all import ZZ, var, lcm, cached_method
from sage.databases.oeis import OEISSequence

from ore_algebra.ore_operator import OreOperator

from .ore import poly_decomposition, get_recurrence_algebra, solution, required_init
from .sequences import Sequence, LambdaSequence

logger = logging.getLogger(__name__)

def operator2Mathematica(operator: OreOperator, sequence_var: str = None) -> str:
    r'''
        Method to write an :class:`~ore_algebra.ore_operator.OreOperator` into a string

        This method writes down an Ore Operators as a string where the possible meaning 
        of extra variables (see the case of `q`-sequences) is translated.

        These strings should be valid input as Mathematica code.

        INPUT: 

        * ``operator``: a :class:`~ore_algebra.ore_operator.OreOperator` to write down to a file.
        * ``sequence_var``: the variable to be substitute for the sequence name in Mathematica. If not
          given, we leave the operators as they are.

        EXAMPLES::

            sage: from pseries_basis.misc.ore import *
            sage: from pseries_basis.misc.io import operator2Mathematica
            sage: A, (n, E) = get_recurrence_algebra("n")
            sage: L = n^3 *E^2 + (n^2 + 1)*E + (n-1)
            sage: operator2Mathematica(L)
            '(n - 1)*1+(n^2 + 1)*E+(n^3)*E^2'
            sage: operator2Mathematica(L, "a")
            '(n - 1)*a[n]+(n^2 + 1)*a[n+1]+(n^3)*a[n+2]'
            sage: A, (n,E,Ei) = get_double_recurrence_algebra("n")
            sage: L = n^3 *E^2 + (n^2 + 1)*E + (n-1) + (n^3+1)*Ei - (2*n-1)*Ei^2
            sage: operator2Mathematica(L)
            '(n^3)*E^2+(-2*n + 1)*E^(-2)+(n^2 + 1)*E+(n^3 + 1)*E^(-1)+(n - 1)*1'
            sage: operator2Mathematica(L, "t")
            '(n^3)*t[n+2]+(-2*n + 1)*t[n-2]+(n^2 + 1)*t[n+1]+(n^3 + 1)*t[n-1]+(n - 1)*t[n]'

        We can also have `q`-shifts in the mix. Then the variable `qn` associated is translated into
        the corresponding `q^n`::

            sage: A, (qn,S) = get_qshift_algebra('qn', name_qshift="S", base=QQ['q'])
            sage: q = A.base()('q')
            sage: L = qn^3 *S^2 + (qn^2 + 1)*S + (qn-1)
            sage: operator2Mathematica(L)
            '(q^(1*n) - 1)*1+(q^(2*n) + 1)*S+(q^(3*n))*S^2'
            sage: operator2Mathematica(L, "p")
            '(q^(1*n) - 1)*p[n]+(q^(2*n) + 1)*p[n+1]+(q^(3*n))*p[n+2]'
            sage: A, (qn,S) = get_qshift_algebra('qn', name_qshift="S", base=QQ['q'], power=2)
            sage: q = A.base()('q')
            sage: L = q^3*qn^3 *S^2 + (q*qn^2 + 1)*S + (qn-q^2)
            sage: operator2Mathematica(L)
            '(q^(2*n) - q^2)*1+(q*q^(4*n) + 1)*S+(q^3*q^(6*n))*S^2'
            sage: operator2Mathematica(L, "uu")
            '(q^(2*n) - q^2)*uu[n]+(q*q^(4*n) + 1)*uu[n+1]+(q^3*q^(6*n))*uu[n+2]'
            sage: A, (qn,S,Si) = get_double_qshift_algebra('qn', name_qshift="S", base=QQ['q'])
            sage: q = A.base()('q')
            sage: L = qn^3 *S^2 + (qn^2 + 1)*S + (qn-1) + Si - qn*q*Si^2
            sage: operator2Mathematica(L)
            '(q^(3*n))*S^2+(-q*q^(1*n))*S^(-2)+(q^(2*n) + 1)*S+(1)*S^(-1)+(q^(1*n) - 1)*1'
            sage: operator2Mathematica(L, "y")
            '(q^(3*n))*y[n+2]+(-q*q^(1*n))*y[n-2]+(q^(2*n) + 1)*y[n+1]+(1)*y[n-1]+(q^(1*n) - 1)*y[n]'
    '''
    mons, coeffs = poly_decomposition(operator.polynomial())
    from .ore import (is_double_qshift_algebra, is_qshift_algebra, gens_double_qshift_algebra, gens_qshift_algebra,
                      is_recurrence_algebra, is_double_recurrence_algebra, gens_recurrence_algebra, gens_double_recurrence_algebra)
    import re
    
    ## Getting the variables of the parent structure
    if is_double_qshift_algebra(operator.parent()):
        qn, S, S_i, q, ex = gens_double_qshift_algebra(operator.parent(), "q")
    elif is_qshift_algebra(operator.parent()):
        qn, S, q, ex = gens_qshift_algebra(operator.parent(), "q"); S_i = None
    elif is_double_recurrence_algebra(operator.parent()):
        _, S, S_i, _ = gens_double_recurrence_algebra(operator.parent())
    elif is_recurrence_algebra(operator.parent()):
        _, S, _ = gens_recurrence_algebra(operator.parent()); S_i = None

    ## Changing the coefficients to strings
    if is_double_qshift_algebra(operator.parent(), "q") or is_qshift_algebra(operator.parent(), "q"):
        coeffs = [re.sub(f"{qn}\\^(\\d+|[\\(-?\\d+\\)])", lambda M : f"{q}^({ex*int(M.groups()[0])}*n)", str(coeff)) for coeff in coeffs]
        coeffs = [re.sub(f"{qn}", lambda M : f"{q}^({ex}*n)", str(coeff)) for coeff in coeffs]
    else:
        coeffs = [str(coeff) for coeff in coeffs]

    if sequence_var != None:
        if S_i != None: ## First remove the inverse shift if existed
            mons = [re.sub(
                f"{S_i}(\\^\\(?\\d+\\)?)?", 
                lambda M : f"{sequence_var}[n-{M.groups()[0].removeprefix('^')}]" if M.groups()[0] != None else f"{sequence_var}[n-1]",
                str(mon))
            for mon in mons]
        ## Then we remove the direct shift
        mons = [re.sub(
            f"{S}(\\^\\(?\\d+\\)?)?", 
            lambda M : f"{sequence_var}[n+{M.groups()[0].removeprefix('^')}]" if M.groups()[0] != None else f"{sequence_var}[n+1]",
            str(mon))
        for mon in mons]

        ## We also transform the '1' into sequence notation
        if "1" in mons:
            mons[mons.index("1")] = f"{sequence_var}[n]"
    else:
        if S_i != None: # We transform the inverse shift to negative powers of the direct shift
            mons = [re.sub(
                f"{S_i}(\\^\\(?\\d+\\)?)?", 
                lambda M : f"{S}^(-{M.groups()[0].removeprefix('^')})" if M.groups()[0] != None else f"{S}^(-1)",
                str(mon))
            for mon in mons]
        else:
            mons = [str(mon) for mon in mons]

    return "+".join(f"({coeff})*{mon}" for mon,coeff in zip(mons, coeffs))

def operator2file(operator : OreOperator, file : str):
    r'''
        Method to write an :class:`~ore_algebra.ore_operator.OreOperator` into a file

        This method writes down an Ore Operators to a file (given as argument) with a very
        simplistic format: each line is the coefficient of a monomial times the monomial itself.

        INPUT: 

        * ``operator``: a :class:`~ore_algebra.ore_operator.OreOperator` to write down to a file.
        * ``file``: string with the path to the file were the operator will be written.
    '''
    with open(file, "w") as f:
        mons, coeffs = poly_decomposition(operator.polynomial())
        for i in range(len(mons)):
            f.write(f"({coeffs[i]})*{mons[i]} {'+' if i < len(mons)-1 else ''}\n")
    return

####################################################################################################
###
### RELATION WITH OEIS
###
####################################################################################################
class EnhOEISSequence(OEISSequence):
    r'''
        Enhanced class for OEIS sequences.

        This class extends the functionality of an OEIS sequence. We provide several methods to explore
        the information and the data of the sequence and provides a unified interface for the sequence
        with respect to the package :mod:`pseries_basis`

        INPUT:

        * ``ident``: either a string with the A identifier of the sequence or a :class:`OEISSequence`.
    '''
    @staticmethod
    def __classcall__(cls, ident : str | OEISSequence):
        if isinstance(ident, OEISSequence):
            ident = ident.id()
        return super(EnhOEISSequence, cls).__classcall__(cls, ident)

    @cached_method
    def is_dfinite(self) -> bool:
        r'''
            Method to check whether a sequence is D-finite or not.
        '''
        return not (self.dfinite_recurrence() is None)

    @cached_method
    def dfinite_recurrence(self) -> OreOperator:
        r'''
            Method to compute a difference D-finite recurrence for an OEIS Sequence

            This method reads from the formulas on the OEIS Sequence and computes a recurrence equation
            that is valid for this sequence (for all `n`). The computations in this method requires
            parsing and extracting information from the text of the formulas. We make use of some generic
            formatting in OEIS that does not always hold. Hence, sometimes this method returns ``None``
            even when the sequence is D-finite.
        '''
        formulas = [el for el in self.formulas() if el.find("ecurrence") >= 0]
        for formula in formulas:
            try:
                # removing the first part until "recurrence"
                start_pos = formula.find("ecurrence") + len("ecurrence")
                if formula[start_pos] in (":"): # removing possible connectors like ":".
                    start_pos += 1
                # getting the part of the formula until a "with" a "." or the end of the line
                operator = self.__analyze_formula(formula, start_pos)
                if operator != None:
                    logger.info(f"dfinite_recurrence: found a recurrence operator for OEIS sequence {self.id()}")
                    return operator
                else:
                    logger.info(f"dfinite_recurrence: no valid arguments in the formula")
            except Exception as e:
                logger.info(f"dfinite_recurrence: Unexpected error {e}")
        logger.info(f"dfinite_recurrence: no recurrence operator found for OEIS sequence {self.id()} with the string 'recurrence'")
        for formula in self.formulas(): # trying to find a plain formula
            try:
                operator = self.__analyze_formula(formula, 0)
                if operator != None:
                    logger.info(f"dfinite_recurrence: found a recurrence operator for OEIS sequence {self.id()}")
                    return operator
                else:
                    logger.info(f"dfinite_recurrence: no valid arguments in the formula")
            except Exception as e:
                logger.info(f"dfinite_recurrence: Unexpected error {e}")

        logger.info(f"dfinite_recurrence: no recurrence operator found for OEIS sequence {self.id()}")
        return None

    def order(self) -> int:
        r'''
            Method to get the order of a D-finite OEIS sequence (if it is D-finite)
        '''
        if self.is_dfinite():
            return self.dfinite_recurrence().order()
        raise TypeError("The sequence is not D-finite")

    @property
    def sequence(self) -> Sequence:
        off = self.offsets()[0]
        if self.is_dfinite():
            d = required_init(self.dfinite_recurrence())
            return solution(self.dfinite_recurrence(),tuple(off*[0] + list(self.first_terms(d+1))))
        return LambdaSequence(lambda n : self.first_terms(n+1-off)[-1] if n >= off else None, ZZ)

    def check_consistency(self, bound: int = 10) -> bool:
        r'''
            Method to check the consistency between the generated sequence and the terms in the OEIS database

            The offset of the sequences in OEIS may disturb the recurrence equations, hence, we need that
            the corresponding elements of the generated sequence coincide with the terms of the sequence in OEIS.

            If this method return ``False``, it means we need to improve the attribute :func:`sequence`.
        '''
        if(self.is_dfinite()):
            off = self.offsets()[0]
            return tuple(self.sequence[off:off+bound]) == self.first_terms(bound)
        return True

    def __find_several_first(self, string : str, *to_find : str, start: int = 0) -> int:
        r'''
            Finds one of the strings to find returning the first appearance of them. It returns -1 if none appears.
        '''
        pos = [string.find(part,start) for part in to_find]
        pos = [el for el in pos if el >= 0]
        if len(pos) == 0:
            return -1
        return min(pos)

    def __analyze_formula(self, formula: str, start_pos: int) -> OreOperator:
        n = var('n')
        # getting the part of the formula until a "with" a "." or the end of the line
        end_pos = self.__find_several_first(formula, ".", " with", ",", "[", start = start_pos)
        end_pos = len(formula) if end_pos < 0 else end_pos
        formula = formula[start_pos:end_pos]

        # we compute the orders appearing in the recurrence
        arguments = [eval(el)-n for el in re.findall(r"a\(([^\)]*)\)", formula)]
        arguments = [ZZ(el) for el in arguments if el in ZZ]

        if(len(arguments) > 0): # it is a recurrence equation and not only initial conditions
            neg_order = -min(0, min(arguments))
            return self.__extract_recurrence(formula, neg_order)
        else:
            logger.info(f"dfinite_recurrence: no valid arguments in the formula")
        return None

    def __extract_recurrence(self, formula: str, neg_order: int) -> OreOperator:
        OE, _ = get_recurrence_algebra('x', 'E')
        parts = [el.strip() for el in formula.split(".")[0].split(",")]
        # substitution reg
        reg_expressions = [(r"a\(n\+(\d+)\)", r"E**(\1 + "+str(neg_order)+")"),
            (r"a\(n-(\d+)\)", r"E**(-\1 + "+str(neg_order)+")"),
            (r"a\(n\)", r"E**("+str(neg_order)+")"),
            (r"(\d+)n", r"\1*n")]
        def __apply_iterative(reg_expressions, string):
            for reg in reg_expressions:
                string = re.sub(*reg, string)
            return string
        for part in parts:
            part = part.replace(":", "")
            if part.find(" = ") > -1:
                lhs, rhs = part.split(" = ")
                try:
                    lhs = __apply_iterative(reg_expressions, lhs)
                    lhs = lhs.replace("n", f"(x+{neg_order})")
                    rhs = __apply_iterative(reg_expressions, rhs)
                    rhs = rhs.replace("n", f"(x+{neg_order})")
                
                    output = OE(lhs) - OE(rhs)
                    _, coeffs = poly_decomposition(output.polynomial())
                    cleaned_denom = lcm([el.denominator() for el in coeffs])
                    return get_recurrence_algebra('x','E', rational=False)[0](cleaned_denom * output)

                except:
                    logger.info(f"extract_recurrence: Can not convert this: {part} (({lhs} ||| {rhs}))")
                    pass
        return None
    
__all__ = ["operator2file", "EnhOEISSequence"]
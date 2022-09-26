
r'''
    File with some connections of the :mod:`pseries_basis` with other packages.

    The methods, classes and functionalities in this module are focused on the 
    interation of sequences, basis, and solutions with things outside SageMath.
'''

import logging
import re
from functools import lru_cache

from sage.all import ZZ, var, lcm
from sage.databases.oeis import OEISSequence

from ore_algebra.ore_operator import OreOperator

from pseries_basis.sequences import LambdaSequence

from .ore import poly_decomp, get_recurrence_algebra, solution, required_init

logger = logging.getLogger(__name__)

def operator2file(operator : OreOperator, file : str):
    r'''
        Method to write an :class:`~ore_algebra.ore_opeerator.OreOperator` into a file

        This method writes down an Ore Operators to a file (given as argument) with a very
        simplistic format: each line is the coefficient of a monomial times the monomial itself.

        INPUT: 

        * ``operator``: a :class:`~ore_algebra.ore_operator.OreOperator` to write down to a file.
        * ``file``: string with the path to the file were the operator will be written.
    '''
    with open(file, "w") as f:
        mons, coeffs = poly_decomp(operator.polynomial())
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
    def __classcall__(cls, ident):
        if isinstance(ident, OEISSequence):
            ident = ident.id()
        return super(EnhOEISSequence, cls).__classcall__(cls, ident)

    @lru_cache
    def is_dfinite(self):
        r'''
            Method to check whether a sequence is D-finite or not.
        '''
        return not (self.dfinite_recurrence() is None)

    @lru_cache
    def dfinite_recurrence(self):
        r'''
            Method to compute a difference D-finite recurrence for an OEIS Sequence

            This method reads from the formulas on the OEIS Sequence and computes a recurrence equation
            that is valid for this sequence (for all `n`). The computations in this method requires
            parsing and extracting information from the text of the formulas. We make use of some generic
            formating in OEIS that does not always hold. Hence, sometimes this method returns ``None``
            even when the sequence is D-finite.
        '''
        formulas = [el for el in self.formulas() if el.find("ecurrence") >= 0]
        for formula in formulas:
            try:
                # removing the first part until "recurrence: "
                start_pos = formula.find("ecurrence: ") + len("ecurrence: ")
                # getting the part of the formula unitl a "with" a "." or the end of the line
                end_pos = self.__find_several_first(formula, ".", " with", ",", start = start_pos)
                end_pos = len(formula) if end_pos < 0 else end_pos
                formula = formula[start_pos:end_pos+1]

                # we compute the orders appearing in the recurrence
                n = var('n')
                arguments = [eval(el)-n for el in re.findall(r"a\(([^\)]*)\)", formula)]
                arguments = [ZZ(el) for el in arguments if el in ZZ]

                if(len(arguments) > 0): # it is a recurrence equation and not only initial conditions
                    neg_order = -min(0, min(arguments))
                    operator = self.__extract_recurrence(formula, neg_order)
                    if operator != None:
                        logger.info(f"dfinite_recurrence: found a recurrence operator for OEIS sequence {self.id()}")
                        return operator
                else:
                    logger.info(f"dfinite_recurrence: no valid arguments in the formula")
            except Exception as e:
                logger.info(f"dfinite_recurrence: Unexpected error {e}")
        logger.info(f"dfinite_recurrence: no recurrence operator found for OEIS sequence {self.id()}")
        return None

    def order(self):
        r'''
            Method to get the order of a D-finite OEIS sequence (if it is D-finite)
        '''
        if self.is_dfinite():
            return self.dfinite_recurrence().order()
        raise TypeError("The sequence is not D-finite")

    @property
    def sequence(self):
        if self.is_dfinite():
            d = required_init(self.dfinite_recurrence())
            return solution(self.dfinite_recurrence(),self.first_terms(d+1))
        return LambdaSequence(lambda n : self.first_terms(n+1)[-1], ZZ)

    def __find_several_first(self, string, *to_find, start=0):
        r'''
            Finds one of the strings to find returning the first appearance of them. It returns -1 if none appears.
        '''
        pos = [string.find(part,start) for part in to_find]
        pos = [el for el in pos if el >= 0]
        if len(pos) == 0:
            return -1
        return min(pos)

    def __extract_recurrence(self, formula, neg_order):
        OE, _ = get_recurrence_algebra('x', 'E')
        parts = [el.strip() for el in formula.split(".")[0].split(",")]
        # substitution reg
        regs = [(r"a\(n\+(\d+)\)", r"E**(\1 + "+str(neg_order)+")"),
            (r"a\(n-(\d+)\)", r"E**(-\1 + "+str(neg_order)+")"),
            (r"a\(n\)", r"E**("+str(neg_order)+")"),
            (r"(\d+)n", r"\1*n")]
        def __apply_iterative(regs, string):
            for reg in regs:
                string = re.sub(*reg, string)
            return string
        for part in parts:
            part = part.replace(":", "")
            if part.find(" = ") > -1:
                lhs, rhs = part.split(" = ")
                try:
                    lhs = __apply_iterative(regs, lhs)
                    lhs = lhs.replace("n", f"(x+{neg_order})")
                    rhs = __apply_iterative(regs, rhs)
                    rhs = rhs.replace("n", f"(x+{neg_order})")
                
                    output = OE(lhs) - OE(rhs)
                    _, coeffs = poly_decomp(output.polynomial())
                    cleaned_denom = lcm([el.denominator() for el in coeffs])
                    return get_recurrence_algebra('x','E', rational=False)[0](cleaned_denom * output)

                except:
                    logger.info(f"extract_recurrence: Can not convert this: {part} (({lhs} ||| {rhs}))")
                    pass
        return None
    
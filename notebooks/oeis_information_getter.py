from alive_progress import alive_bar

from sage.all import oeis

from ore_algebra import guess

from pseries_basis import *
from pseries_basis.io import *

def search_dfinite_order(min_order, min_results = 5, i = 0):
    r'''
        Method to get sequences from OEIS.
        
        This method looks in OEIS for D-finite sequences, returning all the results
        found in bunches of 100 until it finds at least ``min_results`` of a fixed minimal 
        order given in ``min_order``.
        
        This function start looking in the element `100 i` where ``i`` is given also as an 
        argument of this function.
        
        This method requires Internet connection to get the information from OEIS.
        
        INPUT:
        
        * ``min_order``: minimal order for the sequences returned by the method.
        * ``min_results``: minimal number of elements that will be returned.
        * ``i``: original shift in the search of OEIS sequences.
    '''
    import re
    shift = i+1
    results = []
    with alive_bar(int(min_results), title="Searching examples...", force_tty=True) as bar:
        while(len(results) < min_results):
            sequences = [EnhOEISSequence(el) for el in oeis("D-finite", max_results = 100, first_result = (shift-1)*100)]            
            if len(sequences) == 0: # checking if OEIS is done
                raise KeyboardInterrupt("No more D-finite sequences")
            
            # fitering out the non-desired results
            sequences = [seq for seq in sequences if seq.is_dfinite()] # filtering results we can extract the recurrence
            sequences = [seq for seq in sequences if seq.order() >= min_order] # filtering results that have to low order

            for seq in sequences:
                results.append(seq)
                bar()
            shift += 1
    return results, shift

def explore_sequence(sequence, ring, first_bases, second_bases, bound_guess=100):
    r'''
        Method to explore how to write a solution for a recurrence equation in a double sum
        
        This method takes the solution `(a_n)_n` or the recurrence equation defined by ``operator``
        and ``init`` (see method :func:`solution`) and computes a new sequence `(y_n)_n` such that
        
        .. MATH::
        
            y_n = \sum_{k} \sum_{l} a_l B_{2,l}(k) B_{1,k}(n)
            
        Then it would be easy to check whether this sequence is useful for us or not.
        
        The arguments ``first_bases`` and ``second_bases`` provide lists for `B_{1,k}(n)` and `B_{2,l}(k)`. One output for each 
        pair of basis will be computed. If these entries are just one element instead of a list or tuple, then we consider the 
        input as a one element list.
        
        INPUT:
        
        * ``operator``: a difference operator valid for the use of :func:`solution`
        * ``init``: initial values for the sequence (see :func:`solution`)
        * ``first_bases``: list of :class:`PSBasis` compatible with the recurrence operator.
        * ``second_bases``: list of :class:`PSBasis` compatible with the recurrence operator.
    '''
    ## Checking the input
    if not isinstance(first_bases, (list, tuple)):
        first_bases = [first_bases]
    if not isinstance(second_bases, (list, tuple)):
        second_bases = [second_bases]
        
    if any(not isinstance(B, PSBasis) for B in first_bases+second_bases):
        raise TypeError("All basis must be of class PSBasis")
    if any(not B.has_compatibility('E') for B in first_bases+second_bases):
        raise TypeError("All basis must be compatible with 'E'")
    
    result = {}
    for B1 in first_bases:
        for B2 in second_bases:
            b1_sequence = LambdaSequence(lambda n : sum(B1[k](n)*sequence[k] for k in range(n+1)), QQ)
            b2_b1_sequence = LambdaSequence(lambda n : sum(B2[k](n)*b1_sequence[k] for k in range(n+1)), QQ)
            try:
                L = guess(b2_b1_sequence[:bound_guess], ring)
                result[(B1,B2)] = (b2_b1_sequence, L)
            except ValueError:
                result[(B1,B2)] = f"No D-finite recurrence found with {bound_guess} data"
    # case where only 1 pair is required
    if len(result) == 1:
        return result[(first_bases[0], second_bases[0])]
    return result

if __name__=="__main__": # Main part of the script
    first_bases = []
    second_bases = []
    min_order = 3
    results_search = 50
    shift = 0
    # Treating the arguments
    while(True):
        try:
            dfinite, shift = search_dfinite_order(min_order,min_results=results_search,i=shift)
            
            with alive_bar(len(dfinite), title="Exploring examples...", force_tty=True) as bar:
                for el in dfinite:
                    explored = [(seq,explore_sequence(seq.sequence(), seq.dfinite_recurrence().parent(), first_bases, second_bases)) for seq in dfinite]
                    bar()
            with_oeis = []
            with alive_bar(len(dfinite), title="Checking if new is in OEIS...", force_tty=True) as bar:
                for seq,data in explored:
                    if not isinstance(data, dict):
                        data = {(first_bases[0], second_bases[0]): data}
                        
                    for (k,v) in data.items():
                        if not isinstance(v, str):
                            ass_seqs = oeis(v[0][:10])
                            if len(ass_seqs) > 0:
                                with_oeis.append((seq,k,[el.id() for el in ass_seqs]))
                                
            print("Saving succesful explroed examples")
            with open("./example_double_sum/explored.txt", "w+") as exp_file:
                for seq, data in explored:
                    exp_file.write(f"{seq.id()}\n")
                    if not isinstance(data, dict):
                        data = {(first_bases[0], second_bases[0]): data}
                    
                    to_print = "\t".join([f"({k[0]},{k[1]}) ---> [{v[1]}; {v[0][:required_init(v[1])+1]}]\n" for (k,v) in data.items()])
                    exp_file.write(to_print)
            with open("./example_double_sum/with_oeis.txt", "w+") as oeis_file:
                for seq, bases, ass_seqs in with_oeis:
                    oeis_file.write(f"{seq.id()} -- ({bases[0]},{bases[1]}) --> {ass_seqs}\n")
            
        except KeyboardInterrupt:
            break
        except (Exception, RuntimeError):
            break
    print(f"Shift -> {shift}")
    
    
r'''
    Module with the classes necessary to define power series basis based on `q`-analogs.

    A `q`-series is a sequence where the `n`-th element is a function of `q`. There 
    is plenty of work related to these type of sequences which have many relations with 
    some combinatorial problems (such us the partition generating function, etc.).

    There are many objects related with these type of sequences and matching them
    into the framework of :mod:`pseries_basis` is crucial to extend the algorithms
    implemented in :arxiv:`2202.05550`.
'''

from .qbasis import *
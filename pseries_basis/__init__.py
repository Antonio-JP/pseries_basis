r'''
    **Inverse Zeibelger Problem in Sage**
    ########################################################

    ``pseries_basis`` is a `SageMath <https://www.sagemath.org>`_ package that allow to transform linear operators to compute definite-sum solutions of 
    differential or recurrence equations.

    This package is based on the work of the following research articles:

    * A. Jiménez-Pastor, M. Petkovšek: *The factorial-basis method for finding definite-sum solutions of linear recurrences with polynomial coefficients*. 
      `arXiv:2202.05550 <https://arxiv.org/abs/2202.05550>`_ (under revision in *Journal of Symbolic Computation*).

    **Some information about the module**
    ********************************************************
    - *Author*: Antonio Jiménez-Pastor
    - *License*: GNU Public License v3.0
    - *Home page*: :git:`Antonio-JP/pseries_basis`
    - *Documentation*: `<https://antonio-jp.github.io/pseries_basis/>`_
    - *Online demo*: `On Binder <https://mybinder.org/v2/gh/Antonio-JP/pseries_basis/master?labpath=notebooks%2Fpaper_examples.ipynb>`_

    **Main use-case**
    *********************************************************

    Let `\mathbb{K}` be a computable field and consider `\mathbb{K}[[x]]` its ring of formal power series. The problem of creative telescoping can be stated as follows: 
    let `F(x,n)` be a function. Is there any linear operator that annihilates the sum `\sum_n F(x,n)`?

    Solving this problem is equivalent to find an operator `L` acting only on `x` and a function `G(x,n)` (called a *certificate*) such that:

    .. MATH::

        L \cdot F(x,n) = G(x, n+1) - G(x,n),

    since once we have this *telescoping equation* for `F(x,n)` we can then sum-up w.r.t. `n` obtaining the equation

    .. MATH::
        
        L \cdot \sum_n F(x,n) = 0.``

    There are many studies with respect to this problem and, in fact, it has been solved in many cases. This package, however, tries to solved a somehow *inverse* problem:

    **Problem:** let `L` be a linear operator and `K(x,n)` be a kernel. Compute a new operator `\tilde{L}` such that for any solution of the form 
    `f(x) = \sum_n a_nK(x,n)` to `L \cdot f(x) = 0`, we have 

    .. MATH::
        \tilde{L} \cdot a_n = 0.``

    This is a partial solution to the more general case of **Inverse Zeilberger Problem**:

    **Inverse Zeilberger Problem:** let `L` be a linear operator. Find all the possible solutions that can be express as a definite-sum in such a way that Zeilberger 
    algorithm will succeed. 

    **Installation**
    #########################################################

    This package can be installed, used, modified and distributed freely under the conditions of the 
    `GNU General Public License v3 <https://www.gnu.org/licenses/gpl-3.0.html>`_.

    There are two different ways of installing the package into your SageMath distribution:

    **Install from source code**
    **********************************************************

    The package can be obtained from the public git repository on GitHub:
    * from the repository: :git:`Antonio-JP/pseries_basis` for the webpage view,
    * or by cloning the repository using https: :git:`Antonio-JP/pseries_basis.git`,
    * or downloading the latest version: :git:`Antonio-JP/pseries_basis/archive/master.zip`.

    After cloning or downloading the source code, you may install it by running the following command line from the main folder of the repository::

        $ make install

    **Install via pip**
    **********************************************************

    Another option to install this package is to use the pip functionality within SageMath. This will install the latest stable version of the package and will 
    take care of any dependencies that may be required.

    To install it via pip, run the following in a terminal where ``sage`` can be executed::

        $ sage -pip install [--user] git+https://github.com/Antonio-JP/pseries_basis.git

    The optional argument *--user* allows to install the package just for the current user instead of a global installation.

    **Loading the package**
    ***********************************************************

    Once installed, the full functionality of the package can be used after importing it with the command::

        sage: from pseries_basis import *

    **Examples of use**
    ***********************************************************

    This package is based on the concept of *compatibility* of a basis with a linear operator (see the main `paper <https://arxiv.org/abs/2202.05550>`_ for a proper definition). 
    The bases can be obtained in several ways and they usually include a set of operator they are compatible with. For example, consider the binomial basis 
    `P_n(x) = \binom{x}{n}`. It is well known that for `n\in\mathbb{N}`, `P_n(x)` is a polynomial of degree `n`. 
    We can create this basis easily::

        sage: B = BinomialBasis()
        sage: B
        Binomial basis (x) choose n
        sage: B[:3]
        [1, x, 1/2*x^2 - 1/2*x]

    We can also create basis using the idea of a *falling factorial*::

        sage: F = FallingBasis(1,0,1)
        sage: F
        Falling Factorial Basis (1, x, x(x-1),...)
        sage: F[:3]
        [1, x, x^2 - x]
        sage: [F.root_sequence()(i) for i in range(10)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    There are plenty of methods to check compatibility with a basis. We refer to the `documentation <https://antonio-jp.github.io/pseries_basis/>`_ for further information.

    **Dependencies**
    ***********************************************************

    This package has been developed on top of `SageMath`_ and depends on the following packages:

    * ``ore_algebra`` (:git:`mkauers/ore_algebra>`): developed by `M. Kauers <http://www.kauers.de/>`_ and `M. Mezzarobba <http://marc.mezzarobba.net/>`_.
    * ``dd_functions`` (:git:`Antonio-JP/dd_functions`): developed by `A. Jiménez-Pastor <https://scholar.google.com/citations?user=1gq-jy4AAAAJ&hl=es>`_.

    **Package under active development**

    This package is still under an active development and further features will be included in future version of the code. This means that several bugs may exist or appear. 
    We would thank anyone that, after detecting any error, would post it in the issues page (:git:`Antonio-JP/pseries_basis/issues`) of the repository 
    using the label bug (:git:`github/docs/labels/bug`).

    Moreover, any requested feature can be post in the issues page (:git:`Antonio-JP/pseries_basis/issues`) of the repository using the label 
    enhancement (:git:`github/docs/labels/enhancement`).

    **Acknowledgements**
    ***********************************************************

    This package has been developed with the financial support of the following institutions:

    * The Austrian Science Fund (FWF): by W1214-N15, project DK15.
    * The Oberösterreich region: by the Innovatives OÖ-2010 plus program.
    * The Île-de-France region: by the project "XOR".
    * The Poul Due Jensen Foundation: grant 883901.
'''
import logging, sys

## Configuring logger for this package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(f"{__name__}.log")
ch = logging.StreamHandler(sys.stderr)
fh.setFormatter(formatter); ch.setFormatter(formatter)
logger.addHandler(fh); logger.addHandler(ch)
logger.propagate = False
# logger.setLevel(logging.DEBUG) # setting up a level for logging


from .misc import *
from .order import *
from .psbasis import *
from .polynomial import *
from .qbasis import *
from .sequences import *

# fixing imports that were destroyed from these
from sage.all import factorial, order # pylint: disable=unused-import

def pseries_bases_version():
    import pkg_resources
    return pkg_resources.get_distribution('pseries_basis').version
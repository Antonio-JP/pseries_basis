
# **Inverse Zeibleger Problem in Sage** 

`psbasis` is a Sage package developed in the research funded by the Austrian Science Fund  (FWF): W1214-N15, project DK15.

This package allows Sage users create Power Series basis for $K[[x]]$ for a fixed field $K$. The main functionality of this package is to provide the user methods to transform difference/differential equations into recurrence relations assuming a expansion on the Power Series basis we have created. This is related with the Inverse Zeibelger Problem studied by Marko Petkovšek [in this paper](https://arxiv.org/abs/1804.02964).

This is a experimenting package and it is not intended to be pushed into any Sage distribution. 

## **1. Installing the package**

This package can be obtained after asking via email to the author [(Antonio Jimenez-Pastor)](ajpastor@risc.uni-linz.ac.at). One obtained, the user can run the command `make install` in the appropriate folder to install the package `psbasis` into your Sage distribution.

## **2. Using the package**
Now that the package is installed, we can start using it importing the appropriate package:

`from psbasis import *`

Now several objects like PowerBasis, BinomialBasis and similar will be available with all the corresponding functionality.

## **3. Extra requirements**
For a complete use of this package, it is **required** to have previously installed the *ore_algebra* package from M. Kauers and M. Mezzarobba (see their *git* repository [here](https://github.com/mkauers/ore_algebra)).

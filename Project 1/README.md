# Singular Value Decomposition & Spring/Mass System Solver

## Overview

This project involves computing the S.V.D. of a K matrix in order to solve the force-displacement equation of a spring/mass system.

### Scripts
- `main.py` - Contains command line interface & S.V.D./S.M.S. functions
- `test.py` - Contains test cases for S.V.D./S.M.S. functions

#### Functions
- The `SVD(A, eps=1e-10)` function computes the S.V.D. of an input matrix A (along with an optional epsilon value).
- The `SMS(masses, spring_constants, eps=1e-10, g=-9.81)` function solves the Ku=f equation for a spring mass system given lists of masses and spring constants (along with an optional epsilon value and acceleration due to gravity value).

## Usage

Users can either use the command-line interface for basic functionality (via `python ./main.py`), or call the functions directly in Python for more customizeability. Additional test cases for both S.V.D. and S.M.S. can be seen by executing the test file (via `python ./test.py`).

## Discussion

### Comparison to NumPy's S.V.D. Blackbox

NumPy's implementation of S.V.D. differs slightly from the form implemented here. For example, the singular values are returned as a list in NumPy's implementation, whereas they are returned as a matrix S in this implementation. Additionally, in the case that one or more of the singular values is within the epsilon tolerance (and therefore treated as zero), there may be some discrepancies in the U and VT matrices. However, both implementations multiply back to the original A matrix, and both compute the inverse of A to be the same as long as A is square (and throw errors otherwise).

There are several test cases available in the test script that illustrate the above details.

### Two Free Ends in a S.M.S.

The K matrix for a specific S.M.S. has dimensions of NxM, where M is the number of masses, and N is the number of springs. This is then combined with the displacement vector u, which is Mx3, to form the force vector f.

When two free ends are used as boundary conditions (or equivalently, when there are 1 fewer springs than masses), the system becomes underdetermined. This is because there are now N=M-1 equations and M unknowns.

The K matrix can still undergo S.V.D, however since there is no inverse, the vector f cannot be calculated exactly.

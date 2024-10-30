# Singular Value Decomposition & Spring/Mass System Solver

## Overview

This project involves computing the S.V.D. of a K matrix in order to solve the force-displacement equation of a spring/mass system.

### Scripts
- `main.py`

#### Functions
- The `SVD(A, eps=1e-10)` function computes the S.V.D. of an input matrix A (along with an optional epsilon value).
- The `SMS(masses, spring_constants, eps=1e-10)` function solves the Ku=f equation for a spring mass system given lists of masses and springs (along with an optional epsilon value).

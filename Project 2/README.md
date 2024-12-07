# CG-FEM Solver for 1D Heat Equation

## Overview

This project involves approximating the 1D heat equation given an initial condition, forcing function, and Dirichlet boundary conditions.

### Scripts
- `main.py` - Contains solver and examples with various parameters

#### Functions
- The `get_M_K()` function computes the M and K matrices needed before the heat equation is solved.
- The `solve_heat_eqn()` function uses 1D Lagrange polynomials and Gaussian Quadrature to approximate a solution to the PDE over a set of timesteps.

## Usage

Users can either use the command-line interface for basic functionality (via `python ./main.py`), or call the functions directly in Python for more customizeability.

## Discussion

Discussion and data analysis for this project is located in the `Writeup.pdf` file.
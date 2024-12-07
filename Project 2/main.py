# Written by Jack Rowe

import logging
import numpy as np
import matplotlib.pyplot as plt

# Quadrature constants
gauss_q = [-1/np.sqrt(3), 1/np.sqrt(3)]
gauss_w = [1,1]

# Basis function
def phi(xi: float) -> np.ndarray:
    return np.array([(1-xi)/2, (1+xi)/2])

# Basis function spatial derivative
def dphi(xi: float) -> np.ndarray:
    return np.array([-0.5, 0.5])

# Original function
def f(x: float, t: float) -> float:
    return (np.pi**2 -1) * np.exp(-t) * np.sin(np.pi * x)

# Initial condition
def u0(x: float) -> float:
    return np.sin(np.pi * x)

# Analytical solution
def af(x: float, t: float) -> float:
    return np.exp(-t) * np.sin(np.pi * x)

def get_M_K(N: int, x: np.ndarray) -> [np.ndarray, np.ndarray]:
    M = np.zeros((N,N))
    K = np.zeros((N,N))

    # Loop over all elements
    for e in range(N-1):
        nodes = [e, e+1]
        xe = x[nodes] # element endpoints
        he = xe[1]-xe[0] # element length

        # Local element matrices
        Me = np.zeros((2,2))
        Ke = np.zeros((2,2))

        # GQ Integration over parent space x->xi (q's are already in xi space)
        for q, w in zip(gauss_q, gauss_w):
            phi_vals = phi(q)
            dphi_vals = dphi(q)

            for i in range(2):
                for j in range(2):
                    Me[i,j] += w * phi_vals[i] * phi_vals[j] * (he/2)
                    Ke[i,j] += w * dphi_vals[i] * dphi_vals[j] * (2/he)

        # Global matrices assembly
        for i in range(2):
            for j in range(2):
                M[nodes[i], nodes[j]] += Me[i,j]
                K[nodes[i], nodes[j]] += Ke[i,j]

    return M,K

def solve_heat_eqn(N: int, x: np.ndarray, M: np.ndarray, K: np.ndarray, b: [float, float], Ti: float, Tf: float, dt: float, mode: str='fe') -> np.ndarray:

    # make sure dt is valid for Ti and Tf
    _nt = (Tf-Ti)/dt
    nt = int(_nt)
    if (nt != _nt):
        logging.error(f"(dt={dt}) Please use a dt value that evenly divides (Tf-Ti={Tf-Ti})")
        exit()

    M_inv = np.linalg.inv(M)

    u = u0(x) # Initial condition

    # Loop over all timesteps
    for n in range(1, nt+1):
        ctime = Ti + n * dt # current time
        F = np.zeros(N) # RHS vector

        # Loop over all elements
        for e in range(N-1):
            nodes = [e, e+1]
            xe = x[nodes] # element endpoints
            he = xe[1]-xe[0] # element length

            # Local RHS vector
            Fe = np.zeros(2)

            # GQ Integration over parent space x->xi (q's are already in xi space)
            for q, w in zip(gauss_q, gauss_w):
                phi_vals = phi(q)
                x_val = ((1-q) * xe[0] / 2) + ((1+q) * xe[1] / 2) # map back to physical space to calculate f at these points

                f_val = f(x_val, ctime)
                Fe += w * f_val * phi_vals * (he/2)

            # Global F assembly
            for i in range(2):
                F[nodes[i]] += Fe[i]

        # Forward Euler
        if (mode == 'fe'):
            u = u + dt * np.dot(M_inv,
                                F - np.dot(K,u))
        # Backward Euler
        elif (mode == 'be'):
            u = np.dot(np.linalg.inv(M + dt * K),
                       (F * dt + np.dot(M, u)))

        else:
            logging.error(f'Unkown mode {mode}')
            exit()

        # Enforce boundary conditions
        u[0] = b[0]
        u[-1] = b[1]

    return u

def main():

    # ======================================
    # Problem 2a

    N = 11 # Number of nodes
    x = np.linspace(0,1,N) # x space

    M,K = get_M_K(N,x) # Mass and stiffness matrices
    b = [0,0] # Boundary conditions

    Ti = 0.0 # Initial time
    Tf = 1.0 # Final time

    fig, ax = plt.subplots(1, 2, figsize=(8,8))

    dt = 1/551 # Timestep size
    u = solve_heat_eqn(N, x, M, K, b, Ti, Tf, dt) # Final solution
    logging.info(u)
    ax[0].plot(x, u, marker='o', label='Approx')
    ax[0].plot(x, af(x, 1), marker='o', label='Exact')
    ax[0].legend()
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('u')
    ax[0].set_title(f'dt=1/551')

    dt = 1/570
    u = solve_heat_eqn(N, x, M, K, b, Ti, Tf, dt)
    logging.info(u)
    ax[1].plot(x, u, marker='o', label='Approx')
    ax[1].plot(x, af(x, 1), marker='o', label='Exact')
    ax[1].legend()
    ax[1].set_xlabel('x')
    ax[1].set_title(f'dt=1/570')

    plt.suptitle("Approx vs. Exact Solutions for F.E.")
    plt.show()

    # ======================================
    # Problem 2b (sorry 3d sucks to get working sometimes)

    dt_denoms = [v for v in range(550, 620)]
    X, Y = np.meshgrid(x, dt_denoms)
    Z = np.zeros_like(X)

    for i, dtd in enumerate(dt_denoms):
        u = solve_heat_eqn(N, x, M, K, b, Ti, Tf, 1/dtd)
        Z[i, :] = u

    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    mask = (Z_flat >= -2) & (Z_flat <= 2)

    X_flat = X_flat[mask]
    Y_flat = Y_flat[mask]
    Z_flat = Z_flat[mask]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='viridis', marker='o')

    ax.set_title('Final u vector vs. x and dt')
    ax.set_xlabel('x')
    ax.set_ylabel('dt')
    ax.set_zlabel('u')
    ax.set_zlim(-2,2)
    plt.show()

    # ======================================
    # Problem 2c

    N_vals = [v*2+1 for v in range(1, 6)]

    fig, ax = plt.subplots(1, len(N_vals), figsize=(8,8))
    for i, Nv in enumerate(N_vals):
        x = np.linspace(0, 1, Nv)
        M, K = get_M_K(Nv, x)
        u = solve_heat_eqn(Nv, x, M, K, b, Ti, Tf, dt)

        ax[i].plot(x, u, marker='o')
        ax[i].set_title(f'N = {Nv}')
        ax[i].set_ylim(0,0.5)

    ax[len(N_vals)//2].set_xlabel('x')
    ax[0].set_ylabel('u')

    plt.show()

    # ======================================
    # Problem 3a

    N = 11
    x = np.linspace(0,1,N)

    M,K = get_M_K(N,x)

    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    dt = 1/570
    u = solve_heat_eqn(N, x, M, K, b, Ti, Tf, dt, mode='be')
    logging.info(u)

    ax.plot(x, u, marker='o', label='Approx')
    ax.plot(x, af(x, 1), marker='o', label='Exact')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Approx vs. Exact Solutions for B.E.')
    ax.set_title(f'dt=1/570')

    plt.show()

    # ======================================
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

# Written by Jack Rowe
# October 2024

import numpy as np
import logging

def SVD(A: np.ndarray, eps: float=1e-10) -> [[np.ndarray, np.ndarray, np.ndarray], float, np.ndarray]:

    logging.info(f'Running SVD...')

    # (1) Find Singular Values:
    # - get eigenvalues of A^TA-LI
    # - sort in descending order
    # - construct S using sqrt(l)

    eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
    i = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[i].real
    eigenvectors = eigenvectors[:, i].real
    logging.info(f'Eigenvalues:\n{eigenvalues}')

    singular_values = np.sqrt(eigenvalues[abs(eigenvalues) > eps]).real
    logging.info(f'Singular Values:\n{singular_values}')

    S = np.zeros((A.shape[0], A.shape[1]), dtype=float)
    for i in range(len(singular_values)):
        S[i,i] = singular_values[i]
    logging.info(f'S:\n{S}')

    # (2) Find V:
    # - get eigenvectors (available from step 1)
    # - store in columns of V

    V = eigenvectors
    logging.info(f'V:\n{V}')

    # (3) Find U:
    # - back-solve using U=AVS^-1
    # (easy math for S^-1 since its diagonal)

    S_inv = np.zeros((S.shape[1], S.shape[0]), dtype=float)
    for i in range(len(singular_values)):
        S_inv[i, i] = 1 / singular_values[i]

    U = A @ V @ S_inv
    logging.info(f'U:\n{U}')

    # (4) Calculate condition number:

    k = max(singular_values) / min(singular_values)
    logging.info(f'k:\n{k:.2f}')

    # (5) Calculate inverse of A:

    A_inv = np.zeros_like(A, dtype=float)
    if (A_inv.shape[0] != A_inv.shape[1]):
        logging.error(f'Matrix A is non-square (not invertible)')
    elif np.any(abs(eigenvalues) < eps):
        logging.error(f'Matrix A has singular value close to 0 (not invertible)')
    else:
        A_inv = V @ S_inv @ U.T
        logging.info(f'A_inv:\n{A_inv}')

    return [U,S,V.T], k, A_inv

def SMS(masses: list, spring_constants: list, eps: float=1e-10, g: float=-9.81) -> [list, list, list]:

    logging.info(f'Running SMS...')

    # FREE/FREE if n_springs = n_masses - 1 !!! NOT INVERTIBLE !!!
    # FREE/FIXED if n_springs = n_masses
    # FIXED/FIXED if n_springs = n_masses + 1

    n_masses = len(masses)
    n_springs = len(spring_constants)

    if n_masses > n_springs:
        logging.error(f'System is rank deficient (n_masses > n_springs); K matrix will not be invertible. Proceeding anyway.')

    # (1) Construct A
    # e(n_springs*1) = A(n_springs*n_masses)u(n_masses*1)
    A = np.zeros((n_springs, n_masses))
    for i in range(min(n_masses,n_springs)):
        A[i,i] = 1
    for i in range(1,n_springs):
        A[i,i-1] = -1
    logging.info(f'A:\n{A}')

    # (2) Construct C
    # C = diagonal matrix of 1/k_i
    C = np.zeros((n_springs, n_springs))
    for i in range(n_springs):
        C[i,i] = 1 / spring_constants[i]
    logging.info(f'C:\n{C}')

    # (3) Construct K
    # K = A^TCA
    K = A.T @ C @ A
    logging.info(f'K:\n{K}')

    # (4) Run SVD on K
    logging.info('====================')
    USVT, k, K_inv = SVD(K)
    logging.info('====================')

    # (5) Solve Ku=f
    # u = K^-1f
    # assuming gravity as external force
    f = [m * g for m in masses]
    logging.info(f'Forces:\n{f}')
    u = K_inv @ f
    logging.info(f'Displacements:\n{u}')

    # (6) Solve e=Au & w=Ce
    e = A @ u
    w = C @ e
    logging.info(f'Elongations:\n{e}')
    logging.info(f'Internal Stresses:\n{w}')

    return u,e,w

def main():

    n_masses = int(input("Enter the number of masses (>=1): "))
    if (n_masses < 1):
        print("Invalid number of masses")
        exit()
    masses = [0] * n_masses
    for i in range(n_masses):
        masses[i] = float(input(f"Enter mass {i+1}/{n_masses}: "))

    boundary_condition = int(input("Select the boundary condition (0 for FIXED/FREE, 1 for FIXED,FIXED): "))
    if (boundary_condition != 0 and boundary_condition != 1):
        print("Invalid boundary condition")
        exit()

    n_springs = n_masses+boundary_condition
    spring_constants = [0] * n_springs
    for i in range(n_springs):
        spring_constants[i] = float(input(f"Enter spring constant {i+1}/{n_springs}: "))

    SMS(masses, spring_constants)

    return

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    logging.basicConfig(level=logging.WARNING)
    main()
